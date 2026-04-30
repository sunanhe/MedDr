import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import sys
import torch
from src.train.dataset import build_transform
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer

cur_time = time.strftime('%y%m%d%H%M%S', time.localtime())
image_root = "data"

pcam200_instruction_prompt = '''
You are a helpful medical assistant.
Your task is disease diagnosis.
You are given a pathology image.
The possible diagnoses are: 'normal', 'tumor'.
The diagnoses of the most similar cases are ###RAD###.
The diagnoses of the specialist models are ###MOED###.
The response should only contain one option.
'''.replace('\n', ' ').replace('    ', ' ')

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return pixel_values, questions, question_ids, annotations


class RAGDataset(torch.utils.data.Dataset):

    def __init__(self, test_meta, prompt):
        self.test_meta = json.load(open(test_meta))
        self.prompt = prompt
        self.transform = build_transform(is_train=False, input_size=448, pad2square=False)

    def __len__(self):
        return len(self.test_meta)

    def __getitem__(self, idx):

        data = self.test_meta[idx]
        question = self.prompt
        question = question.replace("###RAD###", ",".join(data["rad"]))
        question = question.replace("###MOED###", ",".join(data["moed"]))
        image = data['image']
        annotation = data['label']
        image = Image.open(os.path.join(image_root, image)).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0)

        return {
            'question_id': idx,
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation,
        }

def evaluate_chat_model():
    random.seed(args.seed)

    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    for ds_name in args.datasets:

        if 'pcam200' in ds_name:
            input_prompt = pcam200_instruction_prompt
            test_meta = "data/pcam200_meta.json"
        
        dataset = RAGDataset(
            test_meta=test_meta,
            prompt=input_prompt,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        
        if torch.distributed.get_rank() == 0:
            dataloader = tqdm(dataloader)

        for _, (pixel_values, questions, question_ids, annotations) in enumerate(dataloader):

            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            with torch.no_grad():
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    print_out=False
                )
            answers = [pred]

            for ques, id, ans, anno in zip(questions, question_ids, answers, annotations):
                outputs.append({
                    'question': ques,
                    'question_id': id,
                    'answer': ans,
                    'annotation': anno,
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

        torch.distributed.barrier()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='pcam200')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    args.datasets = args.datasets.split(',')
    if torch.distributed.get_rank() == 0:
        print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)

    from src.model.internvl_chat import InternVLChatModel
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16).cuda().eval()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    pad2square = model.config.pad2square

    if torch.distributed.get_rank() == 0:
        print(f'[test] num_image_token: {model.num_image_token}')
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 30:
        args.num_beams = 1
        if torch.distributed.get_rank() == 0:
            print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        if torch.distributed.get_rank() == 0:
            print(f'[test] total_params: {total_params}B')
    if torch.distributed.get_rank() == 0:
        print(f'[test] image_size: {image_size}')
        print(f'[test] pad2square: {pad2square}')
        print(f'[test] template: {model.config.template}')

    evaluate_chat_model()