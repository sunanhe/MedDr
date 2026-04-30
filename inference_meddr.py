import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
from src.train.dataset import build_transform
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer

pcam200_instruction_prompt = '''
You are a helpful medical assistant. 
Your task is disease diagnosis. 
You are given a Pathology image and are required to decide if the case is normal or tumor. 
You should answer with normal or tumor.
'''.replace('\n', '').replace('    ', ' ')

mrg_instruction_prompt = '''
You are a helpful medical assistant. 
Your task is report generation. 
You are given a chest x-ray image and you are required to generate a summary report about the image.
'''.replace('\n', '').replace('    ', ' ')

pmcvqa_instruction_prompt = '''
You are a helpful medical assistant.
You are require to answer the question based on the medical image.
The question is {}.
'''.replace('\n', '').replace('    ', ' ')

image_root = "/scratch/lsmodhealth/medical"

ds_collections = {
    'retoct': {
        'train': '',
        'test': 'RetOCT_CLS_Test.jsonl',
        'max_new_tokens': 20,
    },
    "iumrg": {
        'train': '',
        'test': 'IU_MRG_R2G_Test.jsonl',
        'max_new_tokens': 200,
    },
    "pmcvqa": {
        'train': '',
        'test': 'PMC-VQA_VQA_Test.jsonl',
        'max_new_tokens': 200,
    },
}

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return pixel_values, questions, question_ids, annotations

class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=224, pad2square=False):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data.get('question', None), data['question_id'], data.get('answer', None)
        image = Image.open(os.path.join(image_root, image)).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0)
        if len(self.prompt) != 0:
            question = self.prompt.format(question)
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation
        }


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

def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        
        if 'pcam200' in ds_name:
            input_prompt = pcam200_instruction_prompt
        elif "pmcvqa" in ds_name:
            input_prompt = pmcvqa_instruction_prompt
        elif 'iumrg' in ds_name:
            input_prompt = mrg_instruction_prompt
        
        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            input_size=image_size,
            pad2square=pad2square
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        
        if torch.distributed.get_rank() == 0:
            dataloader = tqdm(dataloader)

        for _, (pixel_values, questions, question_ids, annotations) in enumerate(dataloader):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                length_penalty=1,
                do_sample=False,
            )
            with torch.no_grad():
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    print_out=False
                )
            answers = [pred]

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                outputs.append({
                    'question': question,
                    'question_id': question_id,
                    'answer': answer,
                    'annotation': annotation,
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
    parser.add_argument('--datasets', type=str,
                        default='retoct, iumrg, pmcvqa')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
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