from PIL import Image

import torch
from transformers import LlamaTokenizer

from src.model.internvl_chat import InternVLChatModel
from src.dataset.transforms import build_transform

IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

model_path = "Sunanhe/MedDr_0401"

device = "cuda"

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = InternVLChatModel.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda().eval()

image_size = model.config.force_image_size or model.config.vision_config.image_size
pad2square = model.config.pad2square
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id

image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)

instruction_prompt = open("examples/instruction_prompt.txt").readlines()

for i in range(5):
    print("################# Case: ", i, "#################")
    image_path = f"examples/test{i+1}.jpg"
    image = Image.open(image_path).convert('RGB')
    image = image_processor(image).unsqueeze(0).to(device).to(torch.bfloat16)
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=image,
            question=instruction_prompt[i].strip(),
            generation_config=generation_config,
            print_out=False
        )
    print(response)

