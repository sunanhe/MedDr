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

"""
################# Case:  0 #################
Findings:
- The retinography image shows a significant amount of retinal hemorrhages scattered throughout the retina.
- The hemorrhages vary in size and shape, with some appearing as small spots and others as larger, more diffuse areas of bleeding.
- The retinal vessels appear to be dilated and tortuous, indicating increased blood flow and pressure within the vessels.
- There is a notable presence of exudates, which are small, yellowish deposits of protein and lipid material that have leaked from the blood vessels into the retinal tissue.

Impressions:
- The retinography image reveals a case of BRVO (Branch Retinal Vein Occlusion), which is a blockage of a branch of the retinal vein.
- The presence of retinal hemorrhages, dilated and tortuous vessels, and exudates are indicative of the underlying vascular pathology associated with BRVO.
- Further evaluation and management may be necessary to address the underlying cause of the BRVO and to prevent potential complications, such as macular edema or neovascularization.
################# Case:  1 #################
Findings:
- The image shows a close-up view of the colon, with a focus on the colon's inner lining.
- The colon appears to be inflamed, with visible redness and swelling.
- There are small white specks scattered throughout the colon's inner lining.

Impressions:
- The presence of inflammation and white specks in the colon's inner lining suggests a possible diagnosis of ulcerative colitis.
- Further testing and consultation with a gastroenterologist may be necessary to confirm the diagnosis and determine the appropriate treatment plan.
################# Case:  2 #################
the optic nerve
################# Case:  3 #################
no
################# Case:  4 #################
Osteophytes,Surgical implant
"""