import torch 
device = torch.device("cuda")

from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from utils_model import load_model_from_config 
import json

ldm_config = "/data/haoyuanliu/project/stable_signature/v1-inference.yaml"
ldm_ckpt = "/data/haoyuanliu/project/huggingface/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt"

print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
config = OmegaConf.load(f"{ldm_config}")
ldm_ae = load_model_from_config(config, ldm_ckpt)
ldm_aef = ldm_ae.first_stage_model
ldm_aef.eval()

# loading the fine-tuned decoder weights
state_dict = torch.load("/data/haoyuanliu/project/stable_signature/output/checkpoint_000.pth")
unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
print(unexpected_keys)
print("you should check that the decoder keys are correctly matched")

# loading the pipeline, and replacing the decode function of the pipe
model = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model).to("cuda:0")
pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))
ori_pipe = StableDiffusionPipeline.from_pretrained(model).to("cuda:2")

# img = pipe("the cat drinks water.").images[0]
# img.save("cat.png")


SAVE_PATH = "/data/haoyuanliu/project/stable_signature/wm_images"

with open("/data/haoyuanliu/project/SleeperMark/stage2/dataset/metadata_eval.jsonl", 'r') as file:
    for line in file:
        data = json.loads(line)
        index = data['index']
        if index >= 1250:
            break
        # if index <= 6:
        #     continue
        file_name = str(data['index']) + ".png"
        prompt = data['prompt']
        print(index, prompt)
        generator = torch.Generator(device="cpu").manual_seed(123)
        img = pipe(prompt, guidance_scale=7.5, generator=generator, safety_checker=None).images[0]
        img.save(f"{SAVE_PATH}/wm/{file_name}") 
        generator = torch.Generator(device="cpu").manual_seed(123)
        img = ori_pipe(prompt, guidance_scale=7.5, generator=generator, safety_checker=None).images[0]
        img.save(f"{SAVE_PATH}/origin/{file_name}")   