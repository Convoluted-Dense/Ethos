import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler

import argparse

print("Loading models into memory... (This takes a few seconds)")
start_load = time.time()

script_dir = os.path.dirname(os.path.abspath(__file__))
controlnet_path = os.path.join(script_dir, "stable-diffusion-webui-forge", "models", "ControlNet", "control_v11p_sd15_canny.pth")
model_path = os.path.join(script_dir, "stable-diffusion-webui-forge", "models", "Stable-diffusion", "realisticVisionV60B1_v51VAE.safetensors")
lora_path = os.path.join(script_dir, "stable-diffusion-webui-forge", "models", "Lora", "lcm-lora-sdv1-5.safetensors")

# 1. Load ControlNet
controlnet = ControlNetModel.from_single_file(
    controlnet_path, 
    torch_dtype=torch.float16,
    use_safetensors=True
)

# 2. Load Base Model
pipe = StableDiffusionControlNetPipeline.from_single_file(
    model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# 3. Load LCM LoRA
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()

# 4. Setup LCM Scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.to("cuda")
print(f"Models loaded in {time.time() - start_load:.2f} seconds!")

# --- Generation ---

parser = argparse.ArgumentParser(description="Generate Dashcam Augmentation using Diffusers locally")
parser.add_argument("input_image", help="Path to input image")
parser.add_argument("output_image", help="Path to save output image")
args = parser.parse_args()

input_image_path = args.input_image
output_image_path = args.output_image

print(f"Reading input image: {input_image_path}")
image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to typical SD size (e.g. 512x512) for speed
image = cv2.resize(image, (512, 512))

# Extract Canny Edges
low_threshold = 100
high_threshold = 200
edges = cv2.Canny(image, low_threshold, high_threshold)
edges = edges[:, :, None]
edges = np.concatenate([edges, edges, edges], axis=2)
canny_image = Image.fromarray(edges)

prompt = "dashcam footage, highly detailed, photorealistic, 8k, raw footage, dirty windshield, poor lighting, cctv"
negative_prompt = "cartoon, animated, drawn, rendered, 3d, fake"

print("Starting Generation...")
start_gen = time.time()

# Generate with exactly 6 steps for LCM
result = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    num_inference_steps=6,
    guidance_scale=1.0, # LCM requires 1.0 or 1.5 CFG
).images[0]

gen_time = time.time() - start_gen

result.save(output_image_path)
print(f"Generation Complete! Took exactly {gen_time:.3f} seconds.")
