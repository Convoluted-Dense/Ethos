import os
import time
import argparse
import random
import json
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler

def process_batch(pipe, batch_frames, batch_canny, prompts, output_folder, start_idx, base_seed):
    """Runs the diffusion pipeline on a batch of images and saves them."""
    
    batch_size = len(batch_frames)
    # Give each frame in the batch a consistent noise seed
    generators = [torch.Generator(device="cuda").manual_seed(base_seed + i) for i in range(batch_size)]
    
    pos_prompts = [p["positive"] for p in prompts]
    neg_prompts = [p["negative"] for p in prompts]
    
    with torch.no_grad():
        results = pipe(
            prompt=pos_prompts,
            negative_prompt=neg_prompts,
            image=batch_frames,       # Base image for img2img
            control_image=batch_canny, # Canny map for ControlNet
            num_inference_steps=4,
            strength=0.85,
            guidance_scale=1.0,       # LCM uses 1.0 or 1.5
            generator=generators
        ).images

    # Save outputs
    for i, img in enumerate(results):
        out_path = os.path.join(output_folder, f"frame_{start_idx + i:05d}.jpg")
        img.save(out_path)
        print(f"  Saved {out_path} - Scene: {prompts[i]['scenario'][:60]}...")

def main():
    parser = argparse.ArgumentParser(description="Fast Native Video Processing using random prompts")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_folder", help="Path to output folder")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of frames to process simultaneously")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    print("Loading models into memory... (This takes a few seconds)")
    start_load = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to local models in try/stable-diffusion-webui-forge/
    forge_dir = os.path.join(script_dir, "try", "stable-diffusion-webui-forge")
    controlnet_path = os.path.join(forge_dir, "models", "ControlNet", "control_v11p_sd15_canny.pth")
    model_path = os.path.join(forge_dir, "models", "Stable-diffusion", "realisticVisionV60B1_v51VAE.safetensors")
    lora_path = os.path.join(forge_dir, "models", "Lora", "lcm-lora-sdv1-5.safetensors")

    if not os.path.exists(controlnet_path):
        raise FileNotFoundError(f"Cannot find ControlNet model at {controlnet_path}")

    # Load Models
    controlnet = ControlNetModel.from_single_file(
        controlnet_path, 
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # Use the Img2Img pipeline
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    print(f"Models loaded in {time.time() - start_load:.2f} seconds!")

    # Load pregenerated prompts
    prompts_file = os.path.join(script_dir, "pregenerated_prompts.json")
    with open(prompts_file, "r") as f:
        pregenerated_prompts = json.load(f)
    print(f"Loaded {len(pregenerated_prompts)} prompts from {prompts_file}")

    cap = cv2.VideoCapture(args.input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened. Total frames: {total_frames}")

    seed = random.randint(1000, 999999)

    batch_frames = []
    batch_canny = []
    batch_prompts = []
    frame_idx = 0
    processed_count = 0

    batch_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue

        # Resize for speed and stability
        frame = cv2.resize(frame, (512, 512))
        
        # Calculate Canny Map
        low_threshold, high_threshold = 100, 200
        edges = cv2.Canny(frame, low_threshold, high_threshold)
        edges = np.stack([edges, edges, edges], axis=-1)

        # Convert CV2 (BGR) to PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(Image.fromarray(frame_rgb))
        batch_canny.append(Image.fromarray(edges))
        
        # Pick random prompt
        random_prompt = random.choice(pregenerated_prompts)
        batch_prompts.append(random_prompt)

        if len(batch_frames) == args.batch_size:
            print(f"\nProcessing batch of {args.batch_size} frames (ending at frame {frame_idx}/{total_frames})...")
            process_batch(pipe, batch_frames, batch_canny, batch_prompts, args.output_folder, frame_idx - (args.batch_size * args.frame_skip) + args.frame_skip, seed + processed_count)
            
            gen_time = time.time() - batch_start_time
            print(f"  Batch complete in {gen_time:.2f}s ({gen_time / args.batch_size:.2f}s per frame)")
            
            processed_count += args.batch_size
            batch_frames = []
            batch_canny = []
            batch_prompts = []
            batch_start_time = time.time()

        frame_idx += 1

    # Process remaining frames
    if len(batch_frames) > 0:
        print(f"\nProcessing final batch of {len(batch_frames)} frames...")
        process_batch(pipe, batch_frames, batch_canny, batch_prompts, args.output_folder, frame_idx - (len(batch_frames) * args.frame_skip) + args.frame_skip, seed + processed_count)
        processed_count += len(batch_frames)

    cap.release()
    print(f"\nFinished! Processed {processed_count} frames and saved to {args.output_folder}.")

if __name__ == "__main__":
    main()
