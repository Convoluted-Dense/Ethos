import os
import sys
import json
import re
import time
import math
import traceback
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

# Fix Windows console encoding for Unicode/emoji output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = "diffused_3_log.json"
BATCH_SIZE = 3  # We want 1 prompt applied to 3 images

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PIPELINE (Realistic Vision + LCM + Canny)
# ─────────────────────────────────────────────────────────────────────────────
def load_controlnet_pipeline(workspace_dir: Path):
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler
    print(f"\n[SD] Loading ControlNet pipeline...")
    
    controlnet_path = workspace_dir / "try" / "stable-diffusion-webui-forge" / "models" / "ControlNet" / "control_v11p_sd15_canny.pth"
    model_path = workspace_dir / "try" / "stable-diffusion-webui-forge" / "models" / "Stable-diffusion" / "realisticVisionV60B1_v51VAE.safetensors"
    lora_path = workspace_dir / "try" / "stable-diffusion-webui-forge" / "models" / "Lora" / "lcm-lora-sdv1-5.safetensors"

    if not controlnet_path.exists(): raise FileNotFoundError(f"Missing {controlnet_path}")
    if not model_path.exists(): raise FileNotFoundError(f"Missing {model_path}")
    if not lora_path.exists(): raise FileNotFoundError(f"Missing {lora_path}")

    controlnet = ControlNetModel.from_single_file(
        str(controlnet_path), 
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        str(model_path),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe.load_lora_weights(str(lora_path))
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.to("cuda")
    # pipe.enable_attention_slicing() # Removed to speed up generation!
    pipe.set_progress_bar_config(disable=True)
    print(f"   [OK] ControlNet pipeline ready\n")
    return pipe

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir
    
    SRC_DIR = workspace_dir / "dataset" / "img"
    OUT_DIR = workspace_dir / "dataset" / "img_diffused"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_images = sorted([f for f in SRC_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    total_images = len(all_images)
    print(f"\n[+] Found {total_images} source images in {SRC_DIR}")

    # Load existing log
    log = {}
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                log = json.load(f)
            print(f"[+] Loaded existing log with {len(log)} entries")
        except Exception:
            pass

    # Filter out already diffused images
    to_process = [f for f in all_images if not (OUT_DIR / f.name).exists()]

    print(f"[+] Will generate: {len(to_process)} images\n")
    if to_process and len(to_process) < len(all_images):
        first_idx = all_images.index(to_process[0])
        print(f"[+] Resuming from image index {first_idx} ({to_process[0].name})\n")

    # Load 2500 Earth prompts from text file
    EARTH_PROMPTS = []
    try:
        with open(workspace_dir / "try" / "2500_prompts.txt", "r", encoding="utf-8") as f:
            EARTH_PROMPTS = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[!] Could not load 2500_prompts.txt: {e}")
        
    if not EARTH_PROMPTS:
        print("[!] Falling back to default prompt.")
        EARTH_PROMPTS = ["Dashcam footage, asphalt highway road, sunny day, clear blue sky, green fields, trees, photorealistic, 8k resolution, raw footage"]

    try:
        with open(workspace_dir / "try" / "negative_prompt.txt", "r", encoding="utf-8") as f:
            neg_prompt = f.read().strip()
    except Exception:
        neg_prompt = "cartoon, animated, drawn, rendered, 3d, fake"
        
    print(f"[+] Loaded {len(EARTH_PROMPTS)} Earth prompts for rotation.")

    pipe = load_controlnet_pipeline(workspace_dir)

    start_time = time.time()
    errors = 0
    generated = 0

    # Process in batches of 2
    num_batches = math.ceil(len(to_process) / BATCH_SIZE)
    pbar = tqdm(range(num_batches), desc="Generating", unit="batch")

    for i in pbar:
        batch_files = to_process[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        
        # Get absolute index in all_images to cycle prompts every 10 images
        try:
            abs_idx = all_images.index(batch_files[0])
        except ValueError:
            abs_idx = i * BATCH_SIZE
        
        prompt_idx = (abs_idx // 10) % len(EARTH_PROMPTS)
        pos_prompt = EARTH_PROMPTS[prompt_idx]
        scenario = f"Dynamic Prompt {prompt_idx}"
        
        pbar.set_postfix({"prompt_idx": prompt_idx, "sz": len(batch_files)})
        
        # 2. Process images
        canny_images = []
        base_images = []
        original_sizes = []
        for img_path in batch_files:
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            original_sizes.append((w, h))
            
            # Extract base image and Canny
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512))
            base_images.append(Image.fromarray(img_resized))
            
            # Apply Gaussian Blur (3x3 kernel) before Canny
            img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
            edges = cv2.Canny(img_blurred, 100, 200)
            edges = np.stack([edges, edges, edges], axis=-1)
            canny_images.append(Image.fromarray(edges))
            
        if not canny_images:
            continue
            
        # 3. Diffusion setup
        seed = 8675309
        generators = [torch.Generator(device="cpu").manual_seed(seed) for _ in range(len(batch_files))]
        
        batch_prompts = [pos_prompt] * len(canny_images)
        batch_neg_prompts = [neg_prompt] * len(canny_images)
        
        try:
            with torch.inference_mode():
                results = pipe(
                    prompt=batch_prompts,
                    negative_prompt=batch_neg_prompts,
                    image=base_images,
                    control_image=canny_images,
                    num_inference_steps=4,
                    strength=0.90,
                    guidance_scale=1.0,
                    generator=generators
                ).images
                
            # 4. Save results and log
            for idx, (img_path, orig_size) in enumerate(zip(batch_files, original_sizes)):
                res_img_np = np.array(results[idx])
                res_img_bgr = cv2.cvtColor(res_img_np, cv2.COLOR_RGB2BGR)
                res_img_resized = cv2.resize(res_img_bgr, orig_size)
                
                out_path = OUT_DIR / img_path.name
                cv2.imwrite(str(out_path), res_img_resized)
                
                log[img_path.name] = {
                    "scenario": scenario,
                    "positive": pos_prompt,
                    "negative": neg_prompt,
                    "seed": seed,
                }
                generated += 1
                
            # Save log periodically
            with open(LOG_FILE, "w") as f:
                json.dump(log, f, indent=4)
                
        except Exception as e:
            errors += 1
            print(f"\n[!] Error on batch {i}: {e}")
            traceback.print_exc()
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  DONE! Generated: {generated} images in {OUT_DIR}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Errors: {errors}")
    print(f"  Log saved to: {LOG_FILE}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
