import os
import time
import torch
import cv2
import math
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from tqdm import tqdm
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    
    input_dir = workspace_dir / "dataset" / "img"
    output_dir = workspace_dir / "dataset" / "img_diffused_2"
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models into memory... (This takes a few seconds)")
    start_load = time.time()

    controlnet_path = script_dir / "stable-diffusion-webui-forge" / "models" / "ControlNet" / "control_v11p_sd15_canny.pth"
    model_path = script_dir / "stable-diffusion-webui-forge" / "models" / "Stable-diffusion" / "realisticVisionV60B1_v51VAE.safetensors"
    lora_path = script_dir / "stable-diffusion-webui-forge" / "models" / "Lora" / "lcm-lora-sdv1-5.safetensors"

    if not controlnet_path.exists():
        print(f"Error: ControlNet model not found at {controlnet_path}")
        return
    if not model_path.exists():
        print(f"Error: Base model not found at {model_path}")
        return
    if not lora_path.exists():
        print(f"Error: LoRA model not found at {lora_path}")
        return

    # 1. Load ControlNet
    controlnet = ControlNetModel.from_single_file(
        str(controlnet_path), 
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # 2. Load Base Model
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        str(model_path),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # 3. Load LCM LoRA
    pipe.load_lora_weights(str(lora_path))
    pipe.fuse_lora()

    # 4. Setup LCM Scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Move to GPU
    pipe.to("cuda")
    
    # Memory optimization to prevent OOM on lower VRAM GPUs
    pipe.enable_attention_slicing()
    
    print(f"Models loaded in {time.time() - start_load:.2f} seconds!")

    # 50 predefined positive and negative prompt pairs for massive domain diversity
    PROMPT_PAIRS = [
        {
            "pos": "Dashcam view, blinding golden hour sun directly ahead, severe lens flare, long harsh black shadows cast across the road obscuring lane lines, high contrast, photorealistic, 8k",
            "neg": "cartoon, animated, drawn, rendered, 3d, fake, clear visibility"
        },
        {
            "pos": "Dashcam view, pitch black night, torrential rain absorbing headlights, zero visibility beyond a few meters, flooded black asphalt, terrifying darkness, photorealistic",
            "neg": "cartoon, daytime, bright, clear sky"
        },
        {
            "pos": "Dashcam view, driving on highway, windshield heavily shattered with deep spiderweb cracks distorting the view, bright daylight, harsh reflections on broken glass, photorealistic",
            "neg": "cartoon, clean windshield, clear view"
        },
        {
            "pos": "Dashcam view, lens completely covered in thick brown mud splatters and dirty water droplets, barely visible road ahead, heavy overcast, rural environment, photorealistic",
            "neg": "cartoon, clear lens, clean road"
        },
        {
            "pos": "Wet asphalt after rain, huge deep puddles perfectly reflecting the bright blue sky, creating optical illusions on the road surface that look like holes, high contrast, photorealistic, 8k",
            "neg": "cartoon, dry road, flat lighting"
        },
        {
            "pos": "Heavy snow blizzard, complete whiteout conditions, road entirely covered in snow, no visible lane markings, thick white fog, snow blowing across windshield, photorealistic",
            "neg": "cartoon, summer, green grass, clear sky"
        },
        {
            "pos": "Torrential downpour, windshield wipers moving fast causing motion blur, thick sheets of water cascading down the glass, severely distorted view of flooded road ahead, photorealistic",
            "neg": "cartoon, sunny, calm, dry"
        },
        {
            "pos": "Extreme dense white fog, visibility less than 5 meters, faint ghostly outlines of trees, grey washed-out road merging into fog, absolutely no contrast, photorealistic",
            "neg": "cartoon, high contrast, sunny, bright"
        },
        {
            "pos": "Driving through a dense forest at midday, extremely harsh dappled sunlight, chaotic sharp shadows of tree branches across the road, confusing high contrast patterns, photorealistic",
            "neg": "cartoon, flat lighting, clear open road"
        },
        {
            "pos": "Dashcam pointed towards bright midday sun, extreme overexposure, washed out colors, heavy chromatic aberration, lens flares rings, road completely lost in bright white light, photorealistic",
            "neg": "cartoon, perfectly exposed, balanced lighting"
        },
        {
            "pos": "Nighttime dashcam footage, pitch black rural road, blinded by intense bright oncoming high beam headlights, heavy lens flare, no streetlights, barely visible road edges, photorealistic",
            "neg": "cartoon, daytime, sun, clear visibility"
        },
        {
            "pos": "Night city drive, heavy rain, thousands of large water droplets on windshield refracting bright neon signs and traffic lights into glowing blurry orbs, chaotic colorful bokeh obscuring view, photorealistic",
            "neg": "cartoon, daytime, clear sky, dry"
        },
        {
            "pos": "Severe desert dust storm, haboob, thick orange/brown sand filling the air, road completely obscured by blowing sand, terrifying low visibility, apocalyptic lighting, photorealistic",
            "neg": "cartoon, clear air, forest, rain"
        },
        {
            "pos": "Winter morning, road covered in smooth black ice, low morning sun causing blinding specular glare bouncing off the icy road surface directly into the camera lens, photorealistic",
            "neg": "cartoon, dry asphalt, overcast"
        },
        {
            "pos": "Concrete bridge flooded with deep muddy water, water flowing rapidly across lanes, debris floating, overcast grey sky, confusing visual boundaries between road and river, photorealistic, 8k",
            "neg": "cartoon, clear dry bridge, sunny"
        },
        {
            "pos": "Driving fast past a row of trees at sunset, strobing light effect, rapid flickering between blinding sun and dark tree shadows across the road, high contrast, photorealistic",
            "neg": "cartoon, flat lighting, overcast"
        },
        {
            "pos": "Dashcam view, thick greasy smears across the windshield catching the sunlight, severely hazy and blurry view of the highway ahead, poor visibility, photorealistic",
            "neg": "cartoon, perfect clean glass, clear view"
        },
        {
            "pos": "Driving through a forest wildfire, thick grey and orange smoke completely obscuring the road ahead, glowing embers, terrifying low visibility, bright orange sky, photorealistic",
            "neg": "cartoon, clear sky, calm forest"
        },
        {
            "pos": "Nighttime city street, torrential rain, black asphalt completely submerged, reflecting thousands of confusing city lights, no visible road markings, chaotic reflections, photorealistic",
            "neg": "cartoon, daytime, dry street"
        },
        {
            "pos": "Rural dirt road completely covered in a fresh layer of unplowed white snow, bright overcast sky causing flat white lighting, zero contrast between road and snowbanks, photorealistic",
            "neg": "cartoon, summer, high contrast, green"
        },
        {
            "pos": "Dashcam view from inside a cold car, windshield heavily fogged up with thick condensation, hazy obscured view of a suburban street, soft blurry outlines, photorealistic",
            "neg": "cartoon, sharp focus, clear glass"
        },
        {
            "pos": "Driving away from the sun, car own massive black shadow projected forward covering the road, bright blinding light illuminating the surroundings but road in deep shadow, photorealistic",
            "neg": "cartoon, flat lighting, overcast"
        },
        {
            "pos": "Entering a pitch black tunnel, tunnel floor completely flooded with still dark water reflecting the entrance light, impossible to tell where water ends and floor begins, photorealistic",
            "neg": "cartoon, bright tunnel, dry floor"
        },
        {
            "pos": "Severe hail storm, road covered in white ice pellets, heavy rain, grey sky, chaotic messy visual, windshield getting pelted with heavy ice, photorealistic, raw dashcam",
            "neg": "cartoon, sunny, calm, dry road"
        },
        {
            "pos": "Old cracked highway, faded and completely missing white lane markings, patched grey and black asphalt creating false lines, bright flat midday lighting, photorealistic",
            "neg": "cartoon, perfect new road, sharp lines"
        },
        {
            "pos": "Afternoon sun reflecting intensely off a recently rained-on wet asphalt road, blinding white specular highlights on the road surface, high contrast, photorealistic",
            "neg": "cartoon, dry road, flat lighting"
        },
        {
            "pos": "Highway driving, windshield covered in dozens of large messy bug splatters and smears, obstructing the view of the road, bright sunny day, photorealistic, raw dashcam",
            "neg": "cartoon, clean windshield, clear view"
        },
        {
            "pos": "Following a truck on a dirt road, massive thick cloud of brown dust completely obscuring the vision ahead, low visibility, harsh sunlight, photorealistic",
            "neg": "cartoon, clear air, rain, green grass"
        },
        {
            "pos": "Heavy rain at night, hundreds of streetlights and car brake lights reflecting off deep puddles, visually confusing and overwhelming colorful reflections on the road, photorealistic",
            "neg": "cartoon, daytime, dry road"
        },
        {
            "pos": "Winter morning, windshield partially covered in thick white frost patterns, only a small cleared patch to see through, bright morning sun hitting the frost, photorealistic",
            "neg": "cartoon, summer, clear glass"
        },
        {
            "pos": "Approaching a city underpass completely flooded with deep dark water, water level reaching the concrete walls, heavy rain, dark shadows, photorealistic",
            "neg": "cartoon, dry underpass, sunny"
        },
        {
            "pos": "Driving directly into a huge red sunset, massive circular lens flare rings covering the entire image, washing out the road details, highly cinematic and difficult visibility, photorealistic",
            "neg": "cartoon, flat lighting, midday"
        },
        {
            "pos": "Tractor path in farm fields, heavily flooded with deep brown mud puddles, grass patch in center, extremely messy and confusing road boundaries, pouring rain, photorealistic",
            "neg": "cartoon, smooth highway, dry"
        },
        {
            "pos": "Clear blue sky after a snowstorm, road and landscape completely white, intense blinding sunlight reflecting off the snow, extreme brightness, low contrast details, photorealistic",
            "neg": "cartoon, summer, dark night"
        },
        {
            "pos": "A massive splash of water hits the windshield directly covering the camera lens in a sheet of distorted water, road ahead completely warped and blurry, photorealistic",
            "neg": "cartoon, dry glass, clear view"
        },
        {
            "pos": "Pitch black night, road covered in invisible black ice, subtle terrifying reflections of headlights on the icy asphalt, very low visibility, photorealistic",
            "neg": "cartoon, daytime, dry road, clear"
        },
        {
            "pos": "Industrial area, massive thick plume of black smoke blowing across the road, suddenly reducing visibility to zero, dark shadows, photorealistic",
            "neg": "cartoon, clear sky, clean air"
        },
        {
            "pos": "Sun shower, raining while the sun is brightly shining, thousands of raindrops catching the bright sunlight like diamonds, visually overwhelming glitter effect on the windshield, photorealistic",
            "neg": "cartoon, dry, overcast"
        },
        {
            "pos": "Nighttime urban road, broken streetlight flickering rapidly creating intense strobe effect, confusing shadows moving across the flooded road, photorealistic",
            "neg": "cartoon, daytime, steady light"
        },
        {
            "pos": "A large translucent plastic bag blown by the wind is stuck on the windshield, partially obscuring the view of the highway, wrinkled plastic texture catching light, photorealistic",
            "neg": "cartoon, clean windshield, clear view"
        },
        {
            "pos": "Broken asphalt road covered in hundreds of deep pothole puddles filled with muddy water, grey sky, visually chaotic and broken road surface, photorealistic",
            "neg": "cartoon, smooth new highway, dry"
        },
        {
            "pos": "Cheap dashcam lens effect, extreme chromatic aberration and color fringing on the edges of the frame, heavy distortion, bright daylight, photorealistic",
            "neg": "cartoon, perfect lens, sharp focus"
        },
        {
            "pos": "Night driving, heavy rain, windshield wipers dragging water causing long horizontal light smears from oncoming headlights, severely distorted vision, photorealistic",
            "neg": "cartoon, daytime, dry glass"
        },
        {
            "pos": "Overcast day with perfectly flat grey lighting, no shadows anywhere, grey road blends perfectly into grey sky and grey surroundings, zero contrast, photorealistic",
            "neg": "cartoon, high contrast, sunny, colorful"
        },
        {
            "pos": "Driving towards a lake at sunset, intense bright orange sun reflecting off both the water and the wet road surface simultaneously, completely blinding, photorealistic",
            "neg": "cartoon, overcast, night"
        },
        {
            "pos": "Road cutting through a dark swamp, thick green/grey fog rolling across the asphalt, mossy trees barely visible, very creepy and low visibility, photorealistic",
            "neg": "cartoon, sunny, clear desert"
        },
        {
            "pos": "Winter road with scattered patches of highly reflective ice, afternoon sun bouncing off the patches creating sharp blinding spots on the road surface, photorealistic",
            "neg": "cartoon, dry road, overcast"
        },
        {
            "pos": "Nighttime thunderstorm, split second of a massive lightning flash illuminating the entire flooded landscape in bright blue-white light, intense extreme contrast, photorealistic",
            "neg": "cartoon, daytime, steady lighting"
        },
        {
            "pos": "Winter road covered in thick brown/grey muddy snow slush, extremely messy road surface, no lane lines, dirty windshield, overcast sky, photorealistic",
            "neg": "cartoon, summer, clean dry highway"
        },
        {
            "pos": "Driving through a thick pine forest on a moonless night, headlights broken or very dim, almost absolute darkness, terrifyingly low visibility of the road, photorealistic",
            "neg": "cartoon, daytime, bright sun"
        }
    ]

    # Find and filter images
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    all_images.sort()
    
    # Check if we should overwrite existing images
    OVERWRITE = True # Set to True to overwrite existing files, False to resume/skip
    
    if OVERWRITE:
        to_process = all_images
    else:
        to_process = [f for f in all_images if not (output_dir / f).exists()]
    
    print(f"Total images found in dataset: {len(all_images)}")
    print(f"Already processed: {len(all_images) - len(to_process)}")
    print(f"Remaining to process: {len(to_process)}")

    if not to_process:
        print("All images are already processed! Exiting.")
        return

    batch_size = 6
    print(f"Starting batch generation with batch size = {batch_size}...")

    # Process in batches of 6
    for i in range(0, len(to_process), batch_size):
        print(f"\n--- Processing batch {i // batch_size + 1}/{math.ceil(len(to_process) / batch_size)} (indices {i} to {i + batch_size}) ---", flush=True)
        batch_files = to_process[i:i + batch_size]
        
        canny_images = []
        original_sizes = []
        valid_files = []
        
        for f in batch_files:
            in_path = input_dir / f
            img = cv2.imread(str(in_path))
            if img is None:
                print(f"[Warning] Could not read {in_path}. Skipping.", flush=True)
                continue
                
            h, w = img.shape[:2]
            original_sizes.append((w, h))
            valid_files.append(f)
            
            # Convert to RGB and resize to 512x512 for ControlNet Canny
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512))
            
            # Extract edges
            edges = cv2.Canny(img_resized, 100, 200)
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
            canny_image = Image.fromarray(edges)
            canny_images.append(canny_image)
            
        if not canny_images:
            print("No valid images in this batch. Skipping.", flush=True)
            continue
            
        current_batch_size = len(canny_images)
        batch_prompts = []
        batch_neg_prompts = []
        
        # Track prompts used in this batch for logging
        used_prompt_indices = []
        for filename in valid_files:
            global_idx = all_images.index(filename)
            prompt_idx = (global_idx // 10) % len(PROMPT_PAIRS)
            used_prompt_indices.append(prompt_idx)
            pair = PROMPT_PAIRS[prompt_idx]
            batch_prompts.append(pair["pos"])
            batch_neg_prompts.append(pair["neg"])
            
        print(f"Batch prompt indices: {used_prompt_indices} (cycling every 10 images)", flush=True)
        
        try:
            print(f"Running diffusion model for batch of {current_batch_size} images...", flush=True)
            with torch.inference_mode():
                results = pipe(
                    batch_prompts,
                    negative_prompt=batch_neg_prompts,
                    image=canny_images,
                    num_inference_steps=6,
                    guidance_scale=1.0
                ).images
                
            # Save results
            print("Saving generated images...", flush=True)
            for idx, (f, orig_size) in enumerate(zip(valid_files, original_sizes)):
                res_img = results[idx]
                
                # Convert back to BGR and resize to original dimensions (e.g. 1280x720)
                res_img_np = np.array(res_img)
                res_img_bgr = cv2.cvtColor(res_img_np, cv2.COLOR_RGB2BGR)
                res_img_resized = cv2.resize(res_img_bgr, orig_size)
                
                out_path = output_dir / f
                success = cv2.imwrite(str(out_path), res_img_resized)
                print(f"Saved: {out_path.name} (success={success})", flush=True)
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}", flush=True)
            # If CUDA OOM, clear cache and print advice
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("CUDA Out of Memory occurred. Try reducing the batch size in the script.", flush=True)
                break

    print("Dataset augmentation complete!", flush=True)

if __name__ == "__main__":
    main()
