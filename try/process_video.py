import argparse
import base64
import cv2
import requests
import os
import time

def encode_cv2_image_to_base64(cv2_img):
    # Encode image to jpg and then to base64
    _, buffer = cv2.imencode('.jpg', cv2_img)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Process video frames with Stable Diffusion API")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_folder", help="Path to save the generated frames")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video opened. Total frames: {total_frames}, FPS: {fps}")

    url = "http://127.0.0.1:7860/sdapi/v1/img2img"
    
    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue

        print(f"Processing frame {frame_idx}/{total_frames}...")
        
        # Resize frame to 512x512 for optimal speed and SD1.5 performance
        frame_resized = cv2.resize(frame, (512, 512))
        encoded_image = encode_cv2_image_to_base64(frame_resized)

        # Optimized payload for fast LCM + Canny generation via img2img
        payload = {
            "prompt": "Dashcam footage, driving on a snowy road, heavy snowfall, extremely cold weather, icy, winter, overcast sky, highly detailed, photorealistic, raw footage, dirty windshield <lora:lcm-lora-sdv1-5:1>",
            "negative_prompt": "cartoon, animated, drawn, rendered, 3d, fake, clean render, studio lighting, sunny, clear sky, summer",
            "init_images": [encoded_image],
            "denoising_strength": 0.65,
            "steps": 4,
            "sampler_name": "LCM",
            "cfg_scale": 1.0,
            "seed": 8675309,
            "width": 512,
            "height": 512,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module": "canny",
                            "model": "control_v11p_sd15_canny [d14c016b]",
                            "weight": 1.0,
                            "input_image": encoded_image,
                            "image": encoded_image,
                            "pixel_perfect": True,
                            "control_mode": "Balanced"
                        }
                    ]
                }
            }
        }

        start_time = time.time()
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"API Error on frame {frame_idx}: {e}")
            frame_idx += 1
            continue

        r = response.json()
        
        if 'images' in r and len(r['images']) > 0:
            output_data = base64.b64decode(r['images'][0])
            out_path = os.path.join(args.output_folder, f"frame_{frame_idx:05d}.jpg")
            with open(out_path, 'wb') as f:
                f.write(output_data)
            
            gen_time = time.time() - start_time
            print(f"  Saved {out_path} (Generation took {gen_time:.2f}s)")
            processed_count += 1
        else:
            print(f"  Error: No image returned for frame {frame_idx}")

        frame_idx += 1

    cap.release()
    print(f"Finished! Processed {processed_count} frames and saved to {args.output_folder}")

if __name__ == "__main__":
    main()
