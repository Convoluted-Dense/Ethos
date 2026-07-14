import argparse
import base64
import json
import io
import requests
from PIL import Image

def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Generate Dashcam Augmentation using Stable Diffusion API")
    parser.add_argument("input_image", help="Path to the input simulator image")
    parser.add_argument("output_image", help="Path to save the generated dashcam image")
    args = parser.parse_args()

    print(f"Reading input image: {args.input_image}...")
    
    # Read the image to get dimensions
    try:
        with Image.open(args.input_image) as img:
            width, height = img.size
            # Dimensions must be multiples of 8
            width = (width // 8) * 8
            height = (height // 8) * 8
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    encoded_image = image_to_base64(args.input_image)

    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    
    # The optimized payload
    payload = {
        "prompt": "CCTV footage, cheap dashboard camera video, low quality 1080p, heavily compressed video frame, dirty windshield with bug splatters and smudges, extreme sun glare, chromatic aberration, dash reflection on glass, timestamp text in corner, grainy, underexposed, raw unedited dashcam, motion blur, bad lighting, real life driving <lora:lcm-lora-sdv1-5:1>",
        "negative_prompt": "beautiful, cinematic, professional photography, high resolution, 4k, 8k, sharp focus, clean glass, highly detailed, vivid colors, vibrant, studio lighting, clear, pristine, video game, 3d render, unreal engine",
        "steps": 6,
        "sampler_name": "LCM",
        "cfg_scale": 1.0,
        "width": width,
        "height": height,
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "enabled": True,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny",
                        "weight": 1.0,
                        "input_image": encoded_image,
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }

    print("Sending request to Stable Diffusion API...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        print("Please ensure the WebUI is running and fully loaded.")
        return

    r = response.json()
    
    # The API returns a list of base64 images in 'images'
    if 'images' in r and len(r['images']) > 0:
        output_data = base64.b64decode(r['images'][0])
        with open(args.output_image, 'wb') as f:
            f.write(output_data)
        print(f"Success! Dashcam image saved to: {args.output_image}")
    else:
        print("Error: No image was returned by the API.")

if __name__ == "__main__":
    main()
