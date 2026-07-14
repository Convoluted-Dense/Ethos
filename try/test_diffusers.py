from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import torch

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    controlnet_path = os.path.join(script_dir, "stable-diffusion-webui-forge", "models", "ControlNet", "control_v11p_sd15_canny.pth")
    model_path = os.path.join(script_dir, "stable-diffusion-webui-forge", "models", "Stable-diffusion", "realisticVisionV60B1_v51VAE.safetensors")

    controlnet = ControlNetModel.from_single_file(
        controlnet_path, 
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
