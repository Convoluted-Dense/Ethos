import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_FILE = "pregenerated_prompts.json"
TARGET_SCENES = 2500
BATCH_SIZE = 8

SYSTEM_PROMPT = (
    "You are a wildly creative AI director for an autonomous driving dataset augmentation system. "
    "Your job: invent a completely original, unexpected environment or weather scenario from your imagination "
    "and write a Stable Diffusion prompt pair for it. "
    "IMPORTANT: Stick strictly to environments possible on EARTH (e.g. blizzards, floods, neon cities, "
    "wildfires, monsoons, deserts, overgrown ruins, heavy fog, avalanches, mudslides, volcanic ash, etc). "
    "CRITICAL REQUIREMENT: This is for autonomous driving! The road and lane markings MUST remain clearly visible, "
    "drivable, and structurally intact. "
    "DO NOT use space, aliens, or magic. Keep it grounded but visually spectacular. "
    "Always respond with ONLY a valid JSON object with exactly three keys: "
    '"scenario", "positive", and "negative". No extra text, no markdown, no code block.'
)

USER_PROMPT = (
    "Invent a completely original Earth environment or weather scenario for a dashcam image. "
    "Do NOT use generic scenarios — surprise me. Make it vivid, specific, and unexpected. "
    "CRITICAL: Add phrases to the positive prompt ensuring a clear, drivable road with visible lane markings. "
    'Return ONLY this JSON: {"scenario": "one sentence describing what you invented", '
    '"positive": "detailed SD prompt describing the environment AND explicitly mentioning clearly visible road and lane markings", '
    '"negative": "comma-separated list of things to avoid: blurry, cartoonish, watermark, text, distorted road, missing lane lines, impassable road, etc."}'
)

def parse_json_from_response(response: str):
    try:
        data = json.loads(response)
        if "positive" in data and "negative" in data: return data
    except Exception:
        pass
    match = re.search(r'\{[^{}]*"scenario"[^{}]*"positive"[^{}]*"negative"[^{}]*\}', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "positive" in data and "negative" in data: return data
        except Exception:
            pass
    return None

def main():
    print(f"Loading LLM: {LLM_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompts = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                prompts = json.load(f)
            print(f"Loaded {len(prompts)} existing prompts.")
        except Exception:
            pass

    pbar = tqdm(total=TARGET_SCENES, initial=len(prompts), desc="Generating Prompts")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Pre-tokenize the input
    tokenizer.padding_side = "left"
    model_inputs = tokenizer([text] * BATCH_SIZE, return_tensors="pt", padding=True).to(model.device)

    while len(prompts) < TARGET_SCENES:
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=1.4,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            # Slice output to get only the newly generated tokens
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            added_this_batch = 0
            for response in responses:
                parsed = parse_json_from_response(response)
                if parsed:
                    if "scenario" not in parsed:
                        parsed["scenario"] = "Unknown scenario"
                    prompts.append(parsed)
                    added_this_batch += 1
                    if len(prompts) >= TARGET_SCENES:
                        break
            
            pbar.update(added_this_batch)
            
            with open(OUTPUT_FILE, "w") as f:
                json.dump(prompts, f, indent=4)
                
        except Exception as e:
            print(f"Error during generation: {e}")
            break

    pbar.close()
    print(f"Finished! Generated total {len(prompts)} prompts. Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
