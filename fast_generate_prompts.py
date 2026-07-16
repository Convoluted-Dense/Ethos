import json
import random

# Lists of prompt components to combine
TIMES_OF_DAY = [
    "bright sunny day", "overcast gray afternoon", "golden hour sunset", 
    "twilight dusk", "dark midnight", "neon-lit night", "foggy early morning"
]

WEATHERS = [
    "clear skies", "heavy torrential rain", "light drizzle with wet reflective roads", 
    "thick heavy fog", "blizzard snowstorm with whiteout conditions", 
    "light falling snow", "violent thunderstorm with lightning", "dusty desert wind", 
    "monsoon downpour", "autumn leaves blowing in the wind", "hazy heat wave"
]

ENVIRONMENTS = [
    "modern urban city street", "cyberpunk futuristic metropolis", "dense pine forest highway", 
    "tropical coastal road", "steep winding mountain pass", "barren desert highway", 
    "quiet suburban neighborhood", "rural farmland road", "overgrown post-apocalyptic ruined city", 
    "industrial factory district", "snowy alpine village", "coastal cliffside road", 
    "endless grassy plains"
]

ROAD_CONDITIONS = [
    "clearly marked paved road", "wet asphalt with clear bright lane lines", 
    "snow-plowed road with visible lanes", "cracked but perfectly drivable road with lane markings", 
    "multi-lane highway with bright white lines", "winding two-lane road with double yellow lines",
    "drivable clear path with distinct lane separators"
]

STYLES = [
    "photorealistic dashcam footage", "highly detailed, 8k resolution, cinematic lighting", 
    "realistic wide-angle driving view", "cinematic masterpiece, ultra-realistic",
    "sharp focus, realistic driving simulator graphic"
]

NEGATIVE_PROMPT = (
    "blurry, cartoonish, watermark, text, distorted road, missing lane markings, "
    "impassable road, unrealistic, low resolution, deformed, messy, people, animals, vehicles blocking road"
)

def generate_fast_prompts(count=2500):
    prompts = []
    
    # We will generate `count` random unique combinations
    seen = set()
    
    print(f"Generating {count} prompts instantly using rule-based mixing...")
    
    while len(prompts) < count:
        time = random.choice(TIMES_OF_DAY)
        weather = random.choice(WEATHERS)
        env = random.choice(ENVIRONMENTS)
        road = random.choice(ROAD_CONDITIONS)
        style = random.choice(STYLES)
        
        # Unique signature for deduplication
        combo_sig = f"{time}|{weather}|{env}|{road}|{style}"
        
        if combo_sig not in seen:
            seen.add(combo_sig)
            
            scenario = f"{env} during {time} with {weather}"
            positive = f"{scenario}, {road}, {style}"
            
            prompts.append({
                "scenario": scenario,
                "positive": positive,
                "negative": NEGATIVE_PROMPT
            })
            
    # Save to the file that generate_diffused_3.py expects
    output_file = "pregenerated_prompts.json"
    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=4)
        
    print(f"Successfully saved {count} unique prompts to {output_file}!")

if __name__ == "__main__":
    generate_fast_prompts(2500)
