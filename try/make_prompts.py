import random
import os

road_types = [
    "asphalt highway", "dirt road", "muddy trail", "snowy road", "wet city street", 
    "gravel road", "sandy track", "concrete suburban street", "cobblestone street", 
    "coastal highway", "mountain pass", "icy road", "pothole-filled road", 
    "cracked asphalt", "forest path", "dry desert road", "flooded street"
]

times_of_day = [
    "early morning", "noon", "late afternoon", "golden hour sunset", 
    "night time", "twilight", "dusk", "dawn", "pitch black night"
]

weathers = [
    "sunny", "overcast", "heavy rain", "light drizzle", "heavy blizzard snowstorm", 
    "light snow", "thick fog", "clear blue sky", "stormy", "misty", "hazy"
]

environments = [
    "pine forest", "dry desert", "rural village", "modern cyberpunk city", 
    "abandoned industrial zone", "suburban neighborhood", "tropical jungle", 
    "coastal cliffs", "autumn woods with colorful leaves", "open green fields", 
    "snow-capped mountains", "slum district", "farming countryside"
]

base = "Authentic dashcam footage, GoPro hero 9, {}, highly detailed road surface texture, {}, {}, {}, natural lighting, raw unedited video, grainy, dirty windshield, real life, slight motion blur"

prompts = []
# Generate combinations
for r in road_types:
    for t in times_of_day:
        for w in weathers:
            for e in environments:
                prompts.append(base.format(r, t, w, e))

# Shuffle for randomness
random.seed(42)
random.shuffle(prompts)

# Take 2500
prompts = prompts[:2500]

# Write to file
out_path = os.path.join(os.path.dirname(__file__), "2500_prompts.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for p in prompts:
        f.write(p + "\n")

print(f"Successfully generated {len(prompts)} unique prompts to {out_path}")
