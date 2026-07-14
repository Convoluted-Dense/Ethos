import os
import subprocess

# Define where your simulator images are, and where the dashcam images should go
INPUT_DIR = "sim_images"
OUTPUT_DIR = "dashcam_images"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through every file in the input directory
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"dashcam_{filename}")
        
        print(f"Converting {filename}...")
        
        # This calls the script we just made!
        import sys
        subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_dashcam.py"), 
            input_path, 
            output_path
        ])

print("Batch conversion complete!")
