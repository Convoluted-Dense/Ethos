import os
import cv2
import numpy as np
import random
import math
from tqdm import tqdm

def add_jello_effect(img, amplitude=5.0, frequency=2.5, phase=0.0):
    """
    Simulates rolling shutter / jello effect caused by high-frequency vibrations.
    Shifts rows horizontally based on a sine wave.
    """
    h, w = img.shape[:2]
    out_img = np.zeros_like(img)
    for y in range(h):
        shift = int(amplitude * math.sin(2 * math.pi * frequency * (y / h) + phase))
        if shift > 0:
            src_x1, src_x2 = 0, w - shift
            dst_x1, dst_x2 = shift, w
        elif shift < 0:
            src_x1, src_x2 = -shift, w
            dst_x1, dst_x2 = 0, w + shift
        else:
            src_x1, src_x2 = 0, w
            dst_x1, dst_x2 = 0, w
            
        out_img[y, dst_x1:dst_x2] = img[y, src_x1:src_x2]
        if shift > 0:
            out_img[y, 0:shift] = img[y, 0]
        elif shift < 0:
            out_img[y, w+shift:w] = img[y, w-1]
    return out_img

def main():
    img_dir = r"C:\Golden Buggy\dataset\img"
    out_dir = r"C:\Golden Buggy\jello_tests"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all images and sort them to ensure chronological order
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    all_imgs.sort()
    
    # Skip first 50, then take 500
    if len(all_imgs) < 550:
        print("Not enough images in dataset.")
        return
        
    selected_imgs = all_imgs[50:550]
    
    print(f"Processing {len(selected_imgs)} images (skipped first 50) for jello effect testing...")
    
    for img_name in tqdm(selected_imgs):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Apply jello effect with random phase to simulate continuous vibration
        phase = random.uniform(0, 2 * math.pi)
        
        # Based on the w_vids video, the vibration is high frequency but moderate amplitude
        jello_img = add_jello_effect(img, amplitude=7.0, frequency=3.0, phase=phase)
        
        # Create a side-by-side comparison for each image
        # Resize for easier viewing
        h, w = img.shape[:2]
        scale = 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_rs = cv2.resize(img, (new_w, new_h))
        jello_rs = cv2.resize(jello_img, (new_w, new_h))
        
        # Add labels
        cv2.putText(img_rs, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(jello_rs, 'Jello Augmented', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        comparison = np.hstack((img_rs, jello_rs))
        
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, comparison)
        
    print("Done! Processed 500 images and saved to 'jello_tests'.")

if __name__ == '__main__':
    main()
