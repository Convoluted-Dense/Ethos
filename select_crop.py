import os
import random
import cv2

def main():
    img_dir = os.path.join("dataset", "img")
    if not os.path.exists(img_dir):
        print(f"Error: Directory '{img_dir}' not found.")
        return
        
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"Error: No images found in '{img_dir}'.")
        return
        
    random_img_name = random.choice(images)
    img_path = os.path.join(img_dir, random_img_name)
    print(f"Selected random image for crop definition: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
        
    h, w, c = img.shape
    print(f"Original image dimensions: Width={w}, Height={h}")
    
    print("\n" + "="*50)
    print("INSTRUCTIONS FOR CROP SELECTION:")
    print("1. Click and drag on the image window to draw a bounding box (rectangle).")
    print("2. Press ENTER or SPACE to confirm the selection.")
    print("3. Press 'c' to cancel and close.")
    print("="*50 + "\n")
    
    # cv2.selectROI opens a GUI window allowing drag-and-drop crop selection
    roi = cv2.selectROI("Drag to Select Crop Region (ENTER/SPACE to confirm, C to cancel)", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    rx, ry, rw, rh = roi
    if rw > 0 and rh > 0:
        print("\n" + "#"*50)
        print("CROP BOUNDS DEFINED:")
        print(f"X (left): {rx}")
        print(f"Y (top): {ry}")
        print(f"Width: {rw}")
        print(f"Height: {rh}")
        print("#"*50 + "\n")
        
        # Display the crop values in PyTorch TF.crop format
        print("To apply this crop in train_steering.py, update the crop logic to:")
        print("-" * 60)
        print(f"# Crop using user-selected region")
        print(f"img = TF.crop(img, top={ry}, left={rx}, height={rh}, width={rw})")
        print("-" * 60)
    else:
        print("Crop selection cancelled or invalid.")

if __name__ == "__main__":
    main()
