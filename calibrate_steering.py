import cv2
import json
import numpy as np
import sys
from pathlib import Path
from beamng_collect import find_beamng_window, capture_printwindow

ROI_FILE = "steering_roi.json"

def extract_steering(img, roi):
    x, y, w, h = roi
    crop = img[y:y+h, x:x+w]
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Orange color bounds in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    mid = w // 2
    left_half = mask[:, :mid]
    right_half = mask[:, mid:]
    
    left_orange = np.count_nonzero(left_half)
    right_orange = np.count_nonzero(right_half)
    
    # Find the bounding box of the orange pixels to determine the actual height of the bar
    coords = cv2.findNonZero(mask)
    if coords is not None:
        _, _, _, h_orange = cv2.boundingRect(coords)
        max_area = mid * h_orange
        if max_area > 0:
            left_val = left_orange / max_area
            right_val = right_orange / max_area
            steering = right_val - left_val
            return max(-1.0, min(1.0, steering)), mask, crop
            
    return 0.0, mask, crop

def main():
    print("Searching for BeamNG window...")
    win = find_beamng_window()
    if not win:
        print("BeamNG window not found. Please start the game.")
        sys.exit(1)
        
    print("Capturing frame...")
    img = capture_printwindow(win.hwnd)
    if img is None:
        print("Failed to capture window.")
        sys.exit(1)
        
    print("Please draw a rectangle around the steering UI element.")
    print("Make sure the rectangle is centered exactly on the middle of the steering bar.")
    print("Press ENTER or SPACE to confirm the selection. Press C to cancel.")
    
    roi = cv2.selectROI("Select Steering UI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Steering UI")
    
    if roi == (0, 0, 0, 0):
        print("Steering selection cancelled.")
        sys.exit(0)
        
    print("\nNow, please draw a rectangle around the MINIMAP UI element.")
    print("Press ENTER or SPACE to confirm. Press C to skip minimap.")
    
    map_roi = cv2.selectROI("Select Minimap UI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Minimap UI")
    
    if map_roi == (0, 0, 0, 0):
        print("Minimap selection skipped.")
        map_roi = None
        
    # Save ROIs
    with open(ROI_FILE, "w") as f:
        json.dump({"roi": roi, "map_roi": map_roi}, f)
        
    print(f"Saved Steering ROI {roi} and Map ROI {map_roi} to {ROI_FILE}")
    
    print("Testing steering extraction. Press Q to exit.")
    while True:
        img = capture_printwindow(win.hwnd)
        if img is None:
            continue
            
        steering, mask, crop = extract_steering(img, roi)
        
        # Visualize
        vis = crop.copy()
        h, w = vis.shape[:2]
        mid = w // 2
        # Draw center line
        cv2.line(vis, (mid, 0), (mid, h), (255, 255, 255), 1)
        
        # Display steering value
        text = f"Steering: {steering:.2f}"
        cv2.putText(vis, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Show mask and crop
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.vstack((vis, mask_bgr))
        # Enlarge for easier viewing
        combined = cv2.resize(combined, (w * 2, h * 4), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Steering Test", combined)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
