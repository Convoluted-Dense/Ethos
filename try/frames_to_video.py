import cv2
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Convert a folder of frames into an MP4 video")
    parser.add_argument("input_folder", help="Folder containing the frame images (e.g., output_snowy_frames)")
    parser.add_argument("output_video", help="Output MP4 file name (e.g., final_video.mp4)")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video")
    args = parser.parse_args()

    # Get all .jpg images and sort them alphabetically so they are in order
    image_files = sorted(glob.glob(os.path.join(args.input_folder, "*.jpg")))
    
    if not image_files:
        print(f"Error: No .jpg files found in {args.input_folder}")
        return

    print(f"Found {len(image_files)} frames. Generating video...")

    # Read the first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    # 'mp4v' is a good cross-platform codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))

    for idx, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        out.write(img)
        
        # Print progress every 50 frames
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} frames...")

    out.release()
    print(f"Success! Video saved as {args.output_video}")

if __name__ == "__main__":
    main()
