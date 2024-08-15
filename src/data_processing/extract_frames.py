import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video from input
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        # Read frame
        ret, frame = cap.read()
        
        # If no frames are left, break
        if not ret:
            break
        
        # Save frame as JPEG
        cv2.imwrite(os.path.join(output_folder, f"frame_{count:05d}.jpg"), frame)
        count += 1

    # Release VideoCapture obj
    cap.release()
    print(f"Extracted {count} frames from {video_path} to {output_folder}")

if __name__ == "__main__":
    # Path to input file
    video_path = "data/raw_videos/KhabibvsMcGregor-Trimmed.mp4"
    
    # Path to output folder to save extracted frames
    output_folder = "data/processed_frames/"
    
    # Extract frames
    extract_frames(video_path, output_folder)
