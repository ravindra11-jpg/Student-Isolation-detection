import cv2
import os

# Load video
video_path = '/Users/ravindrasaravanan/Downloads/isolation/bigvideo.mov'
output_folder = '/Users/ravindrasaravanan/Documents/projects/python/Studentanalysis/extracted_imgs_for_anno'

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 30  # Save every 5th frame (adjust as needed)

count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:
        cv2.imwrite(f"{output_folder}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
    count += 1

cap.release()
print(f"Saved {saved_count} frames.")