import cv2
from ultralytics import YOLO




# Check the path to data.yaml
dataset_path = '/Users/ravindrasaravanan/Documents/projects/python/Studentanalysis/isolationdataset/data.yaml'
print("Dataset YAML path:", dataset_path)

# Step 3: Load YOLOv8 model (Nano version is lightweight; you can change to yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8s.pt")  # n = nano, s = small, m = medium, l = large

# Step 4: Train the model
model.train(
    data=dataset_path,  # Path to the dataset YAML
    epochs=50,          # Number of training epochs (increase if needed)
    batch=16,           # Batch size (adjust depending on GPU power)
    imgsz=640,          # Image size (default is 640x640)
    workers=4           # Number of data loading workers (adjust if needed)
)

print("Training complete!")

# Step 5: Check saved weights
print("Best weights saved at: runs/detect/train/weights/best.pt")