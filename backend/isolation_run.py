from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('/Users/ravindrasaravanan/Documents/projects/python/Studentanalysis/runs/detect/train/weights/best.pt')  # path to your trained weights

# Open the webcam
cap = cv2.VideoCapture(0)  # '0' is usually the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.6)
    annotated_frame = results[0].plot()  # manually plot annotations
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Optionally, you can process 'results' if you want to do something specific
    # for result in results:
    #     print(result)

    # Exit condition: Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()