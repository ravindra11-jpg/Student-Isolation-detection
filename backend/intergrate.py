from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
from datetime import datetime
import threading
import os
import csv
from collections import deque, defaultdict
import time

app = Flask(__name__)
CORS(app)

# ── Load YOLO models ──────────────────────────────────────────────────────────
student_model = YOLO(
    "/Users/ravindrasaravanan/Documents/projects/python"
    "/Studentanalysis/FINAL_STUD_DET/detect/train6/weights/best.pt"
)
social_model = YOLO(
    "/Users/ravindrasaravanan/Documents/projects/python"
    "/Studentanalysis/FINAL_ISO_DET/detect/train/weights/best.pt"
)

# ── Global state & logging setup ─────────────────────────────────────────────
isolation_history  = defaultdict(lambda: deque(maxlen=6))
flagged_students   = set()
current_isolated   = []    # Names detected as isolated in the latest pass
last_group_count   = 0     # Number of "socializing" boxes in the latest pass

log_file = "isolation_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "student", "status", "flagged"])

last_log_time = time.time()

# ── CSV-based flagged logic ─────────────────────────────────────────────────
def get_flagged_from_csv(csv_path="isolation_log.csv", window_size=6, threshold=0.8):
    hist_csv = defaultdict(lambda: deque(maxlen=window_size))
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reversed(list(reader)):
                name   = row["student"]
                status = row["status"].strip().lower()
                if len(hist_csv[name]) < window_size:
                    hist_csv[name].appendleft(1 if status == "isolated" else 0)
        return sorted(
            n for n, h in hist_csv.items()
            if len(h) == window_size and sum(h)/window_size >= threshold
        )
    except FileNotFoundError:
        return []

# ── Background thread: detect/update/log ─────────────────────────────────────
def update_stats():
    global last_log_time, current_isolated, last_group_count

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1) Detect students (faces)
        students = []
        for box in student_model(frame, verbose=False, conf=0.7)[0].boxes:
            name        = student_model.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy      = (x1 + x2)/2, (y1 + y2)/2
            students.append((name, (cx, cy)))

        # 2) Detect socializing regions
        social_regions = []
        group_count    = 0
        for box in social_model(frame, verbose=False, conf=0.7)[0].boxes:
            label      = social_model.names[int(box.cls)].lower()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            social_regions.append((label, (x1, y1, x2, y2)))
            if "socializing" in label:
                group_count += 1
        last_group_count = group_count

        # 3) Assign each student a status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_isolated = []
        for name, (cx, cy) in students:
            status = "isolated"
            for label, (x1, y1, x2, y2) in social_regions:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    status = "normal"
                    break

            # update history deque
            isolation_history[name].append(1 if status == "isolated" else 0)
            if status == "isolated":
                current_isolated.append(name)

            # flag if >80% isolated
            hist = isolation_history[name]
            if len(hist) == hist.maxlen and sum(hist)/len(hist) >= 0.8:
                flagged_students.add(name)

        # 4) Log to CSV every 5s
        if time.time() - last_log_time >= 5:
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                for name, hist in isolation_history.items():
                    last_status = "isolated" if hist[-1] == 1 else "normal"
                    writer.writerow([
                        timestamp,
                        name,
                        last_status,
                        "yes" if name in flagged_students else "no"
                    ])
            last_log_time = time.time()

# ── Video streaming (overlay name–status) ────────────────────────────────────
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # detect students
        students = []
        for box in student_model(frame, conf=0.7, verbose=False)[0].boxes:
            name = student_model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            students.append((name, (x1, y1, x2, y2), (cx, cy)))

        # detect social context
        social_regions = []
        for box in social_model(frame, conf=0.7, verbose=False)[0].boxes:
            label = social_model.names[int(box.cls)].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            social_regions.append((label, (x1, y1, x2, y2)))

        # draw boxes + labels
        for name, (sx1, sy1, sx2, sy2), (cx, cy) in students:
            status = "isolated"
            for label, (bx1, by1, bx2, by2) in social_regions:
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    status = "socializing"
                    break

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{name} – {status}",
                (sx1, sy1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() + b'\r\n'
        )
    cap.release()

# ── Flask routes ────────────────────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def get_stats():
    return jsonify({
        "total_students":     len(isolation_history),
        "isolated_students":  len(current_isolated),
        "active_groups":      last_group_count,
        "students_present":   sorted(isolation_history.keys()),
        "flagged_students":   get_flagged_from_csv(),
        "currently_isolated": current_isolated
    })

# ── Start background thread & run Flask app ────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=update_stats, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
