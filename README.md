# Student Isolation Detection

Student Isolation Detection is primarily a machine-learning computer-vision project for classroom interaction analysis. It combines two YOLOv8 object-detection models: one identifies known students in a video frame, and the other identifies isolated and socializing regions. A temporal rule then combines their detections to classify recent behavior and flag students whose isolation is consistent across multiple observations.

The repository contains two main applications:

- a Flask and OpenCV backend that runs YOLOv8 inference from a webcam and exposes a live MJPEG feed and statistics API; and
- a Next.js dashboard that displays the camera feed, currently isolated students, present students, and flagged students.

## Features

- Detects four named students: `darshan`, `kshitij`, `ravindra`, and `sai`
- Detects two social-context classes: `Isolated` and `socializing`
- Streams annotated webcam frames from a Flask endpoint
- Calculates isolation statistics every five seconds
- Maintains a six-frame isolation history
- Flags a student when at least 80% of that history is isolated
- Appends timestamped status records to `backend/isolation_log.csv`
- Provides a Next.js dashboard for live monitoring

## How It Works

The system uses a two-model YOLOv8 pipeline:

1. **Student detection** locates and identifies each known student.
2. **Social-context detection** locates regions classified as `Isolated` or `socializing`.
3. **Spatial association** compares each student's center point with the detected regions.
4. **Temporal aggregation** stores the latest six observations and raises a flag when at least 80% are isolated.

For every detected student, the backend calculates the center point of the student's bounding box and checks whether it is inside a detected social-context region. A detected `socializing` region is therefore intended to provide evidence that the student is not isolated. The live implementation also retains the isolated/social-context labels and can be adapted to use only the `socializing` class if that association rule is made explicit.

The in-memory result is returned by the `/stats` endpoint, while the `/video_feed` endpoint serves an annotated MJPEG stream. Isolation status is written to the CSV log every five seconds. The `isolation_log.csv` file is also used to calculate the flagged-student list.

## Machine-Learning Pipeline

### 1. Named-Student Model

The named-student model is a YOLOv8 nano detector trained on classroom images. It learns the appearance of four known students:

| Class | Meaning |
| --- | --- |
| `darshan` | student1 identity |
| `kshitij` | student2 identity |
| `ravindra` | student3 identity |
| `sai` | student4 identity |

The dataset is at `backend/student_detection.v4i.yolov8/` and is included in YOLOv8 format:

| Split | Images | Annotation files | Bounding boxes |
| --- | ---: | ---: | ---: |
| Training | 212 | 212 | 212 |
| Validation | 45 | 45 | 45 |

The training run in `backend/FINAL_STUD_DET/detect/train6/` used the nano pretrained model `yolov8n.pt`, 640-pixel input images, 16-image batches, and 50 epochs. The run's recorded metrics include:

- Peak precision: `0.9938`
- Peak recall: `1.0000`
- Peak mAP50: `0.9950`
- Peak mAP50-95: `0.86789`

These values describe the included validation run and should not be treated as a general real-world accuracy guarantee.

### 2. Social-Context Model

The social-context model is a YOLOv8 small detector trained to localize two regions or behaviors:

| Class | Meaning |
| --- | --- |
| `Isolated` | A region associated with isolated behavior or positioning |
| `socializing` | A region associated with students interacting socially |

The dataset is at `backend/isolationdataset/`:

| Split | Images | Annotation files | Bounding boxes | Isolated | Socializing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 171 | 171 | 447 | 266 | 181 |
| Validation | 46 | 46 | 110 | 65 | 45 |

The training run in `backend/FINAL_ISO_DET/detect/train/` used the small pretrained model `yolov8s.pt`, 640-pixel input images, 16-image batches, and 50 epochs. The run's recorded peak validation metrics include:

- Peak precision: `0.94711`
- Peak recall: `0.99065`
- Peak mAP50: `0.97843`
- Peak mAP50-95: `0.83592`

The final run weights and training curves are stored in `backend/FINAL_ISO_DET/detect/train/`.

### 3. Model Fusion and Isolation Decision

Both models infer with a confidence threshold of `0.7`. At each frame, the student model supplies a named center point while the social-context model supplies region boxes.

The backend marks a student as normal when the student's center is inside a detected social-context box. It marks the student isolated when no detected region contains that center. The implementation therefore is a spatial rule rather than a third trained classification model.

Each student's observations are retained in a `deque` of at most six frames:

```python
# 1 = isolated; 0 = normal
isolation_history[name].append(1 if status == "isolated" else 0)
```

A student is placed in `flagged_students` only when the deque is full and at least 80% of its values indicate isolation:

```python
len(hist) == hist.maxlen and sum(hist) / len(hist) >= 0.8
```

The history prevents a brief occlusion or one uncertain frame from immediately becoming a persistent isolation event.

> This project is a proof of concept. Detection names are fixed to the classes present in the included training dataset, and isolation status is based on a bounding-box region rather than individual behavior or a reliable clinical assessment.

## Repository Structure

```text
StudentIsolationDetection/
├── backend/
│   ├── intergrate.py                 # Flask API and live detection server
│   ├── isolation_log.csv             # Generated and historical status log
│   ├── isolation_run.py              # Standalone isolated/social-context demo
│   ├── stud_det_run.py               # Standalone named-student demo
│   ├── extractdataset.py             # Extracts frames from a source video
│   ├── train_yolov8.py               # YOLOv8 isolation-model training script
│   ├── yolov8n.pt                    # Nano base model
│   ├── yolov8s.pt                    # Small base model
│   ├── isolationdataset/             # Isolation/social-context dataset
│   ├── student_detection.v4i.yolov8/ # Named-student dataset
│   └── FINAL_*_DET/                  # Saved training runs and best weights
└── frontend/
    └── student-isolation-detection/  # Next.js dashboard
```

## Requirements

### Backend

- Python 3.9 or newer
- A webcam or other camera supported by OpenCV
- PyTorch-compatible installation of `ultralytics`
- Python packages:
  - `flask`
  - `flask-cors`
  - `opencv-python`
  - `ultralytics`

YOLOv8 from Ultralytics normally installs supporting dependencies such as NumPy, PyTorch, and pandas.

### Frontend

- Node.js 18 or newer
- npm (included with Node.js)

## Backend Setup

Create and activate a virtual environment from the project directory:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install flask flask-cors opencv-python ultralytics
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Update the model paths in `backend/intergrate.py` to match this checkout. The preferred paths are:

```python
student_model = YOLO("FINAL_STUD_DET/detect/train6/weights/best.pt")
social_model = YOLO("FINAL_ISO_DET/detect/train/weights/best.pt")
```

The backend should be launched from the `backend` directory so it writes `isolation_log.csv` there:

```bash
cd backend
python intergrate.py
```

The API will listen on port `5000` and is available on:

- video stream: `http://localhost:5000/video_feed`
- statistics: `http://localhost:5000/stats`

For access from another device, use the Flask host's LAN IP address instead of `localhost`.

### Included API Response

`GET /stats` returns JSON similar to:

```json
{
  "total_students": 4,
  "isolated_students": 1,
  "active_groups": 2,
  "students_present": ["darshan", "kshitij"],
  "flagged_students": ["kshitij"],
  "currently_isolated": ["kshitij"]
}
```

`GET /video_feed` returns a multipart MJPEG stream.

## Frontend Setup

Install dependencies and start the development server:

```bash
cd frontend/student-isolation-detection
npm install
npm run dev
```

Open `http://localhost:3000`.

The dashboard currently makes requests directly to a fixed LAN address in:

- `src/components/IsolationMonitor.tsx`
- `src/components/StatisticsPanel.tsx`

Replace `http://192.168.32.229:5000` in both files with the address of the machine running the Flask backend. For example:

```text
http://localhost:5000
```

For a browser on the same computer, `localhost` works. For a browser on a phone or another computer, use the backend host's reachable LAN IP address.

## Standalone Backend Scripts

The following scripts are useful for testing the trained models without starting the Flask server:

```bash
cd backend
python stud_det_run.py
python isolation_run.py
```

Press `q` to close an OpenCV window.

Both standalone scripts contain machine-specific model paths and should be updated to point to the weights in this repository. The social-context model and student model are also available from:

```text
FINAL_ISO_DET/detect/train/weights/best.pt
FINAL_STUD_DET/detect/train6/weights/best.pt
```

## Training

The datasets are annotated in YOLO format and use Roboflow metadata.

### Isolation or Social-Context Model

The isolation dataset contains:

- `Isolated`
- `socializing`

Its configuration is in `backend/isolationdataset/data.yaml`.

Before training, update the absolute paths in that YAML file or change the dataset paths to be relative to the current environment. Then train with:

```bash
cd backend
python train_yolov8.py
```

`train_yolov8.py` starts from `yolov8s.pt`, uses 640-pixel images, a batch size of 16, and 50 epochs. Ultralytics writes the resulting run to `runs/detect/train/weights/best.pt` by default.

### Named-Student Model

The named-student dataset contains four classes:

- `darshan`
- `kshitij`
- `ravindra`
- `sai`

Its configuration is at `backend/student_detection.v4i.yolov8/data.yaml`. The dataset contains `train` and `valid` image and label directories.

To retrain this model, use Ultralytics with the model and dataset paths for your environment:

```bash
cd backend
yolo detect train model=yolov8n.pt data=student_detection.v4i.yolov8/data.yaml epochs=50 imgsz=640 batch=16
```

The training configuration files inside `FINAL_*_DET` document the completed training runs and the Ultralytics options used.

## Frame Extraction

`backend/extractdataset.py` reads a source video and saves one frame for every 30 source frames. Before running it, update:

- `video_path`, which points to the source video
- `output_folder`, which points to `extracted_imgs_for_anno`

Then run:

```bash
cd backend
python extractdataset.py
```

## CSV Log Format

`backend/isolation_log.csv` uses the following columns:

| Column | Description |
| --- | --- |
| `timestamp` | Local date and time of the log entry |
| `student` | Detected student class name |
| `status` | `isolated` or `normal` |
| `flagged` | `yes` if the student meets the repeated-isolation rule, otherwise `no` |

The backend writes every five seconds and keeps isolation history for the latest six checks.

## Frontend Scripts

Run these commands from `frontend/student-isolation-detection`:

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Next.js development server |
| `npm run build` | Create a production build |
| `npm run start` | Run the production build |
| `npm run lint` | Run the configured Next.js lint command |

## Production Build

```bash
cd frontend/student-isolation-detection
npm run build
npm run start
```

Keep Flask running on port `5000` and ensure the frontend's API and video URLs point to that service.

## Configuration Notes

- The Flask server and webcam monitor both use camera index `0`.
- The Flask server runs `update_stats()` and `generate_frames()` separately, and each opens `cv2.VideoCapture(0)`; both therefore compete for access to the same camera.
- Detection confidence is set to `0.7` in the Flask backend.
- The isolation threshold is 80% across six observations.
- The backend log uses the working directory's `isolation_log.csv`; launch it from `backend` to keep logs with the project.
- Several source files contain absolute paths from the original development machine. Update those paths before running scripts outside this repository layout.

## Troubleshooting

### Flask cannot load a model

Verify the `.pt` weight files exist and update the paths in `intergrate.py`. Confirm that the model can be opened from the directory where the backend is launched.

### Webcam is unavailable

Check camera permissions and close other applications using the camera. If the required camera is not index `0`, change `cv2.VideoCapture(0)` to the appropriate index or video source.

### Dashboard cannot reach Flask

Confirm Flask is running, verify `/stats` in a browser, and replace the fixed LAN IP in both dashboard components. Check that the frontend and backend devices permit traffic on port `5000`.

### CORS or mixed-content errors

The Flask backend enables CORS. When serving the frontend from a remote HTTPS address, it may need to connect to an HTTPS backend or use a same-origin proxy instead of a plain HTTP Flask URL.

### Dataset training fails

Check the `train`, `val`, and `test` paths in the relevant `data.yaml` file. Image and label filenames must align within each split, and the dataset class count must match the `names` list.
