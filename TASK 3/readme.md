# 🎯 Task 3: Green Ball Tracking in Video (Basic + YOLO Approach)

In this task, you will track a green-colored ball in a video using two methods:

1. ✅ **Basic Approach (No YOLO)** — Use fundamental image processing techniques (color filtering, contour detection) with OpenCV.
2. 🧠 **YOLO-Based Approach** — Use a pre-trained YOLO model to detect and track the ball.

---

## 📌 Objective

Develop a working pipeline to **detect and track a green ball** throughout a video file using two different paradigms:

| Method          | Description                                |
|-----------------|--------------------------------------------|
| Basic CV        | No external models — just color filtering, masking, thresholding, and contour tracking with OpenCV. |
| YOLO-based      | Use pretrained YOLOv3/v4 to detect and track the ball. Optionally apply a confidence filter and bounding box. |

---

## 🗂️ Files Included

| File Name               | Description                                |
|------------------------|--------------------------------------------|
| `green_ball_video.mp4` | The input video to be processed            |
| `track_basic.py`       | Python script using OpenCV and color masking only |
| `track_yolo.py`        | Python script using YOLOv3 or YOLOv4       |
| `yolov3.cfg`           | YOLO config file (if used)                 |
| `yolov3.weights`       | YOLO weights (not tracked — download separately) |
| `coco.names`           | Class names for YOLO                       |

---
