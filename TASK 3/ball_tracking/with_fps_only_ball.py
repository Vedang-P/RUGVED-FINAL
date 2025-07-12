import cv2
import torch
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.4
BALL_CLASS_ID = 32

def nothing(x):
    pass

def track_object_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    base_delay = (1000 / video_fps) if video_fps > 0 else 33

    cv2.namedWindow("Object Tracking")
    cv2.createTrackbar("Speed (%)", "Object Tracking", 100, 300, nothing)

    while True:
        speed_percent = cv2.getTrackbarPos("Speed (%)", "Object Tracking")
        speed_percent = max(speed_percent, 10)  # avoid 0%

        # Fast forward logic: skip frames
        skip_frames = int(speed_percent / 100) - 1 if speed_percent > 100 else 0
        for _ in range(skip_frames):
            cap.read()  # skip frame

        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        predictions = results.pred[0]

        for *box, conf, cls in predictions:
            if int(cls) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        fps_display = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if speed_percent <= 100:
            delay = int(base_delay / (speed_percent / 100))
            cv2.waitKey(delay)
        else:
            cv2.waitKey(1)  # minimal delay for fast-forward

        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = "/Users/vedang/Desktop/rugved work/ball_tracking/Ball_Tracking.mp4"
track_object_in_video(video_path)
 