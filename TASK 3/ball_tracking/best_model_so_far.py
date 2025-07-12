import cv2
import torch
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.5  # confidence threshold

def track_object_in_video(video_path, speed_scale=1.0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video FPS and compute frame delay
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int((1000 / fps) * speed_scale) if fps > 0 else 33

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection
        results = model(frame)
        predictions = results.pred[0]  # tensor [N, 6] (x1, y1, x2, y2, conf, cls)

        for *box, conf, cls in predictions:
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Compute and show FPS
        end_time = time.time()
        fps_display = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "/Users/vedang/Desktop/rugved work/ball_tracking/Ball_Tracking.mp4"
track_object_in_video(video_path, speed_scale=1.0)  # 1.0 = real speed, 0.5 = slower, 2.0 = faster
