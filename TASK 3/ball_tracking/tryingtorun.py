import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to track object in the video
def track_object_in_video(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Get the confidence scores
        scores = results.xywh[0][:, 4].cpu().numpy()

        # Get the detected boxes and class labels
        boxes = results.xywh[0][:, :4].cpu().numpy()
        labels = results.xywh[0][:, 5].cpu().numpy()

        # Iterate through the detections
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                x1, y1, w, h = boxes[i]
                x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Class: {labels[i]} Confidence: {scores[i]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Tracking', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function with your video path
video_path = 'ball_tracking/Ball_Tracking.mp4'  # Ensure this path is correct
track_object_in_video(video_path)
