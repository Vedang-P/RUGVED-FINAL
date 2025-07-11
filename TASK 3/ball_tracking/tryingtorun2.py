import cv2
import numpy as np

def track_ball_by_color(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range for the ball in HSV (adjust these)
        lower_color = np.array([30, 150, 50])    # Example: green
        upper_color = np.array([85, 255, 255])   # Adjust for your ball's color

        # Create a mask for the color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
       # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # filter out small areas
                (x, y, w, h) = cv2.boundingRect(contour)
                center = (int(x + w/2), int(y + h/2))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, "Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

        cv2.imshow("Ball Tracking", frame)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
track_ball_by_color("/Users/vedang/Desktop/rugved work/ball_tracking/Ball_Tracking.mp4")
