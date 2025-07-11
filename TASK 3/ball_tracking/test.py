import cv2

# Load the video
cap = cv2.VideoCapture("ball_tracking/Ball_Tracking.mp4")

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Let user select ROI
bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
cv2.waitKey(0)  # Important on macOS!
cv2.destroyWindow("Select Object to Track")

# Create tracker (use CSRT or MOSSE for better results)
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    print("frame is being read" )
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(20)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
