import cv2

# Load DNN face detector model
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found or not accessible.")
    exit()

print("Webcam started. Press 'Q' to quit, 'S' to save screenshot.")

while True:
    # Read frame from webcam
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    (h, w) = frame.shape[:2]

    # Prepare frame for DNN model
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    # Detect faces
    net.setInput(blob)
    detections = net.forward()

    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:   # webcam pe 0.5 theek hai (real-time, clear faces hote hain)
            face_count += 1

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show confidence %
            text = f"{confidence * 100:.1f}%"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show face count on top left
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Real-Time Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # press Q to quit
    if key == ord('q'):
        print("Quitting...")
        break

    # press S for screenshot 
    elif key == ord('s'):
        cv2.imwrite("screenshots/webcam_capture.jpg", frame)
        print("Screenshot saved in screenshots folder!")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam released.")