import cv2

# Load the DNN face detector model
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Load the image
image = cv2.imread("sample_images/faces.jpg")

if image is None:
    print("Error: Image not found. Check the path.")
    exit()

(h, w) = image.shape[:2]

# Prepare the image for the DNN model
# DNN model expects 300x300 size and specific preprocessing
blob = cv2.dnn.blobFromImage(
    image, 1.0, (300, 300),
    (104.0, 177.0, 123.0)
)

# Pass the image through the network
net.setInput(blob)
detections = net.forward()

# Loop over detections and draw boxes for confident ones
face_count = 0
confidence_threshold = 0.15 # only count detections above 50% confidence

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > confidence_threshold:
        face_count += 1

        # Get bounding box coordinates (scaled to original image size)
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (x1, y1, x2, y2) = box.astype("int")

        # Draw rectangle and confidence score
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{confidence*100:.1f}%"
        cv2.putText(
            image, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

print(f"Faces detected: {face_count}")

# Show and save result
cv2.imshow("DNN Face Detection", image)
cv2.imwrite("screenshots/dnn_detected_output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()