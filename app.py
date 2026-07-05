import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Page config
st.set_page_config(
    page_title="AI Face Detection System",
    page_icon="🎭",
    layout="centered"
)

# Title and description
st.title("🎭 AI Face Detection System")
st.markdown("Upload an image and the AI will detect all faces in it.")
st.markdown("---")

# Load DNN model (cached so it loads only once)
@st.cache_resource
def load_model():
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net

net = load_model()

# Confidence threshold slider
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.01,
    help="Lower value = detect more faces (but may include false positives)"
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    (h, w) = image.shape[:2]

    # Prepare image for DNN
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    # Detect faces
    net.setInput(blob)
    detections = net.forward()

    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            face_count += 1

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show confidence %
            label = f"{confidence * 100:.1f}%"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show result
    st.markdown("---")

    # Metric box
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Faces Detected", face_count)
    with col2:
        st.metric("Confidence Threshold", f"{confidence_threshold*100:.0f}%")

    # Display image
    st.image(image_rgb, caption="Detected Faces", use_column_width=True)

    # Success/warning message
    if face_count > 0:
        st.success(f"✅ {face_count} face(s) successfully detected!")
    else:
        st.warning("⚠️ No faces detected. Try lowering the confidence threshold.")

else:
    # Placeholder when no image uploaded
    st.info("👆 Please upload an image to get started.")