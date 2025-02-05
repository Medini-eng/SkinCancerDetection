# import streamlit as st
# import torch
# import numpy as np
# import cv2
# from ultralytics import YOLO
# from PIL import Image

# # Load YOLOv8 model
# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")  # Ensure best.pt is in the same directory

# model = load_model()

# # Streamlit UI
# st.title("Skin Cancer Detection with YOLOv8")
# st.write("Upload an image to detect skin cancer using AI.")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert image to OpenCV format
#     img_array = np.array(image)
#     img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

#     # Run YOLOv8 inference
#     results = model(img_array)

#     # Draw bounding boxes and labels
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
#             confidence = box.conf[0].item()  # Confidence score
#             label = result.names[int(box.cls[0])]  # Class label
            
#             # Draw bounding box
#             cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img_array, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Convert image back to PIL format for display
#     result_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#     st.image(result_image, caption="Detection Results", use_column_width=True)
import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Real-Time Skin Cancer Detection")
st.write("Turn on your camera and analyze skin lesions in real-time.")

# Start webcam
video = st.camera_input("Turn on Camera")

if video:
    # Convert to OpenCV format
    image = Image.open(video)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference
    results = model(img_array)

    # Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
            confidence = box.conf[0].item()  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label
            
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to RGB for Streamlit display
    result_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="Live Detection", use_column_width=True)
