import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import firebase_admin
from firebase_admin import credentials, firestore

# Check if Firebase is already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("dreamalink-819c7-firebase-adminsdk-fbsvc-566077b5b3.json")  # Ensure the file path is correct
    firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()


# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ensure best.pt is in the same directory

model = load_model()

# Streamlit UI
st.title("Skin Cancer Detection with YOLOv8")
st.write("Upload an image to detect skin cancer using AI.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

detected = False  # Flag to check if cancer is detected
detection_message = "Cancer Not Detected (Probably Normal Skin)"  # Default message
ml_accuracy = 0.0  # Default accuracy

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference
    results = model(img_array)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
            confidence = box.conf[0].item()  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label
            
            # Draw bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detected = True
            detection_message = f"Detected: {label}"
            ml_accuracy = confidence * 100  # Convert confidence to percentage

    # Convert image back to PIL format for display
    result_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="Detection Results", use_column_width=True)

# ------------------ Patient Details Form ------------------

st.subheader("Patient Information")
with st.form("patient_form"):
    name = st.text_input("Full Name")
    address = st.text_area("Address")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email ID")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    submit_details = st.form_submit_button("Submit & Send to Doctor")

# ------------------ Store Data in Firebase ------------------
if submit_details:
    patient_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "email": email,
        "address": address,
        "detection_result": detection_message,
        "model_accuracy": ml_accuracy
    }

    # Store in Firestore
    db.collection("patients").add(patient_data)

    st.success(f"✅ Data sent to Firebase successfully!\n\n"
               f"**Patient Details:**\n"
               f"- Name: {name}\n"
               f"- Age: {age}\n"
               f"- Gender: {gender}\n"
               f"- Phone: {phone}\n"
               f"- Email: {email}\n"
               f"- Address: {address}\n\n"
               f"**AI Model Detection:** {detection_message}\n"
               f"**Model Accuracy:** {ml_accuracy:.2f}%\n\n"
               f"📌 *Waiting for doctor's diagnosis...*")
