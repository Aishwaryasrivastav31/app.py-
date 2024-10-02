import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model_path = "C:\\Users\\Lenovo\\Desktop"  # Replace with your model path

try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Set up the Streamlit app
st.title("YOLOv8 Object Detection")
st.write("Upload an image for object detection:")

# Image upload functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image
        image = uploaded_file.read()

        # Convert to numpy array for processing
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference on the image
        results = model(img)

        # Display results
        st.image(results.render()[0], caption='Detected Image', use_column_width=True)
        st.write("Inference Results:")
        st.write(results.pandas().xyxy)
    except Exception as e:
        st.error(f"Error processing image: {e}")
