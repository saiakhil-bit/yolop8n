import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO('yolov8n.pt')

# Streamlit application setup
st.title("YOLOv8 Object Detection")
st.write("Upload a video to perform object detection.")

# Upload video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform object detection
    st.write("Processing the video...")
    results = model(source=temp_video_path, conf=0.4, save=True, show=False)

    stframe = st.empty()  # Placeholder for video frames

    # Display processed video frames
    for result in results:
        for detection in result.boxes.data:  # Iterate over detected boxes
            # Get the frame from the result
            frame = result.orig_img  # This should be the original image/frame
            # Ensure the frame is in the correct format
            if isinstance(frame, np.ndarray):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                stframe.image(frame_rgb, channels="RGB")
            else:
                st.warning("Unable to process frame; skipping.")

    st.write("Detection completed!")
else:
    st.warning("Please upload a video file to proceed.")

