import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import math

# Page config (IMPORTANT for UI)
st.set_page_config(
    page_title="YOLOv8 Detection App",
    page_icon="🔍",
    layout="wide"
)



# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size:40px;
        font-weight:bold;
        color:#4A90E2;
        text-align:center;
    }
    .subtitle {
        font-size:18px;
        text-align:center;
        color:black;
        margin-bottom:20px;
    }
    
    .stImage > div > div > p {
        color: black !important;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">🔍 YOLOv8 Object Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and detect objects instantly</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Load model
model = YOLO("yolov8n.pt")

# Class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Upload section
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)
    img = np.array(image)

    with col1:
        st.image(image, caption=" Original Image", use_column_width=True, channels="RGB")
        st.markdown("<p style='color:black; font-weight:bold; text-align:center;'>📷 Original Image</p>", unsafe_allow_html=True)

    results = model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])

            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(confidence * 100) / 100

                cls = int(box.cls[0])
                class_name = classNames[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(img, f"{class_name} {confidence}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0), 2)

    with col2:
        st.image(img, caption=" Detected Image", use_column_width=True)
        st.markdown("<p style='color:black; font-weight:bold; text-align:center;'> 🎯 Detected Image</p>", unsafe_allow_html=True)

else:
    st.info("Please upload an image to start detection.")