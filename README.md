
# 🔍 YOLOv8 Object Detection Web App

This is a Streamlit-based web application that performs real-time object detection using the YOLOv8 model. Users can upload images and detect multiple objects with adjustable confidence threshold.

---

## 🚀 Features

- 📤 Upload images (JPG, PNG, JPEG)
- 🎯 Detect objects using YOLOv8
- ⚙️ Adjustable confidence threshold
- 🖼️ Displays original and detected images side-by-side
- 🧠 Clean and interactive UI using Streamlit
- 🏷️ Bounding boxes with labels and confidence score

---

## 🛠️ Tech Stack

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pillow

---

## 📂 Project Structure

object_detection_yolov8/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## How It Works

- The app uses a pre-trained YOLOv8 model (yolov8n.pt)
- Detects objects from COCO dataset classes
- Filters predictions based on confidence threshold
- Displays bounding boxes with labels on detected objects


## Output Pictures

![object_detect](https://github.com/Vaishnavi26-Kasture/object_detection_yolov8/blob/main/object_detect.png)






![object_detection](https://github.com/Vaishnavi26-Kasture/object_detection_yolov8/blob/main/object_detection.png)

















