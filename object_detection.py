# Import the required libraries
from ultralytics import YOLO  # To use the YOLOv8 model
import cv2                    # For webcam and image processing
import math                  # For rounding confidence score

# Start the webcam (0 means default webcam)
cap = cv2.VideoCapture(0)

# Set the width and height of the webcam window
cap.set(3, 640)  # Width = 640 pixels
cap.set(4, 480)  # Height = 480 pixels

# Load the pre-trained YOLOv8 model (make sure this file path is correct)
model = YOLO("yolo-Weights/yolov8n.pt")

# List of all object classes that YOLO can detect (from COCO dataset)
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

# Run the loop continuously to read video frames from webcam
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # If reading fails, break the loop
    if not success:
        print("Failed to read from webcam.")
        break

    # Run the YOLO model on the image (stream=True allows video input)
    results = model(img, stream=True)

    # Process each result from the model
    for r in results:
        boxes = r.boxes  # Get all bounding boxes

        # Loop through each detected object
        for box in boxes:
            # Get the coordinates of the box (top-left and bottom-right corners)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int for OpenCV

            # Draw the rectangle around the object
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Get the confidence score of detection
            confidence = float(box.conf[0])  # Extract confidence value
            confidence = math.ceil(confidence * 100) / 100  # Round to 2 decimal places
            print("Confidence --->", confidence)

            # Get the class ID and class name (like person, car, etc.)
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print("Class name -->", class_name)

            # Add label (class name and confidence) to the image
            cv2.putText(img, f"{class_name} {confidence}", (x1, y1 - 10),  # Position text above box
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the image with detected objects in a window named 'Webcam'
    cv2.imshow('Webcam', img)

    # Press 'q' key to exit the loop and close the window
    if cv2.waitKey(1) == ord('q'):
        break





