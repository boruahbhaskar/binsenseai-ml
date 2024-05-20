import cv2
import numpy as np
from preprocess_image import preprocess_image
import json

def detect_objects(preprocessed_image):
    """
    Object Detection Module (YOLO)

    Args:
        preprocessed_image (numpy array): Preprocessed image

    Returns:
        detections (list): Bounding boxes and class probabilities for detected objects
    """
    # Load YOLO weights and config
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

    # Get the output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Create a blob from the preprocessed image
    blob = cv2.dnn.blobFromImage(preprocessed_image, 1/255, (416, 416), swapRB=True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Run the forward pass to get the detections
    outputs = net.forward(output_layers)

    # Initialize the detections list
    detections = []

    # Loop through the detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get the bounding box coordinates
                center_x = int(detection[0] * preprocessed_image.shape[1])
                center_y = int(detection[1] * preprocessed_image.shape[0])
                width = int(detection[2] * preprocessed_image.shape[1])
                height = int(detection[3] * preprocessed_image.shape[0])
                # Append the detection to the list
                detections.append((center_x, center_y, width, height, confidence, class_id))

    return detections

 # get the preprocessed image

image = cv2.imread('/Users/bboruah/Machine_Learning/CapstonProjectTest/Image-Inventory-Reconciliation-with-SVM-and-CNN/data/amazon_bin_images/00085.jpg')

json_path = "/Users/bboruah/Machine_Learning/CapstonProjectTest/Image-Inventory-Reconciliation-with-SVM-and-CNN/data/amazon_bin_metadata/00085.json"
metadata = json.loads(open(json_path).read())

preprocessed_image = preprocess_image(image, metadata)

detections = detect_objects(preprocessed_image)
