import torch
import numpy as np
from ultralytics import YOLO
import cv2



def detect_and_crop(image: np.ndarray) -> np.ndarray:
    """
    Detect cat face in an image using YOLO and return cropped region.
    If no detection is found, return the original image.
    """

    # Load YOLOv8 model at startup
    try:
        model = YOLO("catface_detector.pt", verbose=False)
        print("✅ YOLO cat face detector loaded successfully.")
    except Exception as e:
        print("⚠️ Warning: YOLO model not loaded.", e)
        model = None

    if model is None:
        print("⚠️ YOLO model not available. Returning original image.")
        return image

    # Convert PIL Image to NumPy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert BGR if accidentally loaded that way
    if image.shape[-1] == 3 and np.mean(image[..., 0]) > np.mean(image[..., 2]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model.predict(source=image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return image

    # Select detection with highest confidence
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    x1, y1, x2, y2 = boxes[best_idx].astype(int)

    # Crop image safely
    h, w, _ = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return image
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
