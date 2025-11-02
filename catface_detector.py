import torch
import numpy as np
from ultralytics import YOLO
import cv2

def detect_and_crop(image: np.ndarray, shrink_factor: float = 0.80) -> np.ndarray:
    """
    Detects cat face in an image using YOLO and returns a square cropped region.
    If no detection is found, returns the original image.
    The crop keeps aspect ratio square and adds white padding if needed.
    """

    # Load YOLOv8 model once (lazy load)
    try:
        model = YOLO("catface_detector.pt")
    except Exception as e:
        print("⚠️ Warning: YOLO model not loaded.", e)
        return image

    # Convert PIL to NumPy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Ensure RGB order
    if image.shape[-1] == 3 and np.mean(image[..., 0]) > np.mean(image[..., 2]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model.predict(source=image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return image

    # Pick highest confidence detection
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    x1, y1, x2, y2 = boxes[best_idx].astype(int)

    # Compute square box around center with shrink factor
    h, w, _ = image.shape
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * shrink_factor
    new_x1, new_y1 = int(cx - side / 2), int(cy - side / 2)
    new_x2, new_y2 = int(cx + side / 2), int(cy + side / 2)

    # Compute padding if out of bounds
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - w)
    pad_bottom = max(0, new_y2 - h)

    # Crop safely inside image
    cropped = image[max(0, new_y1):min(h, new_y2),
                    max(0, new_x1):min(w, new_x2)]

    # Add white padding if needed
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

    # Optional: resize to 224×224 for recognition model
    cropped = cv2.resize(cropped, (224, 224))

    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)