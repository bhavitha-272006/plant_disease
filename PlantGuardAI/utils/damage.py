"""
Damage and Computer Vision utility module for PlantGuard AI.
Wraps verified vision pipeline with leaf foliage segmentation,
valid severity calculation, and experimental symptom annotation.
"""

from PIL import Image
from modules.vision import (
    analyze_severity_and_vision,
    segment_leaf_foliage,
    get_experimental_symptom_mask,
    assess_image_quality
)


def prepare_image(pil_img: Image.Image):
    import numpy as np
    import cv2
    rgb = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return rgb, hsv


def get_leaf_mask(hsv):
    import cv2
    import numpy as np
    lower = np.array([15, 20, 20])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def get_damage_mask(hsv, leaf_mask):
    import cv2
    import numpy as np
    brown = cv2.inRange(hsv, np.array([3, 30, 20]), np.array([30, 255, 210]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
    damage = cv2.bitwise_or(brown, dark)
    return cv2.bitwise_and(damage, leaf_mask)


def analyze_damage(pil_img: Image.Image):
    """
    Returns (percentage, severity, heatmap)
    Calculated strictly over valid leaf pixels.
    """
    res = analyze_severity_and_vision(pil_img)
    if not res["valid"]:
        return 0.0, "Undetermined", res["heatmap"]

    pct = res["affected_area_pct"] if res["affected_area_pct"] is not None else 0.0
    return pct, res["severity_level"], res["heatmap"]


def annotate(pil_img: Image.Image):
    """
    Returns annotated image with detected bounding boxes and contours.
    """
    res = analyze_severity_and_vision(pil_img)
    return res["annotated"]
