"""
Computer Vision Analysis and Severity Estimation Module for PlantGuard AI.

Provides:
- Image quality assessment (blur, brightness, contrast, foliage ratio).
- Leaf foliage segmentation (Otsu & LAB/HSV thresholding).
- Experimental visual symptom segmentation (heuristic HSV color segmentation).
  SCIENTIFIC INTEGRITY NOTICE: Symptom highlighting and severity calculations are
  experimental visual estimates, NOT validated agronomic disease severity measurements.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image


def assess_image_quality(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Assess basic image quality to detect blur, poor lighting, or non-leaf input.
    """
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = laplacian_var < 60.0

    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    is_underexposed = mean_brightness < 40.0
    is_overexposed = mean_brightness > 225.0
    is_low_contrast = std_contrast < 20.0

    warnings = []
    if is_blurry:
        warnings.append(f"Image may be blurry (Laplacian variance {laplacian_var:.1f} < 60.0)")
    if is_underexposed:
        warnings.append("Image appears underexposed / too dark")
    if is_overexposed:
        warnings.append("Image appears overexposed / washed out")
    if is_low_contrast:
        warnings.append("Image has very low contrast")

    return {
        "laplacian_variance": round(laplacian_var, 1),
        "is_blurry": is_blurry,
        "mean_brightness": round(mean_brightness, 1),
        "std_contrast": round(std_contrast, 1),
        "warnings": warnings,
        "is_acceptable": len(warnings) == 0
    }


def segment_leaf_foliage(rgb: np.ndarray) -> np.ndarray:
    """
    Segment the leaf foliage from the background using combined LAB and HSV color spaces.
    Returns binary mask (255 for leaf, 0 for background).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    lower_hsv = np.array([15, 20, 20])
    upper_hsv = np.array([100, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    a_channel = lab[:, :, 1]
    mask_lab = cv2.inRange(a_channel, 0, 126)

    combined = cv2.bitwise_or(mask_hsv, mask_lab)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros_like(cleaned)
    total_area = rgb.shape[0] * rgb.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (0.01 * total_area):
            cv2.drawContours(leaf_mask, [cnt], -1, 255, -1)

    if cv2.countNonZero(leaf_mask) == 0:
        leaf_mask = cleaned

    return leaf_mask


def get_experimental_symptom_mask(rgb: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """
    Experimental visual symptom segmentation using heuristic HSV color thresholds.
    NOTE: Heuristic visual aid, NOT a trained deep learning disease detector.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    brown_lower = np.array([3, 30, 20])
    brown_upper = np.array([30, 255, 210])
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

    dark_lower = np.array([0, 0, 0])
    dark_upper = np.array([180, 255, 65])
    dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)

    yellow_lower = np.array([18, 70, 70])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    symptom_raw = cv2.bitwise_or(brown_mask, dark_mask)
    symptom_raw = cv2.bitwise_or(symptom_raw, yellow_mask)
    symptom_mask = cv2.bitwise_and(symptom_raw, leaf_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    symptom_mask = cv2.morphologyEx(symptom_mask, cv2.MORPH_OPEN, kernel)
    symptom_mask = cv2.morphologyEx(symptom_mask, cv2.MORPH_CLOSE, kernel)

    return symptom_mask


def analyze_severity_and_vision(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Execute Computer Vision pipeline:
    - Foliage segmentation
    - Quality check
    - Experimental symptom segmentation
    - Mathematical severity estimation: (affected_leaf_pixels / total_valid_leaf_pixels) * 100
    - Heatmap and annotated visualization
    """
    rgb = np.array(pil_img.convert("RGB"))
    height, width = rgb.shape[:2]
    total_frame_pixels = height * width

    quality = assess_image_quality(pil_img)
    leaf_mask = segment_leaf_foliage(rgb)
    leaf_pixels = int(cv2.countNonZero(leaf_mask))
    leaf_coverage_pct = round((leaf_pixels / total_frame_pixels) * 100.0, 1)

    if leaf_pixels < (0.02 * total_frame_pixels) or leaf_pixels == 0:
        return {
            "valid": False,
            "reason": f"Insufficient leaf foliage detected ({leaf_coverage_pct}% of frame). Please capture or upload a clear leaf against a neutral background.",
            "leaf_coverage_pct": leaf_coverage_pct,
            "affected_area_pct": None,
            "severity_level": "Undetermined",
            "leaf_mask": leaf_mask,
            "symptom_mask": np.zeros_like(leaf_mask),
            "heatmap": rgb.copy(),
            "annotated": rgb.copy(),
            "quality": quality,
            "calculation_formula": "Affected Area (%) = 100 * (affected_leaf_pixels / total_valid_leaf_pixels)",
            "nature_of_estimate": "Experimental visual estimate (heuristic color segmentation - not a validated agronomic severity measurement)",
            "metrics": {
                "total_frame_pixels": total_frame_pixels,
                "leaf_pixels": leaf_pixels,
                "affected_pixels": 0,
            }
        }

    symptom_mask = get_experimental_symptom_mask(rgb, leaf_mask)
    affected_pixels = int(cv2.countNonZero(symptom_mask))

    affected_area_pct = round((affected_pixels / leaf_pixels) * 100.0, 1)
    affected_area_pct = min(100.0, max(0.0, affected_area_pct))

    if affected_area_pct < 10.0:
        severity_level = "Mild"
        severity_color = "green"
    elif affected_area_pct <= 30.0:
        severity_level = "Moderate"
        severity_color = "amber"
    else:
        severity_level = "Severe"
        severity_color = "red"

    blurred_mask = cv2.GaussianBlur(symptom_mask, (25, 25), 0)
    heatmap_colored = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_rgb[leaf_mask == 0] = [10, 25, 40]

    annotated = rgb.copy()
    contours, _ = cv2.findContours(symptom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    damage_boxes = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 25:
            continue
        damage_boxes += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 60, 60), 2)
        cv2.drawContours(annotated, [cnt], -1, (255, 200, 0), 1)

    return {
        "valid": True,
        "reason": "Valid leaf mask identified and analyzed.",
        "leaf_coverage_pct": leaf_coverage_pct,
        "affected_area_pct": affected_area_pct,
        "severity_level": severity_level,
        "severity_color": severity_color,
        "symptom_contour_count": damage_boxes,
        "leaf_mask": leaf_mask,
        "symptom_mask": symptom_mask,
        "heatmap": heatmap_rgb,
        "annotated": annotated,
        "quality": quality,
        "calculation_formula": "Affected Area (%) = 100 * (affected_leaf_pixels / total_valid_leaf_pixels)",
        "nature_of_estimate": "Experimental visual estimate (heuristic color segmentation - not a validated agronomic severity measurement)",
        "metrics": {
            "total_frame_pixels": total_frame_pixels,
            "leaf_pixels": leaf_pixels,
            "affected_pixels": affected_pixels,
        }
    }
