"""
Disease Progression Tracking Module for PlantGuard AI.

Compares genuine historical scans or user-selected images over time.
Calculates actual difference in affected leaf area, confidence drift,
and generates authentic visual difference heatmaps.
Does NOT generate fabricated Day 1 / Day 5 examples.
"""

from typing import Dict, Any, Tuple
from datetime import datetime
import numpy as np
import cv2
from PIL import Image


def compute_image_difference(img_a: Image.Image, img_b: Image.Image) -> Tuple[np.ndarray, float]:
    """
    Compute authentic pixel difference heatmap between two aligned scans.
    """
    size = (300, 300)
    arr_a = np.array(img_a.convert("RGB").resize(size), dtype=np.float32)
    arr_b = np.array(img_b.convert("RGB").resize(size), dtype=np.float32)

    # Compute absolute RGB delta
    diff = np.abs(arr_b - arr_a)
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Calculate mean difference percentage
    mean_diff_pct = float(np.mean(diff_gray) / 255.0 * 100.0)

    # Colorize difference map
    diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_MAGMA)
    diff_rgb = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    return diff_rgb, round(mean_diff_pct, 2)


def compare_scans(scan_a: Dict[str, Any], scan_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two genuine scans from history.
    scan_a: Earlier scan
    scan_b: Later scan
    """
    # Timestamp parsing
    time_a = scan_a.get("timestamp", "Earlier Scan")
    time_b = scan_b.get("timestamp", "Later Scan")

    # Severity comparison
    sev_a = scan_a.get("affected_area_pct", 0.0) or 0.0
    sev_b = scan_b.get("affected_area_pct", 0.0) or 0.0
    delta_severity = round(sev_b - sev_a, 1)

    # Confidence comparison
    conf_a = scan_a.get("confidence", 0.0) or 0.0
    conf_b = scan_b.get("confidence", 0.0) or 0.0
    delta_conf = round(conf_b - conf_a, 1)

    # Progression status
    if delta_severity > 5.0:
        trend = "Worsened"
        trend_color = "red"
        interpretation = f"Symptomatic area expanded by +{delta_severity}% across scans."
    elif delta_severity < -5.0:
        trend = "Improved"
        trend_color = "green"
        interpretation = f"Symptomatic area contracted by {delta_severity}% across scans."
    else:
        trend = "Stable"
        trend_color = "amber"
        interpretation = f"Symptomatic area remained approximately stable (delta {delta_severity}%)."

    # Visual difference map if images are present
    diff_map = None
    diff_pct = None
    if "original_image" in scan_a and "original_image" in scan_b:
        try:
            diff_map, diff_pct = compute_image_difference(
                scan_a["original_image"], scan_b["original_image"]
            )
        except Exception:
            diff_map = None

    return {
        "scan_a_time": time_a,
        "scan_b_time": time_b,
        "scan_a_disease": scan_a.get("disease", "Unknown"),
        "scan_b_disease": scan_b.get("disease", "Unknown"),
        "scan_a_severity": sev_a,
        "scan_b_severity": sev_b,
        "delta_severity": delta_severity,
        "delta_confidence": delta_conf,
        "trend": trend,
        "trend_color": trend_color,
        "interpretation": interpretation,
        "visual_difference_map": diff_map,
        "mean_pixel_shift_pct": diff_pct
    }

