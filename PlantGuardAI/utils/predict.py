"""
Predict utility module for PlantGuard AI.
Fixed double-normalization bug and routes to verified core AI pipeline.
"""

from PIL import Image
from modules.core_ai import load_model, predict_plant_disease, get_model_info, compute_gradcam

# Threshold retained for backwards compatibility
CONFIDENCE_THRESHOLD = 50.0


def predict(image: Image.Image):
    """
    Predict plant and disease for given PIL image.
    Preserves input [0, 255] float32 to prevent double normalization in MobileNetV2.
    """
    res = predict_plant_disease(image, confidence_threshold=CONFIDENCE_THRESHOLD)

    if res["is_uncertain"]:
        return "Unknown___Uncertain", res["confidence"]

    return res["raw_class"], res["confidence"]
