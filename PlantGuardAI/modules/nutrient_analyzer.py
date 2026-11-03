"""
Nutrient Deficiency Analysis Module for PlantGuard AI.

Target Classes: 10 CoLeaf Nutritional Deficiency Categories (Mendeley Data brfgw46wzb/2).
Architecture: MobileNetV2 Transfer Learning (No arbitrary color heuristics).
Pipeline: training/train_nutrient_model.py
Status: Architecture checkpoint ready; training on CoLeaf dataset pending.
SCIENTIFIC INTEGRITY: Classification predictions from the untrained checkpoint are disabled.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
NUTRIENT_MODEL_PATH = MODELS_DIR / "nutrient_model.keras"
NUTRIENT_CLASSES_PATH = MODELS_DIR / "nutrient_classes.json"
COLEAF_DIR = BASE_DIR / "data" / "datasets" / "coleaf_nutrient"

_cached_nutrient_classes = None


def load_nutrient_classes() -> List[str]:
    global _cached_nutrient_classes
    if _cached_nutrient_classes is None:
        if NUTRIENT_CLASSES_PATH.exists():
            with open(NUTRIENT_CLASSES_PATH, "r", encoding="utf-8") as f:
                _cached_nutrient_classes = json.load(f)
        elif (COLEAF_DIR / "classes.json").exists():
            with open(COLEAF_DIR / "classes.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            _cached_nutrient_classes = [item["nutrient"] for item in data]
        else:
            _cached_nutrient_classes = [
                "Healthy (Optimal Nutrition)",
                "Nitrogen (N) Deficiency",
                "Phosphorus (P) Deficiency",
                "Potassium (K) Deficiency",
                "Calcium (Ca) Deficiency",
                "Magnesium (Mg) Deficiency",
                "Iron (Fe) Chlorosis",
                "Boron (B) Deficiency",
                "Manganese (Mn) Deficiency",
                "Combined / Multiple Deficiencies"
            ]
    return _cached_nutrient_classes


def get_nutrient_status() -> Dict[str, Any]:
    classes = load_nutrient_classes()
    has_checkpoint = NUTRIENT_MODEL_PATH.exists()
    return {
        "status": "Pending Training",
        "has_checkpoint": has_checkpoint,
        "is_fully_trained": False,
        "status_message": "Nutrient deficiency model pending training on CoLeaf dataset.",
        "target_classes": classes,
        "num_classes": len(classes),
        "dataset_source": "CoLeaf-DB: Peruvian Coffee Leaf Nutritional Deficiencies (Mendeley Data brfgw46wzb/2)",
        "training_pipeline": "training/train_nutrient_model.py",
        "documentation": (
            "Target classes are planned capabilities sourced from the CoLeaf benchmark taxonomy. "
            "Arbitrary color thresholding is disabled. The transfer learning pipeline is constructed. "
            "To preserve scientific integrity, inference from the untrained checkpoint is disabled."
        )
    }


def analyze_nutrients(image: Image.Image) -> Dict[str, Any]:
    """
    Returns nutrient analysis status.
    Inference is explicitly disabled to avoid presenting random weights as real diagnosis.
    """
    status = get_nutrient_status()
    return {
        "model_available": False,
        "pipeline_ready": True,
        "status": "Pending Training",
        "status_message": "Nutrient deficiency model pending training on CoLeaf dataset.",
        "top_candidate": None,
        "confidence": None,
        "target_classes": status["target_classes"],
        "note": (
            "Nutrient analysis model is pending training on the CoLeaf dataset (1,006 genuine coffee leaf images). "
            "Untrained checkpoint predictions are disabled to maintain scientific integrity."
        )
    }
