"""
Pest Detection Module for PlantGuard AI.

Target Classes: 12 Key Agricultural Pests mapped to IP102 Benchmark (xpwu95/IP102).
Architecture: MobileNetV2 / YOLO transfer learning.
Pipeline: training/train_pest_model.py
Status: Architecture checkpoint ready; training on IP102 field dataset pending.
SCIENTIFIC INTEGRITY: Classification predictions from the untrained checkpoint are disabled.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PEST_MODEL_PATH = MODELS_DIR / "pest_model.keras"
PEST_CLASSES_PATH = MODELS_DIR / "pest_classes.json"
IP102_DIR = BASE_DIR / "data" / "datasets" / "ip102_pest"

_cached_pest_classes = None


def load_pest_classes() -> List[str]:
    global _cached_pest_classes
    if _cached_pest_classes is None:
        if PEST_CLASSES_PATH.exists():
            with open(PEST_CLASSES_PATH, "r", encoding="utf-8") as f:
                _cached_pest_classes = json.load(f)
        elif (IP102_DIR / "target12_mapping.json").exists():
            with open(IP102_DIR / "target12_mapping.json", "r", encoding="utf-8") as f:
                targets = json.load(f)
            _cached_pest_classes = [t["target_name"] for t in targets]
        else:
            _cached_pest_classes = [
                "Aphids", "Armyworm", "Beetles (Flea Beetles)", "Bollworm", "Caterpillars (Leaf Roller)",
                "Grasshoppers", "Leafhoppers", "Mites (Spider Mites)", "Sawfly",
                "Stem Borer", "Thrips", "Whitefly"
            ]
    return _cached_pest_classes


def get_pest_status() -> Dict[str, Any]:
    classes = load_pest_classes()
    has_checkpoint = PEST_MODEL_PATH.exists()
    return {
        "status": "Pending Training",
        "has_checkpoint": has_checkpoint,
        "is_fully_trained": False,
        "status_message": "Pest detection model pending training on IP102 dataset.",
        "target_classes": classes,
        "num_classes": len(classes),
        "dataset_source": "IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition (CVPR 2019)",
        "training_pipeline": "training/train_pest_model.py",
        "documentation": (
            "Target classes are planned capabilities mapped to the IP102 benchmark taxonomy. "
            "The transfer learning architecture (MobileNetV2) and training pipeline are constructed. "
            "To preserve scientific integrity, inference from the untrained checkpoint is disabled."
        )
    }


def analyze_pest(image: Image.Image) -> Dict[str, Any]:
    """
    Returns pest analysis status.
    Inference is explicitly disabled to avoid presenting random weights as real diagnosis.
    """
    status = get_pest_status()
    return {
        "model_available": False,
        "pipeline_ready": True,
        "status": "Pending Training",
        "status_message": "Pest detection model pending training on IP102 dataset.",
        "top_candidate": None,
        "confidence": None,
        "target_classes": status["target_classes"],
        "note": (
            "Pest detection model is pending training on the IP102 dataset (75,000+ field images). "
            "Untrained checkpoint predictions are disabled to maintain scientific integrity."
        )
    }
