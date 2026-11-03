"""
Architecture, Dataset & Class Catalog for PlantGuard AI.

Defines:
1. Active 15 classes in models/plant_model.keras.
2. PlantVillage official 38-class taxonomy across 14 crops.
3. PlantDoc 29-class field image validation taxonomy.
4. IP102 102-class pest recognition taxonomy (target 12 agricultural pests).
5. CoLeaf 10-class nutritional deficiency taxonomy.
6. Provenance and dataset manifests.
"""

from typing import Dict, Any, List
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "data" / "datasets"

ACTIVE_CLASSES_15 = [
    {"index": 0, "name": "Apple___Apple_scab", "plant": "Apple", "condition": "Apple scab", "type": "Fungal"},
    {"index": 1, "name": "Apple___Black_rot", "plant": "Apple", "condition": "Black rot", "type": "Fungal"},
    {"index": 2, "name": "Apple___healthy", "plant": "Apple", "condition": "Healthy foliage", "type": "Healthy"},
    {"index": 3, "name": "Grape___Black_rot", "plant": "Grape", "condition": "Black rot", "type": "Fungal"},
    {"index": 4, "name": "Grape___Esca", "plant": "Grape", "condition": "Esca (Black Measles)", "type": "Fungal complex"},
    {"index": 5, "name": "Grape___healthy", "plant": "Grape", "condition": "Healthy foliage", "type": "Healthy"},
    {"index": 6, "name": "Mango___Anthracnose", "plant": "Mango", "condition": "Anthracnose", "type": "Fungal"},
    {"index": 7, "name": "Mango___healthy", "plant": "Mango", "condition": "Healthy foliage", "type": "Healthy"},
    {"index": 8, "name": "Potato___Early_blight", "plant": "Potato", "condition": "Early blight", "type": "Fungal (Alternaria)"},
    {"index": 9, "name": "Potato___Late_blight", "plant": "Potato", "condition": "Late blight", "type": "Oomycete (Phytophthora)"},
    {"index": 10, "name": "Potato___healthy", "plant": "Potato", "condition": "Healthy foliage", "type": "Healthy"},
    {"index": 11, "name": "Tomato___Early_blight", "plant": "Tomato", "condition": "Early blight", "type": "Fungal (Alternaria)"},
    {"index": 12, "name": "Tomato___Late_blight", "plant": "Tomato", "condition": "Late blight", "type": "Oomycete (Phytophthora)"},
    {"index": 13, "name": "Tomato___healthy", "plant": "Tomato", "condition": "Healthy foliage", "type": "Healthy"},
    {"index": 14, "name": "Tomato___Good", "plant": "Tomato", "condition": "Good / Healthy foliage", "type": "Healthy"}
]


def load_dataset_manifest(name: str) -> Dict[str, Any]:
    path = DATASETS_DIR / name / "manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_dataset_classes(name: str) -> List[Any]:
    path = DATASETS_DIR / name / "classes.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_architecture_summary() -> Dict[str, Any]:
    return {
        "active_model": {
            "name": "MobileNetV2 Transfer Learning",
            "weights_file": "models/plant_model.keras",
            "status": "Loaded and Operational",
            "classes_count": len(ACTIVE_CLASSES_15),
            "classes": ACTIVE_CLASSES_15,
            "caveat": "The 15 output classes in models/class_names.json do not have an original training script or dataset in the repo."
        },
        "datasets": {
            "plantvillage": load_dataset_manifest("plantvillage"),
            "plantdoc": load_dataset_manifest("plantdoc"),
            "ip102_pest": load_dataset_manifest("ip102_pest"),
            "coleaf_nutrient": load_dataset_manifest("coleaf_nutrient")
        },
        "training_pipelines": {
            "disease": "training/train_disease_classifier.py (PlantVillage 38 classes / PlantDoc validation)",
            "pest": "training/train_pest_model.py (IP102 12 target pest classes)",
            "nutrient": "training/train_nutrient_model.py (CoLeaf 10 nutritional categories)"
        }
    }
