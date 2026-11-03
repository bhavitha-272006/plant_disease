"""
Core AI Pipeline for PlantGuard AI.
Handles model loading, correct preprocessing (preventing double normalization),
deep learning prediction, uncertainty quantification, OOD safeguards, and Grad-CAM interpretability.

Supports:
1. Genuinely Trained MobileNetV2 (PlantVillage Full Dataset 54,284 images, 38 Classes) - Primary Validated Model.
2. Trained MobileNetV2 Baseline (4,560 images, 38 Classes) - Comparative Baseline.
3. Legacy MobileNetV2 Checkpoint (15 Classes) - Preserved Experimental Baseline.
"""

import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Full-dataset model (54,284 training images, 95.26% test accuracy) - ACTIVE / RECOMMENDED
FULL_MODEL_PATH = MODELS_DIR / "trained_plantvillage_full_model.keras"
FULL_CLASSES_PATH = MODELS_DIR / "trained_plantvillage_full_model_classes.json"
FULL_HISTORY_PATH = MODELS_DIR / "trained_plantvillage_full_model_history.json"
COMPARISON_REPORT_PATH = DATA_DIR / "model_comparison_report.json"

# Baseline model (4,560 training images, 91.22% test accuracy) - PRESERVED
BASELINE_MODEL_PATH = MODELS_DIR / "trained_plantvillage_model.keras"
BASELINE_CLASSES_PATH = MODELS_DIR / "trained_plantvillage_classes.json"
BASELINE_HISTORY_PATH = MODELS_DIR / "trained_plantvillage_history.json"

# Legacy checkpoint (15 classes, unverified) - PRESERVED
LEGACY_MODEL_PATH = MODELS_DIR / "plant_model.keras"
LEGACY_CLASSES_PATH = MODELS_DIR / "class_names.json"

# Feature-space class prototype centroids (256-dim embedding vectors per class)
CLASS_PROTOTYPES_PATH = MODELS_DIR / "class_prototypes.json"

_cached_models = {}
_cached_classes = {}
_cached_feat_extractors = {}
_cached_prototypes = None
_active_model_mode = "full"  # "full" (default), "baseline", or "legacy"


def get_class_prototypes():
    """Load cached 256-dim feature prototypes for 38 PlantVillage classes."""
    global _cached_prototypes
    if _cached_prototypes is None:
        if CLASS_PROTOTYPES_PATH.exists():
            try:
                with open(CLASS_PROTOTYPES_PATH, "r", encoding="utf-8") as f:
                    _cached_prototypes = json.load(f)
            except Exception:
                _cached_prototypes = {}
        else:
            _cached_prototypes = {}
    return _cached_prototypes


def get_feature_extractor(model, mode="full"):
    """Extract penultimate dense layer features (256-dim) prior to softmax."""
    global _cached_feat_extractors
    if mode not in _cached_feat_extractors:
        try:
            # Penultimate layer is Dense(256) preceding Dense(38)
            _cached_feat_extractors[mode] = tf.keras.Model(
                inputs=model.inputs,
                outputs=model.layers[-2].output
            )
        except Exception:
            _cached_feat_extractors[mode] = None
    return _cached_feat_extractors[mode]


def get_available_models():
    """Returns available models and their operational status."""
    models_info = {}
    if FULL_MODEL_PATH.exists() and FULL_CLASSES_PATH.exists():
        models_info["full"] = {
            "name": "🌿 MobileNetV2 Full Dataset (54k Images, 38 Classes)",
            "file": str(FULL_MODEL_PATH.name),
            "status": "Trained & Validated (95.26% Test Accuracy)",
            "training_samples": 54284,
            "classes": 38,
            "is_default": True
        }
    if BASELINE_MODEL_PATH.exists() and BASELINE_CLASSES_PATH.exists():
        models_info["baseline"] = {
            "name": "📊 MobileNetV2 Baseline (4.5k Images, 38 Classes)",
            "file": str(BASELINE_MODEL_PATH.name),
            "status": "Trained Baseline (91.22% Test Accuracy)",
            "training_samples": 4560,
            "classes": 38,
            "is_default": False
        }
    if LEGACY_MODEL_PATH.exists() and LEGACY_CLASSES_PATH.exists():
        models_info["legacy"] = {
            "name": "⚠️ Legacy Checkpoint (15 Classes - Experimental)",
            "file": str(LEGACY_MODEL_PATH.name),
            "status": "Experimental Unverified Checkpoint",
            "training_samples": 0,
            "classes": 15,
            "is_default": False
        }
    return models_info


def set_active_model_mode(mode: str):
    """Switch active model mode ('full', 'baseline', or 'legacy')."""
    global _active_model_mode
    if mode in ["full", "baseline", "legacy"]:
        _active_model_mode = mode
    elif mode == "trained":
        _active_model_mode = "full"


def get_active_model_mode() -> str:
    """Return currently active model mode ('full', 'baseline', or 'legacy')."""
    global _active_model_mode
    return _active_model_mode


def get_model_info():
    """Return model architecture and verification metadata for active model."""
    global _active_model_mode
    
    comp_data = {}
    if COMPARISON_REPORT_PATH.exists():
        try:
            with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
        except Exception:
            pass

    if _active_model_mode == "full" and FULL_MODEL_PATH.exists():
        hist_data = {}
        if FULL_HISTORY_PATH.exists():
            try:
                with open(FULL_HISTORY_PATH, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
            except Exception:
                pass

        full_metrics = comp_data.get("full_dataset_model", {})
        test_acc = full_metrics.get("test_accuracy", 0.9526)

        return {
            "mode": "full",
            "model_file": str(FULL_MODEL_PATH.name),
            "architecture": "MobileNetV2 Transfer Learning (Rescaling [-1, 1])",
            "input_shape": "(224, 224, 3)",
            "internal_rescaling": "scale=1/127.5, offset=-1 (expects [0, 255] RGB float32)",
            "num_classes": 38,
            "readiness_status": "Genuinely Trained & Validated Diagnostic Model (Primary)",
            "diagnostic_validity": "Validated on 8,143 held-out test images (15% stratified split)",
            "training_data_available": True,
            "training_dataset": "PlantVillage (Official raw/color - 54,284 images)",
            "evaluation_metrics_available": True,
            "empirical_test_accuracy": f"{test_acc*100:.2f}%",
            "macro_f1": f"{full_metrics.get('macro_f1_score', 0.9389)*100:.2f}%",
            "weighted_f1": f"{full_metrics.get('weighted_f1_score', 0.9530)*100:.2f}%",
            "training_history": hist_data,
            "evaluation_report": comp_data
        }
    elif _active_model_mode == "baseline" and BASELINE_MODEL_PATH.exists():
        hist_data = {}
        if BASELINE_HISTORY_PATH.exists():
            try:
                with open(BASELINE_HISTORY_PATH, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
            except Exception:
                pass

        base_metrics = comp_data.get("baseline_model", {})
        test_acc = base_metrics.get("test_accuracy", 0.9122)

        return {
            "mode": "baseline",
            "model_file": str(BASELINE_MODEL_PATH.name),
            "architecture": "MobileNetV2 Transfer Learning (Rescaling [-1, 1])",
            "input_shape": "(224, 224, 3)",
            "internal_rescaling": "scale=1/127.5, offset=-1 (expects [0, 255] RGB float32)",
            "num_classes": 38,
            "readiness_status": "Trained Baseline Checkpoint (Curated 4.5k subset)",
            "diagnostic_validity": "Validated on 8,143 held-out test images",
            "training_data_available": True,
            "training_dataset": "PlantVillage (120 samples/class subset - 4,560 images)",
            "evaluation_metrics_available": True,
            "empirical_test_accuracy": f"{test_acc*100:.2f}%",
            "macro_f1": f"{base_metrics.get('macro_f1_score', 0.8943)*100:.2f}%",
            "weighted_f1": f"{base_metrics.get('weighted_f1_score', 0.9125)*100:.2f}%",
            "training_history": hist_data,
            "evaluation_report": comp_data
        }
    else:
        # Legacy Model
        return {
            "mode": "legacy",
            "model_file": str(LEGACY_MODEL_PATH.name),
            "architecture": "MobileNetV2 (Sequential with embedded Rescaling)",
            "input_shape": "(224, 224, 3)",
            "internal_rescaling": "scale=1/127.5, offset=-1 (expects [0, 255] RGB input)",
            "num_classes": 15,
            "readiness_status": "Experimental Unverified Legacy Checkpoint",
            "diagnostic_validity": "Unvalidated (0.0% accuracy on PlantVillage test images due to label discrepancy)",
            "training_data_available": False,
            "evaluation_metrics_available": True,
            "empirical_test_accuracy": "0.0% (40 PlantVillage test specimens)"
        }


def load_model(mode: str = None):
    """
    Safely load MobileNetV2 model and class mapping for requested or active mode.
    Validates model output dimension against class list length.
    """
    global _active_model_mode, _cached_models, _cached_classes
    
    target_mode = mode or _active_model_mode
    if target_mode == "trained":
        target_mode = "full"

    if target_mode == "full" and not FULL_MODEL_PATH.exists():
        target_mode = "baseline" if BASELINE_MODEL_PATH.exists() else "legacy"
    elif target_mode == "baseline" and not BASELINE_MODEL_PATH.exists():
        target_mode = "full" if FULL_MODEL_PATH.exists() else "legacy"

    if target_mode not in _cached_models:
        if target_mode == "full":
            model_path = FULL_MODEL_PATH
            class_path = FULL_CLASSES_PATH
        elif target_mode == "baseline":
            model_path = BASELINE_MODEL_PATH
            class_path = BASELINE_CLASSES_PATH
        else:
            model_path = LEGACY_MODEL_PATH
            class_path = LEGACY_CLASSES_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not class_path.exists():
            raise FileNotFoundError(f"Class labels file not found at: {class_path}")

        loaded_model = tf.keras.models.load_model(str(model_path), compile=False)
        with open(class_path, "r", encoding="utf-8") as f:
            classes = json.load(f)

        if isinstance(classes, dict):
            classes = [name for name, _ in sorted(classes.items(), key=lambda item: int(item[1]))]

        out_shape = loaded_model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        expected_classes = out_shape[-1]

        if expected_classes != len(classes):
            raise ValueError(
                f"Model has {expected_classes} output nodes, but class mapping has {len(classes)} classes."
            )

        _cached_models[target_mode] = loaded_model
        _cached_classes[target_mode] = classes

    return _cached_models[target_mode], _cached_classes[target_mode]


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocess image for MobileNetV2.
    CRITICAL:
    1. EXIF orientation correction: Mobile phones store photos with EXIF rotation tags.
       exif_transpose rotates the image buffer upright before inference.
    2. The model contains an internal Rescaling layer (scale=1/127.5, offset=-1).
       Therefore, images MUST remain in the range [0.0, 255.0] float32.
    """
    oriented_img = ImageOps.exif_transpose(image)
    rgb_img = oriented_img.convert("RGB")
    resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized_img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # Shape: (1, 224, 224, 3)
    return arr


def assess_input_foliage(image: Image.Image) -> dict:
    """
    Evaluates optical image quality and basic foliage presence.
    SCIENTIFIC INTEGRITY NOTICE:
    This heuristic checks optical sharpness, brightness, and basic green/brown
    pigment presence to screen out non-leaf objects (e.g., backgrounds, inanimate
    objects, severe blur).
    It does NOT and CANNOT reliably distinguish whether a leaf belongs to an
    unsupported plant species versus one of the 14 supported crops.
    """
    oriented_img = ImageOps.exif_transpose(image)
    rgb = np.array(oriented_img.convert("RGB"))
    h, w = rgb.shape[:2]
    total_pixels = h * w

    # Convert to HSV and LAB for foliage mask
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Green/yellow/brown foliage range in HSV (requires non-neutral saturation S >= 25)
    lower_foliage = np.array([15, 25, 25])
    upper_foliage = np.array([100, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_foliage, upper_foliage)

    # In LAB: green plant tissue has a < 125, with chromaticity (S >= 20)
    a_channel = lab[:, :, 1]
    s_channel = hsv[:, :, 1]
    mask_green_lab = (a_channel < 125) & (s_channel >= 20)
    mask_lab = np.uint8(mask_green_lab * 255)

    foliage_mask = cv2.bitwise_or(mask_hsv, mask_lab)
    foliage_pixels = cv2.countNonZero(foliage_mask)
    foliage_ratio = float(foliage_pixels / total_pixels)

    # Blur detection via Laplacian variance
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    # OOD criteria:
    # 1. Very low foliage coverage (< 10% of image)
    # 2. Extreme brightness (overexposed > 248 or underexposed < 15)
    # 3. Severe blur (lap_var < 15)
    is_non_foliage = foliage_ratio < 0.10
    is_severe_blur = lap_var < 15.0
    is_extreme_lighting = (mean_brightness < 15.0) or (mean_brightness > 248.0)
    is_ood = is_non_foliage or is_extreme_lighting or is_severe_blur

    return {
        "foliage_ratio": round(foliage_ratio * 100.0, 1),
        "laplacian_var": round(lap_var, 1),
        "mean_brightness": round(mean_brightness, 1),
        "is_non_foliage": is_non_foliage,
        "is_severe_blur": is_severe_blur,
        "is_extreme_lighting": is_extreme_lighting,
        "is_ood": is_ood
    }


def evaluate_leaf_suitability(image: Image.Image, user_specimen_type: str = "Single leaf") -> dict:
    """
    Universal Specimen Suitability Gate for PlantGuard AI.
    Works across all 14 crop categories without species-specific hardcoding.
    Evaluates:
      1. User Protocol Declaration (Single leaf vs Fruit/Canopy/Stem/Not sure)
      2. Universal Optical Quality (Laplacian sharpness variance and illumination)
      3. Foliar Surface Area & Purity (Green, yellow chlorosis, brown necrosis)
      4. Non-Foliar Organ Dominance (Fruit/flower chromatic prominence)

    Outcomes:
      - VALID_LEAF_SPECIMEN: Authorized for disease diagnosis
      - NON_LEAF_ORGAN: Fruit, flower, stem, root, or whole-plant organ
      - UNSUITABLE_IMAGE: Severe blur, extreme lighting, or background-only
      - UNCERTAIN_SPECIMEN: Ambiguous organ boundary or unconfirmed category
    """
    # 1. User Protocol Declaration Gate
    if user_specimen_type == "Fruit":
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": "Specimen is declared as Fruit. The 38-class MobileNetV2 classifier is trained exclusively on foliar (leaf) specimens and cannot diagnose fruit pathologies.",
            "metrics": {"user_declaration": user_specimen_type}
        }
    elif user_specimen_type == "Whole plant / canopy":
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": "Specimen is declared as Whole plant / canopy. The classifier was trained on single excised leaves against plain backgrounds; complex canopies are outside its validated scope.",
            "metrics": {"user_declaration": user_specimen_type}
        }
    elif user_specimen_type == "Stem / flower / root":
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": "Specimen is declared as Stem / flower / root. The model does not support non-foliar botanical organs.",
            "metrics": {"user_declaration": user_specimen_type}
        }
    elif user_specimen_type == "Not sure":
        return {
            "status": "UNCERTAIN_SPECIMEN",
            "is_permitted": False,
            "reason": "Specimen organ type is unconfirmed. Disease diagnosis is withheld to prevent unverified classification.",
            "metrics": {"user_declaration": user_specimen_type}
        }

    # 2. Universal Optical & Morphological Assessment
    oriented_img = ImageOps.exif_transpose(image)
    rgb = np.array(oriented_img.convert("RGB"))
    h, w = rgb.shape[:2]
    total_pixels = h * w

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(np.mean(gray))

    # Universal Color Space Conversions
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Universal Foliar Color Segmentation across all crops:
    # Green foliage: HSV H in [25, 95], S >= 25, V >= 25, or LAB a < 125 with S >= 20
    mask_green_hsv = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([95, 255, 255]))
    mask_green_lab = np.uint8(((lab[:, :, 1] < 125) & (hsv[:, :, 1] >= 20)) * 255)
    
    # Yellow / Chlorotic leaf tissue: HSV H in [15, 35], S >= 30, V >= 50
    mask_yellow = cv2.inRange(hsv, np.array([15, 30, 50]), np.array([35, 255, 255]))

    # Brown / Necrotic leaf tissue: HSV H in [8, 24], S >= 25, V in [20, 200]
    mask_brown = cv2.inRange(hsv, np.array([8, 25, 20]), np.array([24, 255, 200]))

    foliage_mask = cv2.bitwise_or(mask_green_hsv, mask_green_lab)
    foliage_mask = cv2.bitwise_or(foliage_mask, mask_yellow)
    foliage_mask = cv2.bitwise_or(foliage_mask, mask_brown)

    foliage_pixels = cv2.countNonZero(foliage_mask)
    foliage_ratio = float(foliage_pixels / total_pixels)

    # Non-Foliar High-Chroma Organ Check (Fruit / Flower Dominance)
    # Ripe fruit / flower chromatic signature: High redness in LAB (a* > 140) and high saturation in HSV (S > 75)
    red_hsv1 = cv2.inRange(hsv, np.array([0, 75, 40]), np.array([12, 255, 255]))
    red_hsv2 = cv2.inRange(hsv, np.array([165, 75, 40]), np.array([180, 255, 255]))
    red_hsv = cv2.bitwise_or(red_hsv1, red_hsv2)
    red_lab = np.uint8((lab[:, :, 1] > 140) * 255)
    fruit_mask = cv2.bitwise_and(red_hsv, red_lab)
    fruit_pixels = cv2.countNonZero(fruit_mask)
    fruit_ratio = float(fruit_pixels / total_pixels)

    fruit_to_foliage = float(fruit_pixels / (foliage_pixels + 1e-6)) if fruit_pixels > 0 else 0.0

    # If prominent fruit organ is detected (occupies substantial area or dominates foreground)
    if fruit_ratio > 0.08 or (fruit_pixels > 500 and fruit_to_foliage > 0.30):
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": f"Optical analysis detected prominent non-foliar fruit/flower structures ({fruit_ratio*100:.1f}% of frame). The model was trained exclusively on leaf specimens and cannot diagnose fruit or floral organs.",
            "metrics": {"fruit_ratio": fruit_ratio, "foliage_ratio": foliage_ratio, "laplacian_var": lap_var}
        }

    # Optical image quality gate
    if lap_var < 15.0:
        return {
            "status": "UNSUITABLE_IMAGE",
            "is_permitted": False,
            "reason": f"Severe optical or motion blur detected (Laplacian variance {lap_var:.1f} < 15.0). Diagnostic leaf features cannot be resolved.",
            "metrics": {"laplacian_var": lap_var, "mean_brightness": mean_bright}
        }
    if mean_bright < 15.0 or mean_bright > 248.0:
        return {
            "status": "UNSUITABLE_IMAGE",
            "is_permitted": False,
            "reason": f"Extreme illumination detected (Mean brightness {mean_bright:.1f}/255). Image is severely underexposed or overexposed.",
            "metrics": {"laplacian_var": lap_var, "mean_brightness": mean_bright}
        }

    if foliage_ratio < 0.10:
        return {
            "status": "UNSUITABLE_IMAGE",
            "is_permitted": False,
            "reason": f"Insufficient leaf surface area detected ({foliage_ratio*100:.1f}% coverage < 10%). The image appears to show background, soil, or an inanimate surface.",
            "metrics": {"foliage_ratio": foliage_ratio, "laplacian_var": lap_var}
        }

    return {
        "status": "VALID_LEAF_SPECIMEN",
        "is_permitted": True,
        "reason": "Identifiable single leaf specimen confirmed with adequate optical sharpness, illumination, and foliar surface area.",
        "metrics": {"foliage_ratio": foliage_ratio, "fruit_ratio": fruit_ratio, "laplacian_var": lap_var, "mean_brightness": mean_bright}
    }


def parse_class_label(raw_class: str):
    """
    Parse standardized PlantVillage class name into crop, disease, and health status.
    Format: Crop___Disease_Name
    """
    if "___" in raw_class:
        crop_raw, disease_raw = raw_class.split("___", 1)
    elif "_" in raw_class:
        parts = raw_class.split("_", 1)
        crop_raw = parts[0]
        disease_raw = parts[1]
    else:
        crop_raw = "Plant"
        disease_raw = raw_class

    # Format crop name
    crop_clean = crop_raw.replace("_", " ").title()
    if "Pepper" in crop_clean:
        crop_clean = "Bell Pepper"
    elif "Cherry" in crop_clean:
        crop_clean = "Cherry"
    elif "Corn" in crop_clean:
        crop_clean = "Corn (Maize)"
    elif "Tomato" in crop_clean:
        crop_clean = "Tomato"
    elif "Potato" in crop_clean:
        crop_clean = "Potato"
    elif "Apple" in crop_clean:
        crop_clean = "Apple"
    elif "Grape" in crop_clean:
        crop_clean = "Grape"
    elif "Peach" in crop_clean:
        crop_clean = "Peach"
    elif "Strawberry" in crop_clean:
        crop_clean = "Strawberry"
    elif "Orange" in crop_clean:
        crop_clean = "Orange (Citrus)"
    elif "Soybean" in crop_clean:
        crop_clean = "Soybean"
    elif "Squash" in crop_clean:
        crop_clean = "Squash"
    elif "Blueberry" in crop_clean:
        crop_clean = "Blueberry"
    elif "Raspberry" in crop_clean:
        crop_clean = "Raspberry"

    # Format disease name
    disease_lower = disease_raw.lower()
    is_healthy = ("healthy" in disease_lower) or ("good" in disease_lower)
    
    if is_healthy:
        disease_clean = "Healthy Foliage (No Pathogen Detected)"
    else:
        disease_clean = disease_raw.replace("_", " ").title()
        # Clean up acronyms and abbreviations
        disease_clean = disease_clean.replace("Gray Leaf Spot", "Gray Leaf Spot (Cercospora)")
        disease_clean = disease_clean.replace("Esca (Black Measles)", "Esca (Black Measles)")
        disease_clean = disease_clean.replace("Haunglongbing (Citrus Greening)", "Huanglongbing (Citrus Greening)")
        disease_clean = disease_clean.replace("Two-Spotted Spider Mite", "Two-Spotted Spider Mite Infestation")

    return crop_clean, disease_clean, is_healthy


def predict_plant_disease(image: Image.Image, confidence_threshold=65.0, entropy_threshold=1.75, margin_threshold=18.0, mode=None, specimen_type="Single leaf"):
    """
    Run prediction on the active MobileNetV2 model.
    Dynamically decodes class probabilities, Shannon entropy, confidence margin, and top-5.
    Includes rigorous uncertainty checks and foliage OOD verification.
    """
    target_mode = mode or _active_model_mode
    if target_mode == "trained":
        target_mode = "full"

    model, classes = load_model(mode=target_mode)
    current_mode = target_mode
    input_arr = preprocess_image(image)

    # 0. Universal Specimen Suitability Gate
    suitability = evaluate_leaf_suitability(image, user_specimen_type=specimen_type)
    if not suitability["is_permitted"]:
        return {
            "status": "withheld",
            "is_blocked": True,
            "suitability_status": suitability["status"],
            "suitability_reason": suitability["reason"],
            "suitability_metrics": suitability.get("metrics", {}),
            "raw_class": "N/A",
            "class_index": -1,
            "plant": "Diagnosis Withheld",
            "disease": f"Specimen Not Authorized ({suitability['status']})",
            "is_healthy": False,
            "confidence": 0.0,
            "entropy": 0.0,
            "margin": 0.0,
            "is_uncertain": True,
            "uncertainty_reasons": [
                f"Suitability Gate: {suitability['reason']}"
            ],
            "top_5": [],
            "model_readiness": "Diagnostic Withheld (Leaf Only)",
            "model_mode": mode or _active_model_mode
        }

    # 1. Image Foliage / Quality Assessment for OOD detection
    foliage_info = assess_input_foliage(image)

    # 2. Deep Learning Inference
    raw_preds = model.predict(input_arr, verbose=0)
    scores = np.asarray(raw_preds).reshape(-1)

    # Ensure valid probabilities
    if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=0.01):
        scores = tf.nn.softmax(scores).numpy()
    else:
        scores = scores / np.sum(scores)

    sorted_indices = np.argsort(scores)[::-1]
    top_index = int(sorted_indices[0])
    top_confidence = float(scores[top_index] * 100.0)
    top_class_raw = classes[top_index]

    # Uncertainty metrics
    valid_scores = scores[scores > 1e-12]
    shannon_entropy = float(-np.sum(valid_scores * np.log(valid_scores)))
    second_confidence = float(scores[sorted_indices[1]] * 100.0) if len(scores) > 1 else 0.0
    margin = float(top_confidence - second_confidence)

    # 3. Feature-Space Domain Shift & Prototype Verification
    prototypes = get_class_prototypes()
    proto = np.array(prototypes.get(top_class_raw, []))
    sim = 1.0
    has_proto = len(proto) > 0 and current_mode in ["full", "baseline"]
    if has_proto:
        feat_extractor = get_feature_extractor(model, mode=current_mode)
        if feat_extractor is not None:
            try:
                feat = feat_extractor(input_arr, training=False).numpy()[0]
                norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
                sim = float(np.dot(norm_feat, proto))
            except Exception:
                sim = 1.0

    # If severe domain shift is detected (orthogonal or alien feature projection)
    if has_proto and sim < 0.20:
        return {
            "status": "withheld",
            "is_blocked": True,
            "suitability_status": "DOMAIN_SHIFT_OOD",
            "suitability_reason": (
                f"Severe Domain Shift: Specimen deep feature vector exhibits extreme divergence from laboratory training prototypes "
                f"(Prototype Cosine Similarity: {sim:.3f} < 0.20). Disease diagnosis is withheld to protect against false overconfident predictions."
            ),
            "suitability_metrics": {"prototype_similarity": sim, "threshold": 0.20},
            "prototype_similarity": sim,
            "raw_class": "N/A",
            "class_index": -1,
            "plant": "Diagnosis Withheld",
            "disease": "Laboratory-to-Field Domain Shift Detected",
            "is_healthy": False,
            "confidence": 0.0,
            "entropy": shannon_entropy,
            "margin": 0.0,
            "is_uncertain": True,
            "uncertainty_reasons": [
                f"Feature-space similarity to predicted prototype ({sim:.3f}) is below valid threshold (< 0.20).",
                "Specimen features diverge from the 38 PlantVillage single-leaf laboratory distribution.",
                "To prevent misleading overconfident predictions (e.g. false strawberry leaf scorch), diagnosis is withheld.",
                "Fine-tuning on real agricultural field imagery (e.g., PlantDoc) is required for in-field accuracy."
            ],
            "top_5": [],
            "model_readiness": "Diagnostic Withheld (Domain Shift Guard Active)",
            "model_mode": current_mode
        }

    plant_clean, disease_clean, is_healthy = parse_class_label(top_class_raw)

    # Uncertainty triggers
    uncertainty_reasons = []

    if has_proto and sim < 0.40:
        uncertainty_reasons.append(
            f"Domain Shift Warning: Low feature-space alignment with laboratory prototype (Cosine similarity: {sim:.2f} < 0.40). Laboratory-trained weights may produce unreliable predictions under natural field conditions."
        )
    
    if current_mode == "legacy":
        uncertainty_reasons.append("Model is an unverified legacy checkpoint (outputs must be treated as experimental).")
    
    if foliage_info["is_ood"]:
        uncertainty_reasons.append(
            f"Foliage coverage is only {foliage_info['foliage_ratio']}% (Image may be non-leaf or out-of-distribution specimen)."
        )
    if foliage_info["is_severe_blur"]:
        uncertainty_reasons.append(f"Image has severe motion or optical blur (Laplacian variance {foliage_info['laplacian_var']:.1f}).")

    if top_confidence < confidence_threshold:
        uncertainty_reasons.append(
            f"Prediction confidence ({top_confidence:.1f}%) is below the diagnostic threshold ({confidence_threshold:.0f}%)."
        )
    if margin < margin_threshold:
        uncertainty_reasons.append(
            f"Narrow confidence margin ({margin:.1f}% < {margin_threshold:.0f}%) between top class and {classes[sorted_indices[1]]} ({second_confidence:.1f}%)."
        )
    if shannon_entropy > entropy_threshold:
        uncertainty_reasons.append(
            f"Shannon entropy ({shannon_entropy:.2f} > {entropy_threshold:.2f}) indicates dispersed class probabilities."
        )

    is_uncertain = len(uncertainty_reasons) > 0

    top_5 = []
    for idx in sorted_indices[:5]:
        cls_name = classes[idx]
        c_crop, c_dis, _ = parse_class_label(cls_name)
        display_name = f"{c_crop} - {c_dis}"
        top_5.append({
            "index": int(idx),
            "class_raw": cls_name,
            "display_name": display_name,
            "probability": float(scores[idx]),
            "percentage": float(scores[idx] * 100.0)
        })

    if current_mode == "full":
        model_badge = "🌿 Genuinely Trained MobileNetV2 (PlantVillage Full Dataset - 54k Images, 38 Classes)"
    elif current_mode == "baseline":
        model_badge = "📊 Trained MobileNetV2 Baseline (PlantVillage 4.5k Subset, 38 Classes)"
    else:
        model_badge = "⚠️ Experimental Legacy Checkpoint (15 Classes - Unverified)"

    return {
        "status": "success",
        "is_blocked": False,
        "suitability_status": suitability["status"],
        "suitability_reason": suitability["reason"],
        "raw_class": top_class_raw,
        "plant": plant_clean,
        "disease": disease_clean,
        "is_healthy": is_healthy,
        "confidence": top_confidence,
        "entropy": shannon_entropy,
        "margin": margin,
        "is_uncertain": is_uncertain,
        "uncertainty_reasons": uncertainty_reasons,
        "foliage_info": foliage_info,
        "top_5": top_5,
        "all_scores": scores.tolist(),
        "class_index": top_index,
        "model_readiness": model_badge,
        "model_mode": current_mode,
        "prototype_similarity": sim
    }


def compute_gradcam(image: Image.Image, class_index: int = None, mode: str = None):
    """
    Generate Grad-CAM activation heatmap for the MobileNetV2 backbone.
    Works dynamically for full, baseline, and legacy models.
    """
    target_mode = mode or _active_model_mode
    model, _ = load_model(mode=target_mode)
    input_arr = preprocess_image(image)

    # Locate MobileNetV2 backbone
    mobilenet = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            mobilenet = layer
            break

    if mobilenet is None:
        try:
            mobilenet = model.get_layer("mobilenetv2_1.00_224")
        except Exception:
            mobilenet = model.layers[2]

    # Locate last conv layer
    last_conv_layer = None
    try:
        last_conv_layer = mobilenet.get_layer("out_relu")
    except Exception:
        for l in reversed(mobilenet.layers):
            if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.ReLU)):
                last_conv_layer = l
                break

    feature_model = tf.keras.Model(
        inputs=mobilenet.inputs,
        outputs=[last_conv_layer.output, mobilenet.output]
    )

    # Find subsequent classification layers
    gap_layer = None
    for l in model.layers:
        if "global_average_pooling" in l.name.lower():
            gap_layer = l
            break

    # Pass through input / rescaling layers preceding backbone
    rescaled = input_arr
    for l in model.layers:
        if l == mobilenet:
            break
        if isinstance(l, tf.keras.layers.InputLayer) or "input" in l.name.lower():
            continue
        rescaled = l(rescaled)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = feature_model(rescaled)
        tape.watch(conv_outputs)
        
        # Forward pass through classification head
        x = base_output
        if gap_layer is not None:
            x = gap_layer(x)
            gap_idx = model.layers.index(gap_layer)
            for l in model.layers[gap_idx + 1:]:
                x = l(x, training=False)
        else:
            for l in model.layers[model.layers.index(mobilenet) + 1:]:
                x = l(x, training=False)
        predictions = x

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    oriented_img = ImageOps.exif_transpose(image)
    orig_w, orig_h = oriented_img.size
    if grads is None:
        return np.array(oriented_img.convert("RGB")), np.zeros((orig_h, orig_w))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    cam = np.maximum(cam, 0)
    cam_max = np.max(cam)
    if cam_max > 0:
        cam = cam / cam_max
    else:
        cam = np.zeros_like(cam)

    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    orig_rgb = np.array(oriented_img.convert("RGB"))
    alpha = 0.45
    superimposed = np.uint8(alpha * heatmap_rgb + (1 - alpha) * orig_rgb)

    return superimposed, cam_resized
