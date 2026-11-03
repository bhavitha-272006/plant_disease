import re
from pathlib import Path

core_path = Path("PlantGuardAI/modules/core_ai.py")
text = core_path.read_text(encoding="utf-8")

# 1. Add prototype caching and feature extractor helpers
old_block1 = """# Legacy checkpoint (15 classes, unverified) - PRESERVED
LEGACY_MODEL_PATH = MODELS_DIR / "plant_model.keras"
LEGACY_CLASSES_PATH = MODELS_DIR / "class_names.json"

_cached_models = {}
_cached_classes = {}
_active_model_mode = "full"  # "full" (default), "baseline", or "legacy"
"""

new_block1 = """# Legacy checkpoint (15 classes, unverified) - PRESERVED
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
    \"\"\"Load cached 256-dim feature prototypes for 38 PlantVillage classes.\"\"\"
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
    \"\"\"Extract penultimate dense layer features (256-dim) prior to softmax.\"\"\"
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
"""

# Normalize CRLF for matching
text_lf = text.replace("\r\n", "\n")
assert old_block1 in text_lf, "old_block1 not found in core_ai.py"
text_lf = text_lf.replace(old_block1, new_block1, 1)

# 2. Update predict_plant_disease to perform feature-space prototype similarity check
old_pred_block = """    plant_clean, disease_clean, is_healthy = parse_class_label(top_class_raw)

    # Uncertainty triggers
    uncertainty_reasons = []"""

new_pred_block = """    # 3. Feature-Space Domain Shift & Prototype Verification
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
        )"""

assert old_pred_block in text_lf, "old_pred_block not found in core_ai.py"
text_lf = text_lf.replace(old_pred_block, new_pred_block, 1)

# 3. Add prototype_similarity to success return dict
old_success_return = """        "class_index": top_index,
        "model_readiness": model_badge,
        "model_mode": current_mode
    }"""

new_success_return = """        "class_index": top_index,
        "model_readiness": model_badge,
        "model_mode": current_mode,
        "prototype_similarity": sim
    }"""

assert old_success_return in text_lf, "old_success_return not found in core_ai.py"
text_lf = text_lf.replace(old_success_return, new_success_return, 1)

# Write back with original CRLF
core_path.write_text(text_lf.replace("\n", "\r\n"), encoding="utf-8")
print("Successfully updated core_ai.py")

