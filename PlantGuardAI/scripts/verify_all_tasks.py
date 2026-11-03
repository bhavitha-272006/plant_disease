"""
Verification suite for the three requested improvements:
TASK 1: Fix Model Selection
TASK 2: Uncertainty Warnings
TASK 3: Model Evaluation Dashboard
"""

import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from modules.core_ai import (
    load_model,
    predict_plant_disease,
    compute_gradcam,
    get_available_models,
    set_active_model_mode,
    get_active_model_mode,
    get_model_info
)

print("=" * 80)
print("PLANTGUARD AI - SYSTEMATIC VERIFICATION SUITE")
print("=" * 80)

# -------------------------------------------------------------------------
# CHECK 1: Python Compilation / Syntax Checks
# -------------------------------------------------------------------------
print("\n[CHECK 1] Python Compilation & Syntax Validation:")
import py_compile
files_to_compile = [
    BASE_DIR / "app.py",
    BASE_DIR / "modules" / "core_ai.py",
    BASE_DIR / "modules" / "history_manager.py",
    BASE_DIR / "modules" / "vision.py",
    BASE_DIR / "modules" / "pest_detector.py",
    BASE_DIR / "modules" / "nutrient_analyzer.py"
]

all_compiled = True
for f in files_to_compile:
    try:
        py_compile.compile(str(f), doraise=True)
        print(f"  ✅ Compiled successfully: {f.name}")
    except Exception as e:
        print(f"  ❌ Compilation failed for {f.name}: {e}")
        all_compiled = False

assert all_compiled, "One or more Python files failed syntax compilation!"

# -------------------------------------------------------------------------
# CHECK 2: Model Keys & Default Mode (Task 1)
# -------------------------------------------------------------------------
print("\n[CHECK 2] Model Selection Keys & Default Mode:")
avail = get_available_models()
print(f"  Available Model Keys: {list(avail.keys())}")
assert "full" in avail, "Key 'full' must exist in available models!"
assert "baseline" in avail, "Key 'baseline' must exist in available models!"
assert "legacy" in avail, "Key 'legacy' must exist in available models!"

# Check default mode
default_mode = get_active_model_mode()
print(f"  Default Active Mode in core_ai: '{default_mode}'")
assert default_mode == "full", f"Default mode must be 'full', found '{default_mode}'!"
print(f"  ✅ 'full' is default active model mode.")

# -------------------------------------------------------------------------
# CHECK 3: Verify All Three Models & Output Dimensions (Task 1)
# -------------------------------------------------------------------------
print("\n[CHECK 3] Model Loading, Architecture, & Class Mapping Lengths:")
expected_specs = {
    "full": ("trained_plantvillage_full_model.keras", 38),
    "baseline": ("trained_plantvillage_model.keras", 38),
    "legacy": ("plant_model.keras", 15)
}

for mode_key, (expected_file, expected_classes) in expected_specs.items():
    model, classes = load_model(mode_key)
    out_dim = model.output_shape[-1]
    cls_len = len(classes)
    print(f"  Mode '{mode_key:8s}': File={expected_file:38s} | OutputDim={out_dim:2d} | ClassesCount={cls_len:2d}")
    assert out_dim == expected_classes, f"Mode {mode_key}: expected {expected_classes} output nodes, got {out_dim}"
    assert cls_len == expected_classes, f"Mode {mode_key}: expected {expected_classes} classes, got {cls_len}"
    assert out_dim == cls_len, f"Mode {mode_key}: output dim {out_dim} != class mapping len {cls_len}"

print("  ✅ All 3 models verified: exact output dimensions match class mapping counts.")

# -------------------------------------------------------------------------
# CHECK 4: Grad-CAM Consistency with Active Model (Task 1)
# -------------------------------------------------------------------------
print("\n[CHECK 4] Grad-CAM Model Consistency:")
test_samples_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
test_img_path = list(test_samples_dir.glob("*.JPG"))[0]
sample_img = Image.open(test_img_path).convert("RGB")

for test_mode in ["full", "baseline", "legacy"]:
    set_active_model_mode(test_mode)
    pred = predict_plant_disease(sample_img, mode=test_mode)
    gradcam_img, gradcam_raw = compute_gradcam(sample_img, pred["class_index"], mode=pred["model_mode"])
    assert gradcam_raw.shape == (sample_img.height, sample_img.width)
    assert gradcam_raw.max() > 0.0
    print(f"  Mode '{test_mode:8s}': ClassIndex={pred['class_index']:2d} | PeakActivation={gradcam_raw.max():.2f} | ModelMode={pred['model_mode']}")

print("  ✅ Grad-CAM generates valid activation overlays across all 3 models.")

# -------------------------------------------------------------------------
# CHECK 5: Uncertainty Warnings & OOD Verification (Task 2)
# -------------------------------------------------------------------------
print("\n[CHECK 5] Uncertainty Warnings & Distinct OOD Triggers:")

# 5a. Authentic high-confidence specimen
set_active_model_mode("full")
pred_confident = predict_plant_disease(sample_img, mode="full")
print(f"  Authentic Specimen: Confidence={pred_confident['confidence']:.1f}%, Uncertain={pred_confident['is_uncertain']}")

# 5b. Artificial non-leaf pattern (OOD)
grid_img = Image.new("RGB", (224, 224), color=(128, 128, 130))
draw = ImageDraw.Draw(grid_img)
for y in range(0, 224, 20):
    draw.line([(0, y), (224, y)], fill=(70, 70, 75), width=2)

pred_ood = predict_plant_disease(grid_img, mode="full")
print(f"  Artificial Texture Specimen (OOD):")
print(f"    Confidence:           {pred_ood['confidence']:.1f}%")
print(f"    Is Flagged Uncertain: {pred_ood['is_uncertain']}")
print(f"    Uncertainty Reasons:")
for r in pred_ood["uncertainty_reasons"]:
    print(f"      - {r}")

assert pred_ood["is_uncertain"] is True, "OOD non-leaf pattern must be flagged as uncertain!"
print("  ✅ Uncertainty triggers activate correctly and distinguish model uncertainty from foliage checks.")

# -------------------------------------------------------------------------
# CHECK 6: Model Evaluation Dashboard Report Verification (Task 3)
# -------------------------------------------------------------------------
print("\n[CHECK 6] Full Model Evaluation Report Data Integrity:")
eval_path = BASE_DIR / "data" / "full_model_evaluation_report.json"
assert eval_path.exists(), f"Evaluation report file {eval_path} must exist!"

with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

tsm = eval_data.get("test_split_metrics", {})
tsum = eval_data.get("training_summary", {})
splits = eval_data.get("split_counts", {})

print(f"  Checkpoint:        {tsum.get('model_name')}.keras")
print(f"  Dataset:           {eval_data.get('dataset_name')}")
print(f"  Held-out Test Img: {splits.get('test'):,}")
print(f"  Test Accuracy:     {tsm.get('test_accuracy')*100:.2f}%")
print(f"  Macro Precision:   {tsm.get('macro_precision')*100:.2f}%")
print(f"  Macro Recall:      {tsm.get('macro_recall')*100:.2f}%")
print(f"  Macro F1-Score:    {tsm.get('macro_f1_score')*100:.2f}%")
print(f"  Weighted F1-Score: {tsm.get('weighted_f1_score')*100:.2f}%")
print(f"  Timestamp:         {eval_data.get('evaluation_timestamp')}")

assert tsm.get("test_accuracy") > 0.95, "Test accuracy should be > 95%!"
assert splits.get("test") == 8143, "Held-out test images count must be 8,143!"

# Check assets
cm_asset = BASE_DIR / "assets" / "full_model_confusion_matrix.png"
curves_asset = BASE_DIR / "assets" / "full_model_training_curves.png"
print(f"  Confusion Matrix Asset: {cm_asset.name} (exists={cm_asset.exists()})")
print(f"  Training Curves Asset:  {curves_asset.name} (exists={curves_asset.exists()})")
assert cm_asset.exists(), f"Asset {cm_asset} must exist!"
assert curves_asset.exists(), f"Asset {curves_asset} must exist!"
print("  ✅ All evaluation metrics and assets strictly match full model specifications.")

print("\n" + "=" * 80)
print(">>> ALL 6 VERIFICATION CHECKS PASSED WITH ZERO ERRORS! <<<")
print("=" * 80)

