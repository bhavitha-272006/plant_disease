"""
End-to-end domain shift validation.
Tests the prototype cosine similarity guard against:
  - PlantVillage genuine leaves (should PASS)
  - PlantDoc field images (should be WITHHELD when sim < 0.20)
"""
import os, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set CWD to PlantGuardAI project root
_HERE = Path(__file__).resolve().parent.parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from PIL import Image
from modules.core_ai import predict_plant_disease

print("=" * 60)
print("DOMAIN SHIFT GUARD VALIDATION")
print("=" * 60)

# 1. Genuine PlantVillage leaf specimens → should be PERMITTED
pv_test_dir = Path("data/datasets/plantvillage/test_samples")
pv_imgs = list(pv_test_dir.glob("*.JPG"))
print(f"\n--- Genuine PlantVillage Leaves ({len(pv_imgs)} specimens) ---")
passed_pv = 0
for p in pv_imgs:
    img = Image.open(p).convert("RGB")
    pred = predict_plant_disease(img, specimen_type="Single leaf", mode="full")
    blocked = pred.get("is_blocked", False)
    sim = pred.get("prototype_similarity", -1)
    status = "BLOCKED (OOD)" if blocked else f"PASSED ({pred['plant']} - {pred['disease'][:20]}, {pred['confidence']:.1f}%)"
    print(f"  PV: {p.name[:32]} -> {status} | Sim: {sim:.3f}")
    if not blocked:
        passed_pv += 1
print(f"PV Summary: {passed_pv}/{len(pv_imgs)} PASSED (should be all)")

# 2. PlantDoc field images → should be WITHHELD (very low prototype sim)
field_dir = Path("data/datasets/plantdoc/field_eval_samples")
field_imgs = list(field_dir.glob("*.jpg"))
print(f"\n--- PlantDoc Field Images ({len(field_imgs)} specimens) ---")
blocked_field = 0
for p in field_imgs:
    img = Image.open(p).convert("RGB")
    pred = predict_plant_disease(img, specimen_type="Single leaf", mode="full")
    blocked = pred.get("is_blocked", False)
    sim = pred.get("prototype_similarity", -1)
    if blocked:
        status = f"BLOCKED (OOD sim={sim:.3f})"
        blocked_field += 1
    else:
        status = f"PASSED ({pred['plant']} - {pred['disease'][:20]}, {pred['confidence']:.1f}%, sim={sim:.3f})"
    print(f"  Field: {p.name[:32]} -> {status}")
print(f"Field Summary: {blocked_field}/{len(field_imgs)} BLOCKED (prevents overconfident wrong prediction)")

print("\n" + "=" * 60)
print("DOMAIN SHIFT GUARD VALIDATION COMPLETE")
print("=" * 60)
