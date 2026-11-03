import os, sys
from pathlib import Path
BASE_DIR = Path("PlantGuardAI").resolve()
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

def evaluate_leaf_suitability(image: Image.Image, user_specimen_type: str = "Single leaf") -> dict:
    """
    Universal Specimen Suitability Gate for PlantGuard AI.
    Works across all 14 crop categories without species-specific hardcoding.
    Returns:
      - status: VALID_LEAF_SPECIMEN | NON_LEAF_ORGAN | UNSUITABLE_IMAGE | UNCERTAIN_SPECIMEN
      - is_permitted: bool (True only if VALID_LEAF_SPECIMEN)
      - reason: str
      - metrics: dict
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

    # Non-Foliar High-Chroma Organ Check (Fruit / Flower Dominance)
    # Evaluated across all inputs to detect non-foliar plant organs
    red_hsv1 = cv2.inRange(hsv, np.array([0, 75, 40]), np.array([12, 255, 255]))
    red_hsv2 = cv2.inRange(hsv, np.array([165, 75, 40]), np.array([180, 255, 255]))
    red_hsv = cv2.bitwise_or(red_hsv1, red_hsv2)
    red_lab = np.uint8((lab[:, :, 1] > 140) * 255)
    fruit_mask = cv2.bitwise_and(red_hsv, red_lab)
    fruit_pixels = cv2.countNonZero(fruit_mask)
    fruit_ratio = float(fruit_pixels / total_pixels)

    # Universal Foliar Color Segmentation (Green, Chlorotic Yellow, Necrotic Brown)
    mask_green_hsv = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([95, 255, 255]))
    mask_green_lab = np.uint8(((lab[:, :, 1] < 125) & (hsv[:, :, 1] >= 20)) * 255)
    mask_yellow = cv2.inRange(hsv, np.array([15, 30, 50]), np.array([35, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([8, 25, 20]), np.array([24, 255, 200]))

    foliage_mask = cv2.bitwise_or(mask_green_hsv, mask_green_lab)
    foliage_mask = cv2.bitwise_or(foliage_mask, mask_yellow)
    foliage_mask = cv2.bitwise_or(foliage_mask, mask_brown)

    foliage_pixels = cv2.countNonZero(foliage_mask)
    foliage_ratio = float(foliage_pixels / total_pixels)

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

# Run tests
pv_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
sample_files = list(pv_dir.glob("*.JPG"))[:6]
print("=== 1. Testing Genuine PlantVillage Leaves ===")
for p in sample_files:
    img = Image.open(p).convert("RGB")
    res = evaluate_leaf_suitability(img, user_specimen_type="Single leaf")
    print(f"File: {p.name[:35]} -> Status: {res['status']} (Permitted: {res['is_permitted']})")
    assert res["status"] == "VALID_LEAF_SPECIMEN", f"Expected VALID_LEAF_SPECIMEN, got {res['status']}"

print("\n=== 2. Testing Non-Leaf User Declarations ===")
for decl in ["Fruit", "Whole plant / canopy", "Stem / flower / root", "Not sure"]:
    res = evaluate_leaf_suitability(img, user_specimen_type=decl)
    print(f"Declared: {decl:22s} -> Status: {res['status']} (Permitted: {res['is_permitted']})")
    assert res["is_permitted"] is False, f"Expected is_permitted=False for {decl}"

print("\n=== 3. Testing Unsuitable Images (Blur, Background) ===")
blank_img = Image.new("RGB", (224, 224), (200, 200, 200))
res_blank = evaluate_leaf_suitability(blank_img)
print(f"Blank image -> Status: {res_blank['status']} (Permitted: {res_blank['is_permitted']})")
assert res_blank["status"] == "UNSUITABLE_IMAGE"

# Test synthetic fruit image (simulating tomato fruit)
fruit_img = Image.new("RGB", (224, 224), (100, 100, 100))
draw = ImageDraw.Draw(fruit_img)
# Draw some green foliage at the top
draw.rectangle([0, 0, 224, 50], fill=(40, 140, 40))
# Draw a large red tomato fruit in the center
draw.ellipse([40, 50, 184, 200], fill=(215, 25, 25))
res_fruit = evaluate_leaf_suitability(fruit_img, user_specimen_type="Single leaf")
print(f"Fruit with foliage image -> Status: {res_fruit['status']} (Permitted: {res_fruit['is_permitted']})")
assert res_fruit["status"] == "NON_LEAF_ORGAN"

print("\nALL TEST CASES PASSED SUCCESSFULLY!")
