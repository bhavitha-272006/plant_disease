"""
Evaluates plant_model.keras on the curated PlantVillage test specimens.
Computes genuine empirical accuracy and confusion metrics.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.core_ai import predict_plant_disease, load_model
TEST_DIR = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"

def evaluate_plantvillage_test_split():
    if not TEST_DIR.exists():
        return {"error": "Test directory not found"}

    test_images = list(TEST_DIR.glob("*.JPG")) + list(TEST_DIR.glob("*.jpg"))
    if not test_images:
        return {"error": "No test images found"}

    model, classes = load_model()
    
    total = 0
    correct = 0
    results = []

    for img_path in sorted(test_images):
        # Filename convention: ClassName__original_name.JPG
        filename = img_path.name
        true_class = filename.split("__")[0]

        img = Image.open(img_path).convert("RGB")
        pred = predict_plant_disease(img)
        pred_class = pred["raw_class"]

        is_match = (true_class.lower() == pred_class.lower())
        total += 1
        if is_match:
            correct += 1

        results.append({
            "image": filename[:35] + "...",
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": round(pred["confidence"], 1),
            "match": "✅ Match" if is_match else "❌ Miss"
        })

    accuracy = round((correct / total) * 100.0, 1) if total > 0 else 0.0
    return {
        "total_test_images": total,
        "correct_predictions": correct,
        "empirical_accuracy": accuracy,
        "results": results
    }

if __name__ == "__main__":
    res = evaluate_plantvillage_test_split()
    print(f"Total Test Images: {res['total_test_images']}")
    print(f"Correct Predictions: {res['correct_predictions']}")
    print(f"Empirical Accuracy: {res['empirical_accuracy']}%")
    for r in res["results"][:10]:
        print(f"  {r['true_class']} -> {r['predicted_class']} ({r['confidence']}%) | {r['match'].replace('✅', '[PASS]').replace('❌', '[MISS]')}")
