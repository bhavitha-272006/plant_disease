"""
Diagnostic script to inspect the MobileNetV2 plant disease detection model.
"""

from pathlib import Path
import json
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "plant_model.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

print("=" * 60)
print("PLANTGUARD AI - MODEL INTEGRITY & DIAGNOSTIC INSPECTOR")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
    exit(1)

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

print(f"\nModel File: {MODEL_PATH.name}")
print(f"Model Input Shape:  {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")

# Inspect first layer for Rescaling
first_layer = model.layers[0]
print(f"\nFirst Layer: {first_layer.name} ({first_layer.__class__.__name__})")
if hasattr(first_layer, "scale") and hasattr(first_layer, "offset"):
    print(f"  -> Rescaling Scale:  {first_layer.scale} (1/127.5 = {1/127.5})")
    print(f"  -> Rescaling Offset: {first_layer.offset}")
    print("  -> CRITICAL: Input images MUST be [0, 255] float32.")

# Output layer
output_layer = model.layers[-1]
print(f"\nFinal Layer: {output_layer.name} ({output_layer.__class__.__name__})")
print(f"  -> Output Units: {getattr(output_layer, 'units', 'N/A')}")
print(f"  -> Activation:   {getattr(output_layer, 'activation', 'N/A').__name__ if hasattr(getattr(output_layer, 'activation', None), '__name__') else getattr(output_layer, 'activation', 'N/A')}")

# Load classes
if CLASS_PATH.exists():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    print(f"\nLoaded {len(classes)} classes from {CLASS_PATH.name}:")
    for i, c in enumerate(classes):
        print(f"  [{i:02d}] {c}")
else:
    print(f"[WARNING] Class file missing at {CLASS_PATH}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
