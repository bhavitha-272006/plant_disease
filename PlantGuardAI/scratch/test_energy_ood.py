import os, sys, json
from pathlib import Path

BASE_DIR = Path("PlantGuardAI").resolve()
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import tensorflow as tf
from PIL import Image

from modules.core_ai import load_model, preprocess_image

model, classes = load_model(mode="full")
dense_layer = model.get_layer("dense")
kernel = dense_layer.kernel
bias = dense_layer.bias

feature_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

pv_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
sample_files = list(pv_dir.glob("*.JPG"))[:5]

def compute_energy_and_prediction(img, T=1.0):
    arr = preprocess_image(img)
    feat = feature_model(arr, training=False)
    logits = (tf.matmul(feat, kernel) + bias).numpy()[0]
    # Energy: -T * logsumexp(logits / T)
    max_logit = np.max(logits / T)
    energy = -T * (max_logit + np.log(np.sum(np.exp(logits / T - max_logit))))
    
    # Softmax
    exp_z = np.exp(logits - np.max(logits))
    probs = exp_z / np.sum(exp_z)
    top_idx = int(np.argmax(probs))
    conf = probs[top_idx] * 100.0
    return energy, conf, classes[top_idx]

print("--- Testing Genuine PlantVillage Leaves ---")
for p in sample_files:
    img = Image.open(p).convert("RGB")
    e, conf, cls = compute_energy_and_prediction(img)
    print(f"File: {p.name[:30]} | Energy: {e:.2f} | Conf: {conf:.1f}% | Pred: {cls}")

print("\n--- Testing Artificial / Non-Leaf Patterns ---")
# White image
white_img = Image.new("RGB", (224, 224), (255, 255, 255))
e_w, conf_w, cls_w = compute_energy_and_prediction(white_img)
print(f"White Image: Energy: {e_w:.2f} | Conf: {conf_w:.1f}% | Pred: {cls_w}")

# Black image
black_img = Image.new("RGB", (224, 224), (0, 0, 0))
e_b, conf_b, cls_b = compute_energy_and_prediction(black_img)
print(f"Black Image: Energy: {e_b:.2f} | Conf: {conf_b:.1f}% | Pred: {cls_b}")

# Pure red fruit-like circle
from PIL import ImageDraw
red_fruit = Image.new("RGB", (224, 224), (120, 120, 120))
draw = ImageDraw.Draw(red_fruit)
draw.ellipse([40, 40, 184, 184], fill=(220, 20, 20))
e_rf, conf_rf, cls_rf = compute_energy_and_prediction(red_fruit)
print(f"Red Fruit Pattern: Energy: {e_rf:.2f} | Conf: {conf_rf:.1f}% | Pred: {cls_rf}")

