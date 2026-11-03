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
feat_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

with open(BASE_DIR / "models" / "class_prototypes.json", "r") as f:
    protos = json.load(f)

pv_test_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
pv_imgs = list(pv_test_dir.glob("*.JPG"))

print("=== PV TEST SAMPLES (20 SAMPLES) ===")
pv_sims = []
for p in pv_imgs:
    arr = preprocess_image(Image.open(p).convert("RGB"))
    feat = feat_model(arr, training=False).numpy()[0]
    norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    pred_class = classes[top_idx]
    proto = np.array(protos.get(pred_class, []))
    sim = float(np.dot(norm_feat, proto)) if len(proto) > 0 else 0.0
    pv_sims.append(sim)
    print(f"PV: {p.name[:30]} -> {pred_class[:25]} | Sim: {sim:.3f}")

print(f"PV min sim: {min(pv_sims):.3f}, mean sim: {np.mean(pv_sims):.3f}, max sim: {max(pv_sims):.3f}")

print("\n=== PLANTDOC FIELD SAMPLES ===")
field_dir = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"
field_sims = []
for p in field_dir.glob("*.jpg"):
    arr = preprocess_image(Image.open(p).convert("RGB"))
    feat = feat_model(arr, training=False).numpy()[0]
    norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    pred_class = classes[top_idx]
    proto = np.array(protos.get(pred_class, []))
    sim = float(np.dot(norm_feat, proto)) if len(proto) > 0 else 0.0
    field_sims.append(sim)
    print(f"Field: {p.name[:30]} -> {pred_class[:25]} | Sim: {sim:.3f}")

print(f"Field min sim: {min(field_sims):.3f}, mean sim: {np.mean(field_sims):.3f}, max sim: {max(field_sims):.3f}")

