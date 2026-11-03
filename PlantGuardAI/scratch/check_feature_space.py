import os, sys
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

pv_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
field_dir = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"

pv_feats = []
for p in list(pv_dir.glob("*.JPG"))[:15]:
    arr = preprocess_image(Image.open(p).convert("RGB"))
    f = feat_model(arr, training=False).numpy()[0]
    pv_feats.append(f)

pv_feats = np.array(pv_feats)
pv_mean = np.mean(pv_feats, axis=0)

print(f"PV Feature Norms: mean={np.mean(np.linalg.norm(pv_feats, axis=1)):.2f}, std={np.std(np.linalg.norm(pv_feats, axis=1)):.2f}")

print("\n--- Cosine Similarities of PV Leaves to PV Centroid ---")
for p in list(pv_dir.glob("*.JPG"))[:5]:
    arr = preprocess_image(Image.open(p).convert("RGB"))
    f = feat_model(arr, training=False).numpy()[0]
    cos_sim = np.dot(f, pv_mean) / (np.linalg.norm(f) * np.linalg.norm(pv_mean))
    print(f"PV Leaf: {p.name[:30]} | Cosine Sim: {cos_sim:.3f}")

print("\n--- Cosine Similarities of Real-World Field Images to PV Centroid ---")
for p in field_dir.glob("*.jpg"):
    arr = preprocess_image(Image.open(p).convert("RGB"))
    f = feat_model(arr, training=False).numpy()[0]
    cos_sim = np.dot(f, pv_mean) / (np.linalg.norm(f) * np.linalg.norm(pv_mean))
    print(f"Field Img: {p.name[:30]} | Cosine Sim: {cos_sim:.3f}")

