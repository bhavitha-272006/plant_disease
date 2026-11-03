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

pv_color_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "color"
prototypes = {}

print("Computing 38-class prototype centroids from verified specimens...")
for idx, class_name in enumerate(classes):
    class_folder = pv_color_dir / class_name
    if not class_folder.exists():
        continue
    img_files = list(class_folder.glob("*.JPG"))[:12]
    class_feats = []
    for f in img_files:
        try:
            arr = preprocess_image(Image.open(f).convert("RGB"))
            feat = feat_model(arr, training=False).numpy()[0]
            class_feats.append(feat)
        except Exception:
            pass
    if class_feats:
        centroid = np.mean(class_feats, axis=0)
        # Normalize
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        prototypes[class_name] = norm_centroid.tolist()

print(f"Computed prototypes for {len(prototypes)} / {len(classes)} classes.")

# Save prototypes to models/class_prototypes.json
proto_path = BASE_DIR / "models" / "class_prototypes.json"
with open(proto_path, "w", encoding="utf-8") as f:
    json.dump(prototypes, f)
print(f"Saved prototypes to {proto_path}")

# Test similarity on genuine PV leaves
pv_test_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
print("\n--- Testing Genuine PlantVillage Leaves ---")
for p in list(pv_test_dir.glob("*.JPG"))[:6]:
    arr = preprocess_image(Image.open(p).convert("RGB"))
    feat = feat_model(arr, training=False).numpy()[0]
    norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
    
    # Get model prediction
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    pred_class = classes[top_idx]
    
    proto = np.array(prototypes.get(pred_class, []))
    sim = float(np.dot(norm_feat, proto)) if len(proto) > 0 else 0.0
    print(f"PV Leaf: {p.name[:32]} -> Pred: {pred_class[:25]} | Sim: {sim:.3f} | Conf: {preds[top_idx]*100:.1f}%")

# Test similarity on PlantDoc field images
field_dir = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"
print("\n--- Testing PlantDoc Field Images ---")
for p in field_dir.glob("*.jpg"):
    arr = preprocess_image(Image.open(p).convert("RGB"))
    feat = feat_model(arr, training=False).numpy()[0]
    norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
    
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    pred_class = classes[top_idx]
    
    proto = np.array(prototypes.get(pred_class, []))
    sim = float(np.dot(norm_feat, proto)) if len(proto) > 0 else 0.0
    print(f"Field: {p.name[:32]} -> Pred: {pred_class[:25]} | Sim: {sim:.3f} | Conf: {preds[top_idx]*100:.1f}%")

