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

# Find tomato_spoil
found = list(Path(".").glob("**/*tomato_spoil*"))
print("Found images matching tomato_spoil:", found)
for img_path in found:
    if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        arr = preprocess_image(Image.open(img_path).convert("RGB"))
        feat = feat_model(arr, training=False).numpy()[0]
        norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
        preds = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        pred_class = classes[top_idx]
        proto = np.array(protos.get(pred_class, []))
        sim = float(np.dot(norm_feat, proto)) if len(proto) > 0 else 0.0
        print(f"{img_path.name} -> Pred: {pred_class} | Prototype Sim: {sim:.3f} | Softmax Conf: {preds[top_idx]*100:.1f}%")

