import os, sys
from pathlib import Path

BASE_DIR = Path("PlantGuardAI").resolve()
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
sys.path.insert(0, str(BASE_DIR))

from PIL import Image
from modules.core_ai import predict_plant_disease, load_model, preprocess_image
import numpy as np
import tensorflow as tf

field_dir = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"
print("==================================================")
print("TESTING PLANTDOC FIELD SAMPLES ON MOBILENETV2 (38 CLASSES)")
print("==================================================")

model, classes = load_model(mode="full")
# Feature extractor up to global average pooling (1280-dim)
gap_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("global_average_pooling2d").output)

for img_p in field_dir.glob("*.jpg"):
    img = Image.open(img_p).convert("RGB")
    res = predict_plant_disease(img, mode="full", specimen_type="Single leaf")
    
    # Feature vector
    arr = preprocess_image(img)
    feat = gap_model(arr, training=False).numpy()[0]
    feat_norm = np.linalg.norm(feat)
    
    print(f"\nSpecimen: {img_p.name}")
    print(f"  Ground Truth (filename): {img_p.name.split('__')[0]}")
    print(f"  Model Prediction:        {res['plant']} - {res['disease']}")
    print(f"  Confidence:              {res['confidence']:.1f}%")
    print(f"  Shannon Entropy:         {res['entropy']:.2f}")
    print(f"  Margin:                  {res.get('margin', 0):.1f}%")
    t3 = [t['display_name'] + ' (' + str(round(t['percentage'], 1)) + '%)' for t in res.get('top_5', [])[:3]]
    print(f"  Top-3:                   {t3}")
    print(f"  Feature Vector L2 Norm:  {feat_norm:.2f}")
    print(f"  Uncertain Flag:          {res.get('is_uncertain')}")
    print(f"  Uncertainty Reasons:     {res.get('uncertainty_reasons')}")
