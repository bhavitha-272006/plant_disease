import sys, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ['KERAS_HOME'] = str(BASE_DIR / '.keras')

import tensorflow as tf
from PIL import Image
from modules.core_ai import parse_class_label
import numpy as np
import json

mA = tf.keras.models.load_model(str(BASE_DIR / 'models' / 'trained_plantvillage_model.keras'), compile=False)
mB = tf.keras.models.load_model(str(BASE_DIR / 'models' / 'trained_plantvillage_full_model.keras'), compile=False)

with open(BASE_DIR / 'models' / 'trained_plantvillage_classes.json') as f:
    classes = json.load(f)

for fname in ['sample_early_blight_leaf.jpg', 'sample_healthy_leaf.jpg', 'sample_severe_blight_leaf.jpg']:
    p = BASE_DIR / 'data' / 'sample_leaves' / fname
    img = Image.open(p).convert('RGB')
    arr = np.array(img.resize((224, 224)), dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    
    pA = mA.predict(batch, verbose=0)[0]
    pB = mB.predict(batch, verbose=0)[0]
    
    idxA = np.argmax(pA)
    idxB = np.argmax(pB)
    
    cA = classes[idxA]
    cB = classes[idxB]
    
    cropA, disA, _ = parse_class_label(cA)
    cropB, disB, _ = parse_class_label(cB)
    
    print(f"\nFile: {fname}")
    print(f"  Model A (Baseline 4.5k): {cropA} - {disA} ({pA[idxA]*100:.1f}%) [raw: {cA}]")
    print(f"  Model B (Full 54k):     {cropB} - {disB} ({pB[idxB]*100:.1f}%) [raw: {cB}]")
