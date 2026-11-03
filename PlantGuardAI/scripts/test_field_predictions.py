import os
os.environ['KERAS_HOME'] = 'PlantGuardAI/.keras'
import json, sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, 'PlantGuardAI')
from modules.core_ai import predict_plant_disease, load_model

p_dir = Path('PlantGuardAI/data/datasets/plantdoc/field_eval_samples')
print("Testing field samples on full model:")
for f in sorted(p_dir.glob('*.jpg')):
    img = Image.open(f)
    print(f"\nFile: {f.name} | Size: {img.size} | Mode: {img.mode}")
    res = predict_plant_disease(img, mode='full')
    print(f"  Predicted: {res['plant']} - {res['disease']}")
    print(f"  Confidence: {res['confidence']:.1f}% | Margin: {res['margin']:.1f}% | Entropy: {res['entropy']:.2f} | Uncertain: {res['is_uncertain']}")
    print(f"  Uncertainty reasons: {res['uncertainty_reasons']}")
    print(f"  Foliage info: {res['foliage_info']}")
    print(f"  Top 3:")
    for t in res['top_5'][:3]:
        print(f"    - {t['display_name']}: {t['percentage']:.1f}%")

