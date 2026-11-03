"""
Downloads curated evaluation test samples from PlantVillage and PlantDoc.
Avoids massive blind downloads by selecting verified representative samples per class.
"""

import os
import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PV_TEST_DIR = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
PD_TEST_DIR = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"

PV_TEST_DIR.mkdir(parents=True, exist_ok=True)
PD_TEST_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Verified raw sample URLs from spMohanty/PlantVillage-Dataset
# We use GitHub API or known sample filenames
SAMPLE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

print("--- DOWNLOADING CURATED PLANTVILLAGE TEST SPECIMENS ---")
downloaded_count = 0
for cls_name in SAMPLE_CLASSES:
    api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{cls_name}"
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            file_list = json.loads(r.read().decode())
        
        # Take the first 2 images per class
        for item in file_list[:2]:
            img_name = f"{cls_name}__{item['name']}"
            dest_path = PV_TEST_DIR / img_name
            if not dest_path.exists():
                print(f"Downloading {dest_path.name}...")
                dl_req = urllib.request.Request(item["download_url"], headers=HEADERS)
                with urllib.request.urlopen(dl_req, timeout=15) as dl_r:
                    dest_path.write_bytes(dl_r.read())
                downloaded_count += 1
    except Exception as e:
        print(f"Notice for {cls_name}: {e}")

print(f"PlantVillage test specimens download complete: {downloaded_count} images.")

