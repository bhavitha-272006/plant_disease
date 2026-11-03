"""
Downloads curated evaluation test samples from PlantDoc field dataset.
"""

import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PD_TEST_DIR = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"
PD_TEST_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

PLANTDOC_SAMPLE_FOLDERS = [
    "Apple%20Scab%20Leaf",
    "Potato%20leaf%20early%20blight",
    "Potato%20leaf%20late%20blight",
    "Tomato%20Early%20blight%20leaf",
    "Tomato%20leaf%20late%20blight"
]

print("--- DOWNLOADING CURATED PLANTDOC FIELD SPECIMENS ---")
count = 0
for folder in PLANTDOC_SAMPLE_FOLDERS:
    clean_folder = folder.replace("%20", " ")
    api_url = f"https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents/test/{folder}"
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            file_list = json.loads(r.read().decode())
        
        for item in file_list[:2]:
            img_name = f"Field_{clean_folder}__{item['name']}"
            dest_path = PD_TEST_DIR / img_name
            if not dest_path.exists():
                print(f"Downloading {dest_path.name}...")
                dl_req = urllib.request.Request(item["download_url"], headers=HEADERS)
                with urllib.request.urlopen(dl_req, timeout=15) as dl_r:
                    dest_path.write_bytes(dl_r.read())
                count += 1
    except Exception as e:
        print(f"Notice for {clean_folder}: {e}")

print(f"PlantDoc field specimens download complete: {count} images.")

