"""
PlantVillage Dataset Fetch, Extraction, and Inventory Pipeline.
Uses curl.exe for maximum download speed and robustness with resume capability,
then extracts the 'raw/color' folder, removes the archive, and generates
the inventory report.
"""

import os
import sys
import json
import time
import zipfile
import subprocess
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PV_DIR = BASE_DIR / "data" / "datasets" / "plantvillage"
ZIP_PATH = PV_DIR / "data.zip"
TARGET_DIR = PV_DIR / "color"
REPORT_PATH = PV_DIR / "inventory_report.json"
URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"

def download():
    print(f"[STAGE 1/4] Starting / Resuming curl download of PlantVillage dataset...")
    PV_DIR.mkdir(parents=True, exist_ok=True)
    
    # If partial file exists from previous python attempt, remove it or rename to data.zip
    part_file = PV_DIR / "data.zip.part"
    if part_file.exists() and not ZIP_PATH.exists():
        part_file.rename(ZIP_PATH)
        
    cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--retry", "20",
        "--retry-delay", "3",
        "--retry-max-time", "1800",
        "-o", str(ZIP_PATH),
        URL
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] curl exited with code {res.returncode}")
        return False
        
    elapsed = time.time() - t0
    size_mb = ZIP_PATH.stat().st_size / (1024*1024)
    print(f"[DOWNLOAD COMPLETED] {size_mb:.2f} MB downloaded in {elapsed:.1f}s")
    return True

def extract():
    print(f"[STAGE 2/4] Extracting color images from {ZIP_PATH} to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        color_files = [n for n in zf.namelist() if n.startswith("raw/color/") and not n.endswith("/")]
        print(f"  Found {len(color_files)} color files in zip archive.")
        
        count = 0
        last_log = time.time()
        for idx, f in enumerate(color_files):
            parts = f.split("/")
            if len(parts) >= 4:
                class_name = parts[2]
                img_name = parts[3]
                dest_file = TARGET_DIR / class_name / img_name
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                with zf.open(f) as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())
                count += 1
                
            cur_time = time.time()
            if cur_time - last_log >= 10.0 or idx == len(color_files) - 1:
                print(f"  Extracted {count}/{len(color_files)} ({(count/len(color_files)*100):.1f}%)", flush=True)
                last_log = cur_time
                
    elapsed = time.time() - t0
    print(f"[EXTRACTION COMPLETED] {count} images extracted in {elapsed:.1f}s.")
    return count

def cleanup():
    print(f"[STAGE 3/4] Cleaning up archive to conserve disk space...")
    if ZIP_PATH.exists():
        try:
            ZIP_PATH.unlink()
            print("  Archive data.zip removed.")
        except Exception as e:
            print(f"  Cleanup note: {e}")

def validate_and_catalog():
    print(f"[STAGE 4/4] Validating and cataloging classes in {TARGET_DIR}...")
    class_dirs = [d for d in TARGET_DIR.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda d: d.name)
    
    class_counts = {}
    total_valid = 0
    corrupted = []
    
    t0 = time.time()
    for c_dir in class_dirs:
        class_name = c_dir.name
        img_files = list(c_dir.glob("*.*"))
        valid_in_class = 0
        for img_path in img_files:
            try:
                with Image.open(img_path) as im:
                    im.verify()
                valid_in_class += 1
            except Exception as e:
                corrupted.append(str(img_path))
                try:
                    img_path.unlink()
                except Exception:
                    pass
        class_counts[class_name] = valid_in_class
        total_valid += valid_in_class
        
    elapsed = time.time() - t0
    print(f"[VALIDATION COMPLETED] {len(class_counts)} classes, {total_valid} valid images in {elapsed:.1f}s.")
    
    report = {
        "dataset_name": "PlantVillage (Official raw/color)",
        "source": URL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_classes": len(class_counts),
        "total_valid_images": total_valid,
        "total_corrupted_removed": len(corrupted),
        "class_counts": class_counts,
        "classes_list": list(class_counts.keys())
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"[REPORT SAVED] Report written to {REPORT_PATH}")
    print("[DATASET READY] Full PlantVillage dataset is prepared and verified.")
    return report

def main():
    if not TARGET_DIR.exists() or len(list(TARGET_DIR.glob("*/*.*"))) < 1000:
        if not download():
            sys.exit(1)
        extract()
        cleanup()
    else:
        print(f"[INFO] Color dataset already extracted at {TARGET_DIR}.")
        
    validate_and_catalog()

if __name__ == "__main__":
    main()

