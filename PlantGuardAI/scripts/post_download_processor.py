"""
Post-Download Automated Processing & End-to-End Pipeline:
1. Extract raw/color subset from data.zip.
2. Remove zip archive.
3. Validate and catalog dataset inventory (inventory_report.json).
4. Run MobileNetV2 transfer learning training (train_disease_classifier.py).
5. Run full evaluation suite (evaluate_disease_classifier.py).
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
PYTHON_EXE = BASE_DIR / "venv" / "Scripts" / "python.exe"

def extract_color():
    print("=" * 70)
    print("STEP 1: EXTRACTING PLANTVILLAGE COLOR IMAGES")
    print("=" * 70)
    
    if not ZIP_PATH.exists():
        print(f"[ERROR] Zip file not found at {ZIP_PATH}")
        sys.exit(1)
        
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        color_files = [n for n in zf.namelist() if n.startswith("raw/color/") and not n.endswith("/")]
        total_files = len(color_files)
        print(f"[EXTRACT] Found {total_files} color image files inside archive.")
        
        extracted = 0
        last_log = time.time()
        for idx, f in enumerate(color_files):
            parts = f.split("/")
            if len(parts) >= 4:
                c_name = parts[2]
                img_name = parts[3]
                dest_f = TARGET_DIR / c_name / img_name
                dest_f.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(f) as src, open(dest_f, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                
            cur = time.time()
            if cur - last_log >= 10.0 or idx == total_files - 1:
                pct = (extracted / total_files) * 100
                print(f"  Extracted: {extracted}/{total_files} ({pct:.1f}%)", flush=True)
                last_log = cur
                
    elapsed = time.time() - t0
    print(f"[EXTRACTION COMPLETED] {extracted} images extracted in {elapsed:.1f}s.")
    
    # Conserve disk space
    print("[CLEANUP] Removing zip archive to reclaim 2.1 GB disk space...")
    try:
        ZIP_PATH.unlink()
        print("  Archive removed.")
    except Exception as e:
        print(f"  Cleanup note: {e}")
        
    return extracted

def validate_and_catalog():
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATING IMAGES & GENERATING DATASET INVENTORY")
    print("=" * 70)
    
    class_dirs = sorted([d for d in TARGET_DIR.iterdir() if d.is_dir()])
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
    print(f"[VALIDATION] {len(class_counts)} classes, {total_valid} valid images (Corrupted: {len(corrupted)}) in {elapsed:.1f}s.")
    
    report = {
        "dataset_name": "PlantVillage (Official raw/color)",
        "source": "https://huggingface.co/datasets/mohanty/PlantVillage",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_classes": len(class_counts),
        "total_valid_images": total_valid,
        "total_corrupted_removed": len(corrupted),
        "class_counts": class_counts,
        "classes_list": list(class_counts.keys())
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"[REPORT SAVED] Written to {REPORT_PATH}")
    return report

def run_training_pipeline():
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING MOBILENETV2 TRANSFER LEARNING TRAINING")
    print("=" * 70)
    
    # Train MobileNetV2 with transfer learning on the COMPLETE PlantVillage dataset
    # Uses stratified 70% train, 15% val, 15% test splits across all 54,000+ images!
    train_script = BASE_DIR / "training" / "train_disease_classifier.py"
    cmd = [
        str(PYTHON_EXE),
        str(train_script),
        "--epochs", "3",
        "--batch-size", "64"
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[TRAINING FAILED] Code {res.returncode}")
        sys.exit(res.returncode)
        
    print(f"[TRAINING FINISHED] Completed in {time.time() - t0:.1f}s.")

def run_evaluation_pipeline():
    print("\n" + "=" * 70)
    print("STEP 4: RUNNING RIGOROUS EVALUATION & FIELD BENCHMARK")
    print("=" * 70)
    
    eval_script = BASE_DIR / "scripts" / "evaluate_disease_classifier.py"
    cmd = [str(PYTHON_EXE), str(eval_script)]
    
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[EVALUATION FAILED] Code {res.returncode}")
        sys.exit(res.returncode)
        
    print(f"[EVALUATION FINISHED] Completed in {time.time() - t0:.1f}s.")

def main():
    print("Starting automated post-download pipeline...")
    extract_color()
    validate_and_catalog()
    run_training_pipeline()
    run_evaluation_pipeline()
    print("\n" + "#" * 70)
    print("# ALL PIPELINE STEPS SUCCESSFULLY COMPLETED!")
    print("#" * 70)

if __name__ == "__main__":
    main()

