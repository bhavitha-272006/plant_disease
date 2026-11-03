"""
Fast, multi-threaded inspection and validation script for the complete PlantVillage dataset.
Scans all 38 classes, verifies PIL readability, detects corrupt files, and outputs class distribution.
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

DATA_DIR = Path("data/datasets/plantvillage/color")
OUT_FILE = Path("data/datasets/plantvillage/full_dataset_inventory.json")

def verify_single_image(filepath_str):
    try:
        # Fast header and mode check
        with Image.open(filepath_str) as img:
            img.verify()
        return filepath_str, True, os.path.getsize(filepath_str)
    except Exception as e:
        return filepath_str, False, 0

def main():
    print("=" * 60, flush=True)
    print("PLANTGUARD AI - COMPLETE PLANTVILLAGE DATASET AUDIT", flush=True)
    print("=" * 60, flush=True)
    
    start_t = time.time()
    
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist!", flush=True)
        return
        
    class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Discovered {len(class_dirs)} class directories in {DATA_DIR}.", flush=True)
    
    all_files_by_class = {}
    total_files = 0
    
    for c_dir in class_dirs:
        entries = [
            f.path for f in os.scandir(str(c_dir))
            if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        all_files_by_class[c_dir.name] = entries
        total_files += len(entries)
        
    print(f"Total image files discovered: {total_files}", flush=True)
    print("Running parallel PIL image integrity verification (16 threads)...", flush=True)
    
    all_tasks = []
    for c_name, paths in all_files_by_class.items():
        for p in paths:
            all_tasks.append((c_name, p))
            
    valid_counts = {c: 0 for c in all_files_by_class}
    corrupt_files = []
    file_sizes = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(verify_single_image, p) for _, p in all_tasks]
        for idx, (c_name, p) in enumerate(all_tasks):
            f_path, is_valid, size = futures[idx].result()
            if is_valid and size > 0:
                valid_counts[c_name] += 1
                file_sizes.append(size)
            else:
                corrupt_files.append({"class": c_name, "path": f_path})
                
    elapsed = time.time() - start_t
    total_valid = sum(valid_counts.values())
    
    print("\n--- DATASET CLASS DISTRIBUTION ---", flush=True)
    for c_name in sorted(valid_counts.keys()):
        print(f"  {c_name}: {valid_counts[c_name]} valid images", flush=True)
        
    print("\n--- SUMMARY METRICS ---", flush=True)
    print(f"Total Valid Images: {total_valid}", flush=True)
    print(f"Total Corrupt Images: {len(corrupt_files)}", flush=True)
    print(f"Min Class Size: {min(valid_counts.values())} ({[k for k,v in valid_counts.items() if v == min(valid_counts.values())][0]})", flush=True)
    print(f"Max Class Size: {max(valid_counts.values())} ({[k for k,v in valid_counts.items() if v == max(valid_counts.values())][0]})", flush=True)
    print(f"Scan Duration: {elapsed:.2f} seconds", flush=True)
    
    inventory = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_classes": len(valid_counts),
        "total_valid_images": total_valid,
        "total_corrupt_images": len(corrupt_files),
        "class_distribution": valid_counts,
        "corrupt_files": corrupt_files
    }
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"Inventory written to {OUT_FILE}", flush=True)

if __name__ == "__main__":
    main()

