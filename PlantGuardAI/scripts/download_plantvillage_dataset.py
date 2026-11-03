"""
PlantVillage Dataset Downloader, Extractor, and Inventory Validator.

Downloads the official PlantVillage dataset (spMohanty/PlantVillage),
extracts the 'color' images preserving the original class structure,
validates image integrity, counts images per class, and writes an inventory report.
"""

import os
import sys
import json
import time
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image

DATA_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
BASE_DIR = Path(__file__).resolve().parent.parent
PV_DIR = BASE_DIR / "data" / "datasets" / "plantvillage"
ZIP_PATH = PV_DIR / "data.zip"
TARGET_DIR = PV_DIR / "color"
REPORT_PATH = PV_DIR / "inventory_report.json"

def download_with_resume(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".zip.part")
    
    initial_bytes = 0
    if temp_path.exists():
        initial_bytes = temp_path.stat().st_size
        print(f"[RESUME] Found partial download of {initial_bytes / (1024*1024):.1f} MB.")
    
    headers = {"User-Agent": "PlantGuardAI-Downloader/1.0"}
    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"
        
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = response.headers.get("Content-Length")
            status = response.getcode()
            
            if status == 206: # Partial Content
                total_bytes = initial_bytes + int(total_size)
            elif status == 200:
                initial_bytes = 0
                total_bytes = int(total_size) if total_size else 0
            else:
                total_bytes = int(total_size) if total_size else 0
                
            mode = "ab" if initial_bytes > 0 else "wb"
            downloaded = initial_bytes
            last_log_time = time.time()
            start_time = time.time()
            
            print(f"[DOWNLOAD] Target size: {total_bytes / (1024*1024):.1f} MB. Starting download...")
            
            with open(temp_path, mode) as f:
                while True:
                    chunk = response.read(1024 * 512) # 512 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    cur_time = time.time()
                    if cur_time - last_log_time >= 5.0 or downloaded >= total_bytes:
                        elapsed = cur_time - start_time
                        speed_mb = ((downloaded - initial_bytes) / (1024*1024)) / max(elapsed, 0.001)
                        pct = (downloaded / total_bytes * 100) if total_bytes else 0
                        print(f"  Progress: {downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.2f} MB/s", flush=True)
                        last_log_time = cur_time
                        
        # Rename .part to final
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"[DOWNLOAD COMPLETE] Saved to {dest_path} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
            return True
            
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {e}")
        return False

def extract_color_subset(zip_path: Path, output_dir: Path):
    print(f"[EXTRACT] Inspecting zip file: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names = zf.namelist()
        # Filter for raw/color/
        color_files = [n for n in all_names if n.startswith("raw/color/") and not n.endswith("/")]
        print(f"[EXTRACT] Found {len(color_files)} color images inside zip. Extracting...")
        
        extracted_count = 0
        last_log = time.time()
        for idx, file_name in enumerate(color_files):
            # Target path: TARGET_DIR / <class_name> / <filename>
            parts = file_name.split("/")
            if len(parts) >= 4:
                class_name = parts[2]
                img_name = parts[3]
                dest_file = output_dir / class_name / img_name
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                with zf.open(file_name) as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())
                extracted_count += 1
                
            cur_time = time.time()
            if cur_time - last_log >= 5.0 or idx == len(color_files) - 1:
                print(f"  Extracted: {extracted_count}/{len(color_files)} ({(extracted_count/len(color_files)*100):.1f}%)", flush=True)
                last_log = cur_time
                
    print(f"[EXTRACT COMPLETE] Total color images extracted: {extracted_count}")
    return extracted_count

def validate_and_catalog(color_dir: Path, report_path: Path):
    print(f"[VALIDATE] Validating extracted images in {color_dir}...")
    class_dirs = [d for d in color_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda d: d.name)
    
    total_found = 0
    total_valid = 0
    corrupted_images = []
    class_counts = {}
    
    for c_dir in class_dirs:
        class_name = c_dir.name
        img_files = list(c_dir.glob("*.*"))
        valid_count = 0
        
        for img_path in img_files:
            total_found += 1
            try:
                with Image.open(img_path) as im:
                    im.verify()
                valid_count += 1
            except Exception as e:
                print(f"  [CORRUPTED] {img_path.name} in {class_name}: {e}")
                corrupted_images.append(str(img_path))
                try:
                    img_path.unlink() # remove corrupted file
                except Exception:
                    pass
                    
        class_counts[class_name] = valid_count
        total_valid += valid_count
        
    print(f"[INVENTORY SUMMARY] Found {len(class_counts)} classes, {total_valid} valid images (Corrupted removed: {len(corrupted_images)})")
    
    report = {
        "dataset_name": "PlantVillage (Official raw/color)",
        "source": DATA_URL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_classes": len(class_counts),
        "total_valid_images": total_valid,
        "total_corrupted_removed": len(corrupted_images),
        "class_counts": class_counts,
        "classes_list": list(class_counts.keys())
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"[SAVED REPORT] Written to {report_path}")
    return report

def main():
    start_all = time.time()
    
    # 1. Download
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 2_000_000_000:
        success = download_with_resume(DATA_URL, ZIP_PATH)
        if not success:
            print("[FATAL] Download failed.")
            sys.exit(1)
    else:
        print(f"[FOUND EXISTING] Zip archive already exists: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)")
        
    # 2. Extract
    extracted_count = extract_color_subset(ZIP_PATH, TARGET_DIR)
    
    # 3. Clean up zip to conserve disk space
    if ZIP_PATH.exists():
        print(f"[CLEANUP] Removing zip archive {ZIP_PATH} to save disk space...")
        try:
            ZIP_PATH.unlink()
            print("[CLEANUP] Zip archive removed successfully.")
        except Exception as e:
            print(f"[CLEANUP WARNING] Could not remove zip: {e}")
            
    # 4. Validate & Catalog
    report = validate_and_catalog(TARGET_DIR, REPORT_PATH)
    
    elapsed = time.time() - start_all
    print(f"[ALL TASKS COMPLETE] Finished in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()

