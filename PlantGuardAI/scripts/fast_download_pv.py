"""
Fast Multi-Threaded Downloader & Extractor for PlantVillage Dataset.
Downloads the official Hugging Face data.zip using parallel Range requests,
extracts raw/color, removes the zip archive, verifies image integrity,
and generates the complete inventory report.
"""

import os
import sys
import json
import time
import zipfile
import requests
import concurrent.futures
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PV_DIR = BASE_DIR / "data" / "datasets" / "plantvillage"
ZIP_PATH = PV_DIR / "data.zip"
TARGET_DIR = PV_DIR / "color"
REPORT_PATH = PV_DIR / "inventory_report.json"
SOURCE_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
NUM_WORKERS = 6

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def get_content_info():
    print("[1/5] Resolving download URL and file size...")
    r = requests.head(SOURCE_URL, headers=HEADERS, allow_redirects=True, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to resolve URL: HTTP {r.status_code}")
    total_size = int(r.headers.get("content-length", 0))
    final_url = r.url
    print(f"  Target size: {total_size / (1024*1024):.2f} MB ({total_size} bytes)")
    return final_url, total_size

def download_part(url, start_byte, end_byte, part_file: Path, part_id: int):
    # Check if already downloaded
    if part_file.exists():
        existing_size = part_file.stat().st_size
        expected_size = end_byte - start_byte + 1
        if existing_size == expected_size:
            print(f"  [Worker {part_id}] Already complete ({existing_size/(1024*1024):.1f} MB)")
            return True
        else:
            part_file.unlink()

    h = dict(HEADERS)
    h["Range"] = f"bytes={start_byte}-{end_byte}"
    
    expected_bytes = end_byte - start_byte + 1
    t0 = time.time()
    
    with requests.get(url, headers=h, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        downloaded = 0
        last_log = time.time()
        with open(part_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    cur_t = time.time()
                    if cur_t - last_log >= 10.0:
                        pct = (downloaded / expected_bytes) * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"    Worker {part_id}: {mb:.1f} MB ({pct:.1f}%)", flush=True)
                        last_log = cur_t
                        
    elapsed = max(time.time() - t0, 0.001)
    print(f"  [Worker {part_id} DONE] {downloaded/(1024*1024):.1f} MB in {elapsed:.1f}s ({(downloaded/(1024*1024))/elapsed:.2f} MB/s)")
    return True

def assemble_parts(part_files, final_path: Path):
    print(f"[3/5] Assembling {len(part_files)} parts into {final_path.name}...")
    t0 = time.time()
    with open(final_path, "wb") as outfile:
        for idx, pf in enumerate(part_files):
            print(f"  Writing part {idx + 1}/{len(part_files)}...")
            with open(pf, "rb") as infile:
                while True:
                    buf = infile.read(1024 * 1024 * 8) # 8 MB chunks
                    if not buf:
                        break
                    outfile.write(buf)
            pf.unlink() # remove part file after assembling
    print(f"  Assembled {final_path.stat().st_size / (1024*1024):.2f} MB in {time.time() - t0:.1f}s")

def extract_color():
    print(f"[4/5] Extracting raw/color files to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        color_files = [n for n in zf.namelist() if n.startswith("raw/color/") and not n.endswith("/")]
        total_files = len(color_files)
        print(f"  Found {total_files} color files in zip archive.")
        
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
                print(f"  Extracted {extracted}/{total_files} ({(extracted/total_files*100):.1f}%)", flush=True)
                last_log = cur
                
    print(f"  Extraction complete: {extracted} images in {time.time() - t0:.1f}s")
    
    # Clean up zip
    if ZIP_PATH.exists():
        try:
            ZIP_PATH.unlink()
            print("  Removed zip archive to save disk space.")
        except Exception as e:
            print(f"  Warning: {e}")

def validate_and_catalog():
    print(f"[5/5] Validating and cataloging classes in {TARGET_DIR}...")
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
        
    report = {
        "dataset_name": "PlantVillage (Official raw/color)",
        "source": SOURCE_URL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_classes": len(class_counts),
        "total_valid_images": total_valid,
        "total_corrupted_removed": len(corrupted),
        "class_counts": class_counts,
        "classes_list": list(class_counts.keys())
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"[REPORT SAVED] {len(class_counts)} classes, {total_valid} images saved to {REPORT_PATH}")
    return report

def main():
    PV_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if color folder already has > 10,000 images
    if TARGET_DIR.exists():
        existing_imgs = list(TARGET_DIR.glob("*/*.*"))
        if len(existing_imgs) > 10000:
            print(f"[EXISTS] Color dataset already extracted: {len(existing_imgs)} images found.")
            validate_and_catalog()
            return

    final_url, total_size = get_content_info()
    
    part_size = total_size // NUM_WORKERS
    part_specs = []
    part_files = []
    
    for i in range(NUM_WORKERS):
        start = i * part_size
        end = total_size - 1 if i == NUM_WORKERS - 1 else (i + 1) * part_size - 1
        pf = PV_DIR / f"data.zip.part{i}"
        part_specs.append((start, end, pf, i))
        part_files.append(pf)
        
    print(f"[2/5] Launching {NUM_WORKERS} parallel download workers...")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(download_part, final_url, s, e, f, i)
            for s, e, f, i in part_specs
        ]
        concurrent.futures.wait(futures)
        
    total_dl_time = time.time() - t0
    print(f"[DOWNLOAD COMPLETED] All {NUM_WORKERS} parts downloaded in {total_dl_time:.1f}s ({(total_size/(1024*1024))/total_dl_time:.2f} MB/s)")
    
    assemble_parts(part_files, ZIP_PATH)
    extract_color()
    validate_and_catalog()
    print("[SUCCESS] PlantVillage dataset is completely downloaded, verified, and ready!")

if __name__ == "__main__":
    main()

