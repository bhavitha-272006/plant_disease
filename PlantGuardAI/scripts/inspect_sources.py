import urllib.request
import json

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

print("=== IP102 PRETRAINED MODELS & CLASSES ===")
ip102_models = fetch_json("https://api.github.com/repos/xpwu95/IP102/contents/pretrained_models")
if ip102_models:
    for m in ip102_models:
        print(f"  {m['name']} ({m.get('size', 0)} bytes)")

classes_txt = fetch_text("https://raw.githubusercontent.com/xpwu95/IP102/master/classes.txt")
if classes_txt:
    lines = [l.strip() for l in classes_txt.splitlines() if l.strip()]
    print(f"Total IP102 classes in classes.txt: {len(lines)}")
    print("First 15 classes:", lines[:15])

print("\n=== PLANTVILLAGE RAW FOLDERS ===")
pv_raw = fetch_json("https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw")
if pv_raw:
    for item in pv_raw:
        print(f"  {item['type']}: {item['name']}")

print("\n=== PLANTDOC CLASSES ===")
pd_train = fetch_json("https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents/train")
if pd_train:
    print(f"PlantDoc train classes count: {len(pd_train)}")
    for item in pd_train[:15]:
        print(f"  {item['name']}")

print("\n=== CO-LEAF MENDELEY DATASET ===")
mendeley_info = fetch_text("https://data.mendeley.com/api/datasets/brfgw46wzb/2")
if mendeley_info:
    try:
        m_json = json.loads(mendeley_info)
        print("Mendeley Title:", m_json.get("name"))
        print("Files count:", len(m_json.get("files", [])))
        for f in m_json.get("files", [])[:5]:
            print(f"  File: {f.get('filename')} ({f.get('size', 0)} bytes)")
    except Exception as e:
        print("Mendeley parse error:", e)

