import urllib.request
import re
import json

url = 'https://data.mendeley.com/datasets/brfgw46wzb/2'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        html = r.read().decode('utf-8', errors='ignore')
        print('Mendeley page fetched, length:', len(html))
        title = re.findall(r'<title>(.*?)</title>', html)
        print('Title:', title)
        # Check for json blob in script tags
        script_data = re.findall(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
        if script_data:
            data = json.loads(script_data[0])
            dataset_version = data.get('props', {}).get('pageProps', {}).get('datasetVersion', {})
            print('Dataset Name:', dataset_version.get('name'))
            print('Description snippet:', dataset_version.get('description', '')[:300])
            files = dataset_version.get('files', [])
            print('Files count:', len(files))
            for f in files:
                print(f"  {f.get('filename')} - {f.get('size')} bytes - download: {f.get('download_url')}")
except Exception as e:
    print('Error:', e)

