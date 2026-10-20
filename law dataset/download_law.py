import os
import json
import time
import requests
from tqdm import tqdm

BASE = "https://indiacode.ecourtsindia.com/api/v1"
OUTPUT = "indian_law_dataset"

os.makedirs(OUTPUT, exist_ok=True)


def get(url):
    while True:
        r = requests.get(url, timeout=60)

        if r.status_code == 429:
            print("Rate limited. Waiting...")
            time.sleep(10)
            continue

        r.raise_for_status()
        return r.json()


# ---------------------------------------------------
# 1. Get all Acts
# ---------------------------------------------------

print("Getting list of Acts...")

url = f"{BASE}/acts?limit=1000"

acts = []

while url:
    data = get(url)

    acts.extend(data["acts"])

    url = data.get("next")

print(f"Found {len(acts)} Acts")


# Save Act index
with open(f"{OUTPUT}/acts_index.json", "w", encoding="utf-8") as f:
    json.dump(acts, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------
# 2. Download every Act
# ---------------------------------------------------

os.makedirs(f"{OUTPUT}/acts", exist_ok=True)

for i, act in enumerate(tqdm(acts)):

    act_id = act["id"]

    output_file = f"{OUTPUT}/acts/{act_id}.json"

    # Skip already downloaded files
    if os.path.exists(output_file):
        continue

    url = f"{BASE}/acts/{act_id}?include=text"

    try:
        data = get(url)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Failed: {act_id}")
        print(e)

    # Be polite to the API
    time.sleep(0.2)


print("DONE")