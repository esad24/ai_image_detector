import json
import os
import subprocess

IMAGE_DIR = "/home/usluesyr/ai_image_detector/data/ground_truth/gt_2/labled_test_second_round"
JSON_PATH = "data/fake/test/results/qwen3-vl/prompt3/2026-01-22_21-28-42/results.json"
EDITOR = "code"

def open_image(path):
    subprocess.Popen(["code", path])


def open_json(path):
    subprocess.Popen([EDITOR, path])

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [])
#results = data.get("image_scores", [])

open_json(JSON_PATH)

for idx, entry in enumerate(results, start=1):
    filename = entry.get("filename")
    image_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(image_path):
        print(f"Image missing: {filename}")
        continue

    print(f"\n[{idx}/{len(results)}] Reviewing {filename}")
    print(f"Classification: {entry.get('classification')}")
    print(f"Artifacts: {len(entry.get('artifacts', []))}")
    print(f"Image path: {image_path}")

    open_image(image_path)

    input("Edit JSON if needed, then press ENTER for next image...")
