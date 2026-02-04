import os
import json
import shutil

def false_postives(image_folder, json_file):
    json_dir = os.path.dirname(os.path.abspath(json_file))
    destination_folder = os.path.join(json_dir, "false_postives")

    os.makedirs(destination_folder, exist_ok=True)

    with open(json_file, "r") as f:
        data = json.load(f)

    for item in data["results"]:
        filename = item.get("filename")
        classification = item.get("classification")

        # Only consider "fake" images
        if classification == "fake":
            source_path = os.path.join(image_folder, filename)
            dest_path = os.path.join(destination_folder, filename)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {filename}")
            else:
                print(f"File not found: {filename}")

    print(destination_folder)

if __name__ == "__main__":
    # Example usage
    image_folder = "/home/usluesyr/ai_image_detector/data/real/test/images"
    json_file = "/home/usluesyr/ai_image_detector/data/real/test/results/qwen3-vl/prompt4/2026-01-24_18-27-42/results.json"
    false_postives(image_folder, json_file)
