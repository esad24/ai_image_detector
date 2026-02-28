import os
import json
import shutil

def false_negatives(image_folder, json_file):
    json_dir = os.path.dirname(os.path.abspath(json_file))
    destination_folder = os.path.join(json_dir, "false_negatives")

    os.makedirs(destination_folder, exist_ok=True)

    with open(json_file, "r") as f:
        data = json.load(f)

    for item in data["results"]:
        filename = item.get("filename")
        classification = item.get("classification")

        # Only consider "real" images
        if classification == "real":
            source_path = os.path.join(image_folder, filename)
            dest_path = os.path.join(destination_folder, filename)

            if os.path.isfile(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {filename}")
            else:
                print(f"File not found: {filename}")

if __name__ == "__main__":
    # Example usage
    image_folder = "/home/usluesyr/ai_image_detector/data/genArtifact/fake/test/images"
    json_file = "/home/usluesyr/ai_image_detector/data/genArtifact/fake/test/results/qwen3-vl/prompt1/2026-01-22_21-17-46/results.json"
    false_negatives(image_folder, json_file)
