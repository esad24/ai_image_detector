import os
import json

# ==========================
# SET YOUR ROOT FOLDER HERE
# ==========================
ROOT_FOLDER = "/home/usluesyr/ai_image_detector/data/real"


def update_accuracy(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detected_real = data.get("detected_real")
        image_count = data.get("image_count")

        # Check if required values exist and are valid
        if detected_real is None or image_count is None:
            print(f"Missing keys in: {file_path}")
            return

        if image_count == 0:
            print(f"image_count is 0 in: {file_path}")
            return

        # Compute accuracy
        acc = detected_real / image_count

        # Add or overwrite acc
        data["acc"] = acc

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated acc in: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file == "results.json":
                full_path = os.path.join(root, file)
                update_accuracy(full_path)


if __name__ == "__main__":
    main()
