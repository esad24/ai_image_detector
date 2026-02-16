import os
import json

# ==========================
# SET YOUR ROOT FOLDER HERE
# ==========================
ROOT_FOLDER = "/home/usluesyr/ai_image_detector/data/real"


def update_results_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False

        # Rename total_fakes -> detected_fake
        if "total_fakes" in data:
            data["detected_fake"] = data.pop("total_fakes")
            changed = True

        # Rename total_real -> detected_real
        if "total_real" in data:
            data["detected_real"] = data.pop("total_real")
            changed = True

        if changed:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Updated: {file_path}")
        else:
            print(f"No changes needed: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file == "results.json":
                full_path = os.path.join(root, file)
                update_results_json(full_path)


if __name__ == "__main__":
    main()
