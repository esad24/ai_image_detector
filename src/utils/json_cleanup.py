import os
import json

folder = "/home/usluesyr/ai_image_detector/data/real/test/results/gpt-5.2"

recursive = True

def clean_json_file(file_path):
    """Load a JSON file and save a cleaned copy with proper Unicode in the same folder."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Neuen Dateinamen mit _cleaned suffix
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        cleaned_file = os.path.join(dir_name, f"{name}_cleaned{ext}")
        
        with open(cleaned_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned: {file_path} -> {cleaned_file}")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

if recursive:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".json"):
                clean_json_file(os.path.join(root, file))
else:
    for file in os.listdir(folder):
        if file.lower().endswith(".json"):
            clean_json_file(os.path.join(folder, file))
