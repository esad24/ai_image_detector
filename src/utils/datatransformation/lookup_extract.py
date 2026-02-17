import os
import csv
import pandas as pd

# ====== CONFIG ======
original_csv_path = "/home/usluesyr/ai_image_detector/src/utils/datatransformation/stimuli_image_metadata.csv"
real_folder = "/home/usluesyr/ai_image_detector/data/real/test/images"
fake_folder = "/home/usluesyr/ai_image_detector/data/fake/test/images"
output_csv_path = "/home/usluesyr/ai_image_detector/src/utils/datatransformation/images_metadata.csv"
# ====================

# Load original CSV
df = pd.read_csv(original_csv_path)

# Create lookup dictionary
lookup = dict(zip(df["filename"], df["filename_old"]))

rows = []

def try_lookup(filename):
    # Direct match
    if filename in lookup:
        return filename
    
    # Try adding _real before extension
    name, ext = os.path.splitext(filename)
    candidate = name + "_real" + ext
    if candidate in lookup:
        return candidate
    
    return None


def detect_generator(old_filename):
    if not old_filename:
        return ""
    old_filename = old_filename.lower()
    if "mj_" in old_filename:
        return "Midjourney"
    if "sd_" in old_filename:
        return "Stable Diffusion"
    if "ff_" in old_filename:
        return "Adobe Firefly"
    return ""


def process_folder(folder_path, classification_label):
    for file in os.listdir(folder_path):
        matched_filename = try_lookup(file)
        
        if matched_filename:
            old_filename = lookup[matched_filename]
        else:
            old_filename = ""
        
        generator = detect_generator(old_filename)
        
        rows.append({
            "filename": file,
            "classification": classification_label,
            "old_filename": old_filename,
            "generator": generator
        })


# Process folders
process_folder(fake_folder, "fake")
process_folder(real_folder, "real")

# Create DataFrame
output_df = pd.DataFrame(rows)

# Sort so fake images appear first
output_df["classification"] = pd.Categorical(
    output_df["classification"],
    categories=["fake", "real"],
    ordered=True
)
output_df = output_df.sort_values("classification")

# Save CSV
output_df.to_csv(output_csv_path, index=False)

print(f"Done. Saved to {output_csv_path}")