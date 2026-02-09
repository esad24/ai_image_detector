import os

FOLDER_PATH = "/home/usluesyr/ai_image_detector/data/ground_truth/gt_2/labled_test_second_round"

for filename in os.listdir(FOLDER_PATH):
    old_path = os.path.join(FOLDER_PATH, filename)

    if not os.path.isfile(old_path):
        continue

    new_name = filename

    if new_name.startswith("Gemini_Generated_Image_"):
        new_name = new_name.replace("Gemini_Generated_Image_", "", 1)

    name, ext = os.path.splitext(new_name)
    if ext.lower() == ".jpg":
        new_name = name + ".jpeg"

    if new_name != filename:
        new_path = os.path.join(FOLDER_PATH, new_name)
        os.rename(old_path, new_path)

print("Done.")
