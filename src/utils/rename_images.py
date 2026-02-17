import os

FOLDER_PATH = "/home/usluesyr/ai_image_detector/data/real/test/images"

for filename in os.listdir(FOLDER_PATH):
    old_path = os.path.join(FOLDER_PATH, filename)

    if not os.path.isfile(old_path):
        continue

    new_name = filename

    # if new_name.startswith("Gemini_Generated_Image_"):
    #     new_name = new_name.replace("Gemini_Generated_Image_", "", 1)

    # name, ext = os.path.splitext(new_name)
    # if ext.lower() == ".jpg":
    #     new_name = name + ".jpeg"

    # if new_name != filename:
    #     new_path = os.path.join(FOLDER_PATH, new_name)
    #     os.rename(old_path, new_path)

    # Remove "_real" at the end of the filename (before extension)
    name, ext = os.path.splitext(new_name)
    if name.endswith("_real"):
        name = name[:-5]  # remove last 5 characters
        new_name = name + ext

    if new_name != filename:
        new_path = os.path.join(FOLDER_PATH, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

print("Done.")
