import os

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

def load_images(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Image folder does not exist")

    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(ALLOWED_EXTENSIONS):
            images.append(os.path.join(folder_path, file))

    if not images:
        raise ValueError("No images found in folder")

    return images
