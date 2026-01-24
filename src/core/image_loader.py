import os

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

def load_images(image_folder):
    if(image_folder == "fake_train"):
        folder_path = "/home/usluesyr/ai_image_detector/data/fake/train/images"
    elif(image_folder == "fake_test"):
        folder_path = "/home/usluesyr/ai_image_detector/data/fake/test/images"
    elif(image_folder == "real_train"):
        folder_path = "/home/usluesyr/ai_image_detector/data/real/train/images"
    elif(image_folder == "real_test"):
        folder_path = "/home/usluesyr/ai_image_detector/data/real/test/images"
    else:
        if not os.path.isdir(image_folder):
            raise ValueError("Image folder does not exist")

    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(ALLOWED_EXTENSIONS):
            images.append(os.path.join(folder_path, file))

    if not images:
        raise ValueError("No images found in folder")

    return images
