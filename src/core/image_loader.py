import os

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

def load_images(image_folder):
    if(image_folder == "genClass_fake"):
        folder_path = "/home/usluesyr/ai_image_detector/data/genClass/fake/images"
    elif(image_folder == "genClass_real"):
        folder_path = "/home/usluesyr/ai_image_detector/data/genClass/real/images"
    elif(image_folder == "genArtifact_fake"):
        folder_path = "/home/usluesyr/ai_image_detector/data/genArtifact/fake/test/images"
    elif(image_folder == "genArtifact_real"):
        folder_path = "/home/usluesyr/ai_image_detector/data/genArtifact/real/test/images"
    elif(image_folder == "real2gen_fake"):
        folder_path = "/home/usluesyr/ai_image_detector/data/real2gen/fake/images"
    elif(image_folder == "real2gen_fake"):
        folder_path = "/home/usluesyr/ai_image_detector/data/real2gen/real/images"
    else:
        if not os.path.isdir(image_folder):
            raise ValueError("Image folder does not exist")
        folder_path = image_folder

    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(ALLOWED_EXTENSIONS):
            images.append(os.path.join(folder_path, file))

    if not images:
        raise ValueError("No images found in folder")

    return images
