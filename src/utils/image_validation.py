import os

image_path = "/home/usluesyr/ai_image_detector/data/genClass/real/images/3781951e1d5a87db45afd556c3e1c1f3acebeb8d7af9f451417a4dc8eee3b530.jpeg"

print(os.path.exists(image_path))  # Should be True
print(image_path.lower())           # Should end with .jpg, .jpeg, .png, .gif, or .webp

from PIL import Image

try:
    img = Image.open(image_path)
    img.verify()  # Will raise an exception if not a valid image
    print("Image is valid")
except Exception as e:
    print("Invalid image:", e)


from mimetypes import guess_type

mime_type, _ = guess_type(image_path)
print(mime_type)  # Should be image/jpeg, image/png, etc.