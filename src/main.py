from core.argument_parser import parse_args
from core.image_loader import load_images
from core.model_dispatcher import get_model
from writer.result_writer import ResultWriter
from config.prompt_loader import load_prompt

import os

def main():
    args = parse_args()
    images = load_images(args.image_folder)
    if not images:
        print("No images found")
        return
    print(f"Found {len(images)} images. Starting analysis...\n")

    prompt = load_prompt(args.prompt_id)
    model = get_model(args.model, args.api_key)

    writer = ResultWriter(
        image_folder=args.image_folder,
        model_name=args.model,
        prompt_id=args.prompt_id
    )

    for image_path in images:
        print(f"Uploading: {image_path}")
        result = model.send_image(image_path, prompt)
        print(f"Result for {os.path.basename(image_path)} → {result}\n")
        writer.write(image_path, result)

    writer.save_summary(len(images))

if __name__ == "__main__":
    main()

"""
python3 main.py /home/usluesyr/ai_image_detector/data/train 1 gpt-5.2

read -s -p "Enter API Key: " KEY python3 main.py /home/usluesyr/ai_image_detector/data/train 1 gpt-5.2 "$KEY"
"""