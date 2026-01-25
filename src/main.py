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
    model = get_model(args.model)

    writer = ResultWriter(
        image_folder=args.image_folder,
        model_name=args.model,
        temp=model.temp,
        reasoning=model.reasoning,
        prompt_id=args.prompt_id,
        resume=args.resume
    )

    # get already processed filenames to skip
    processed_files = writer.get_processed_filenames()
    if processed_files:
        print(f"Skipping {len(processed_files)} already processed images.")

    for image_path in images:
        filename = os.path.basename(image_path)
        if filename in processed_files:
            print(f"Skipping {filename} (already processed)")
            continue

        print(f"Uploading: {image_path}")
        result = model.send_image(image_path, prompt)
        print(f"Result for {filename} → {result}\n")
        writer.write(image_path, result)

    writer.save_summary(len(images))


if __name__ == "__main__":
    main()



"""
gpt-5.2
python3 main.py -i fake_train -m gpt-5.2 -p 
python3 main.py -i real_train -m gpt-5.2 -p 

python3 main.py -i fake_test -m gpt-5.2 -p 
python3 main.py -i real_test -m gpt-5.2 -p 

qwen3
python3 main.py -i fake_train -m qwen3-vl -p 
python3 main.py -i real_train -m qwen3-vl -p

python3 main.py -i fake_test -m qwen3-vl -p 
python3 main.py -i real_test -m qwen3-vl -p  

llava
python3 main.py -i fake_train -m llava -p 
python3 main.py -i real_train -m llava -p 

python3 main.py -i fake_test -m llava -p 
python3 main.py -i real_test -m llava -p 

gemma3
python3 main.py -i fake_train -m gemma3 -p 
python3 main.py -i real_train -m gemma3 -p 

python3 main.py -i fake_train -m gemma3 -p 
python3 main.py -i real_train -m gemma3 -p 

"""