from core.argument_parser import parse_args
from core.image_loader import load_images
from core.model_dispatcher import get_model
from io.result_writer import ResultWriter
from config.prompts import load_prompt

def main():
    args = parse_args()

    images = load_images(args.image_folder)
    prompt = load_prompt(args.prompt_id)
    model = get_model(args.model)

    writer = ResultWriter(
        model_name=args.model,
        prompt_id=args.prompt_id
    )

    for image_path in images:
        result = model.send_image(image_path, prompt)
        writer.write(image_path, result)

    writer.save_summary(len(images))

if __name__ == "__main__":
    main()
