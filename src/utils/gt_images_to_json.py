import os
import json
import base64
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from dotenv import load_dotenv
from config.response_schema import reasoning

load_dotenv()

# possibly implement batch method !


# IMAGE_FOLDER = "train"
IMAGE_FOLDER = "data/ground_truth/gt_2/labled_test_second_round"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.2"      

client = OpenAI(api_key=OPENAI_API_KEY)

# Function to encode the image
def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(base64_image):
    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """This fake image is labeled with artifact types and explanation.
                                Only include the green labels, not the pink ones !
                                The explanations are bullet points. Turn them into grammatically
                                sentences, but don't add any new information, 
                                besides the location of the artifact, if it is missing.
                                Create a JSON in the following structure:
                                {
                                    "filename": "04e61d9506ffe1d8ac6c48e8b091cc48.jpeg",
                                    "classification": "fake",
                                    "artifacts": [
                                        {
                                            "type": (structural, semantic, physics, stylistic),
                                            "reasoning": "",
                                            "location": ""
                                        },
                            """

                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
            }
        ],
        #max_output_tokens=1000,
        text_format=reasoning
    )
    return response.output_parsed


def main():
    images = []
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for f in files:
            if f.lower().endswith(("jpg", "jpeg", "png")):
                images.append(os.path.join(root, f))


    if not images:
        print("No images found in folder:", IMAGE_FOLDER)
        return

    print(f"Found {len(images)} images. Starting analysis...\n")

    results = []

    i = 0
    for img_path in images:
        print(f"Uploading: {img_path}")
        base64_image  = encode_image(img_path)

        print("Analyzing...")
        result = analyze_image(base64_image)

        print(f"Result for {os.path.basename(img_path)} → {result}\n")

        new_path = img_path.replace("labeled_", "")
        results.append([os.path.basename(new_path), result])

        # i += 1
        # if i == 2: break

    # OUTPUT_JSON = "train.json"
    OUTPUT_JSON = "labeled2.json"


    # Convert results to serializable format
    serializable_results = []
    for filename, result in results:
        serializable_results.append({
            "filename": filename,
            "classification": result.classification,
            "artifacts": [a.__dict__ for a in result.artifacts]
        })

    # Save to JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":

    main()
