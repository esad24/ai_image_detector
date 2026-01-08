import os
import json
import csv
import base64


from openai import OpenAI
from dotenv import load_dotenv
from prompts import PROMPTS
from response_schema import ImageAuthenticityResult

load_dotenv()

# possibly implement batch method !


IMAGE_FOLDER = "data/currentDataset"      
CLASSIFICATION_CSV = "results/classification_results.csv"
CLASSIFICATION_ARTIFACTS_CSV = "results/classification_artifacts.csv"
CLASSIFICATION_ARTIFACTS_EXPLANATION_CSV = "results/classification_artifacts_explanation.csv"


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
        #reasoning={"effort": "medium"},          # somehow this doesnt return a response yet
        #text={"verbosity": "low"},               # Only want 1-word output
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": PROMPTS["prompt3"]["text"]
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
            }
        ],
        text_format=ImageAuthenticityResult,        
        max_output_tokens=1000
    )
    return response.output_parsed


def main():
    # Load image file paths
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

        results.append([os.path.basename(img_path), result])

        #i += 1
        #if i == 2: break


# Classification

    # Save to classification_results
    # with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     # writer.writerow(["filename", "classification"])
    #     writer.writerows(results)


# Classification + Artifacts

#     with open(CLASSIFICATION_ARTIFACTS_CSV, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=['filename', 'classification', 'artifacts'])
        
#         for filename, result in results:
#             writer.writerow({
#                 'filename': filename,
#                 'classification': result_JSON.get('classification', ''),
#                 'artifacts': ", ".join(result_JSON.get('artifacts', []))
#             })

#     print(f"\nSaved results to {CLASSIFICATION_ARTIFACTS_CSV}")

# Classification + Artifact explanation

# Classification + Artifact explanation
    # with open(CLASSIFICATION_ARTIFACTS_EXPLANATION_CSV, "a", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=['filename', 'classification', 'artifacts'])
        
    #     for filename, result in results:
    #         writer.writerow({
    #             'filename': filename,
    #             'classification': result.classification,
    #             'artifacts': json.dumps([a.__dict__ for a in result.artifacts], ensure_ascii=False)
    #         })


    #     print(f"\nSaved results to {CLASSIFICATION_ARTIFACTS_EXPLANATION_CSV}")

    # import json

    OUTPUT_JSON = "results/classification_artifacts_explanation.json"

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

    print(f"\nSaved all results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
