import os
import csv
from openai import OpenAI
import base64
from dotenv import load_dotenv



# possible implement batch method !


IMAGE_FOLDER = "data/images"      
OUTPUT_CSV = "results-explain.csv"

# Set as environment variable later
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.1"      

client = OpenAI(api_key=API_KEY)

# Function to encode the image
def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(base64_image):
    response = client.responses.create(
        model=MODEL_NAME,
        #reasoning={"effort": "medium"},          # somehow this doesnt return a response yet
        #text={"verbosity": "low"},               # Only want 1-word output
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        #"text": "Is this image real or AI-generated? Only answer 'real' or 'fake'."
                        "text": "Is this image real or AI-generated? Explain why and give a confidence score."

                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
            }
        ],
        max_output_tokens=60
    )

    return response.output_text.strip()


def main():
    # Load image file paths
    images = []
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for f in files:
            if f.lower().endswith(("jpg", "jpeg", "png", "webp")):
                images.append(os.path.join(root, f))


    if not images:
        print("No images found in folder:", IMAGE_FOLDER)
        return

    print(f"Found {len(images)} images. Starting analysis...\n")

    results = []

    for img_path in images:
        print(f"Uploading: {img_path}")
        base64_image  = encode_image(img_path)

        print("Analyzing...")
        result = analyze_image(base64_image)

        print(f"Result for {os.path.basename(img_path)} → {result}\n")

        results.append([os.path.basename(img_path), result])

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "real_or_fake"])
        writer.writerows(results)

    print(f"\nDone! Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
