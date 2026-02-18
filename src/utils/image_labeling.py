import os
import csv
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

MODE = "real"

IMAGE_FOLDER = f"/home/usluesyr/ai_image_detector/data/{MODE}/test/images"
OUTPUT_CSV = f"/home/usluesyr/ai_image_detector/data/{MODE}/test/labeled_{MODE}_images.csv"
MODEL_NAME = "gpt-5-mini"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API Key must be provided")

client = OpenAI(api_key=OPENAI_API_KEY)


class ImageResponse(BaseModel):
    category: Literal["Close-Up", "Portrait", "Group"]
    scene: Literal["true", "false"]
    description: str


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def send_image(image_path: str, prompt: dict) -> dict:
    base64_image = encode_image(image_path)

    result = client.responses.parse(
        model=MODEL_NAME,
        #reasoning={"effort": "none"},
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt["text"]
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        text_format=ImageResponse,
    )

    parsed = result.output_parsed
    if parsed is None:
        return {"category": "ERROR", "scene": "ERROR", "description": "ERROR"}

    return {"category": parsed.category, "scene": parsed.scene, "description": parsed.description}


PROMPT = {
    "text": """
You are an image labeling assistant.
Label the image.
Return as a JSON, do not include anything else, explanations, or extra text.
"""
}

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["filename", "category", "scene", "description"])
    writer.writeheader()

    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            continue

        filepath = os.path.join(IMAGE_FOLDER, filename)
        print(f"Processing {filename}...")

        result = send_image(filepath, PROMPT)

        writer.writerow({
            "filename": filename,
            "category": result.get("category", "ERROR"),
            "scene": result.get("scene", "ERROR"),
            "description": result.get("description", "ERROR")
        })

print(f"Labeling complete. Results saved in {OUTPUT_CSV}")