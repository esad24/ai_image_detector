# Author: Claude Opus 4.6

import os
import json
import io
import base64
import imghdr
from anthropic import Anthropic
from dotenv import load_dotenv
from utils.base64_encoder import encode_image
from PIL import Image

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("API Key must be provided")

MAX_BASE64_SIZE = 5 * 1024 * 1024  # 5MB


class AnthropicModel():

    def __init__(self, model_name="claude-sonnet-4-5-20250929", temp=None, reasoning="none"):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model_name
        self.temp = temp
        self.reasoning = reasoning

    def _detect_media_type(self, image_path: str) -> str:
        """Detect actual image format from file bytes, not extension."""
        media_map = {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        detected = imghdr.what(image_path)
        return media_map.get(detected, "image/jpeg")

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode image to base64, resizing if over 5MB limit."""
        media_type = self._detect_media_type(image_path)
        b64 = encode_image(image_path)

        if len(b64) <= MAX_BASE64_SIZE:
            return b64, media_type

        # Too large, resize
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        quality = 85
        max_dim = 1568

        while True:
            img.thumbnail((max_dim, max_dim))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            raw_bytes = buffer.getvalue()
            b64 = base64.standard_b64encode(raw_bytes).decode("utf-8")

            if len(b64) <= MAX_BASE64_SIZE:
                return b64, "image/jpeg"

            max_dim = int(max_dim * 0.8)
            quality = max(quality - 10, 40)

    def send_image(self, image_path: str, prompt: dict) -> str:
        base64_image, media_type = self._encode_image(image_path)

        schema_instruction = ""
        if prompt.get("schema"):
            schema_json = prompt["schema"].model_json_schema()
            schema_instruction = (
                "\n\nYou must respond ONLY with valid JSON matching this exact schema, "
                "no markdown, no extra text:\n"
                f"{json.dumps(schema_json, indent=2)}"
            )

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt["text"] + schema_instruction
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            }
                        }
                    ]
                }
            ],
        )

        output = response.content[0].text

        if prompt.get("schema"):
            output = output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(output)
            validated = prompt["schema"].model_validate(parsed)
            return validated.model_dump()

        return output