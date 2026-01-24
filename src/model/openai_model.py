import os
import base64
from openai import OpenAI


class OpenAIModel():

    def __init__(self, model_name: str = "gpt-5.2", api_key: str=None):
        if not api_key:
            raise ValueError("API Key must be provided")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def send_image(self, image_path: str, prompt: dict) -> dict:
        base64_image = self._encode_image(image_path)

        response = self.client.responses.parse(
        #response = self.client.responses.create(
            model=self.model_name,
            reasoning={"effort": "high"},
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
            text_format=prompt["schema"],        
            #max_output_tokens=1000
        )

        #return response.output_parsed
        return response.output_text
