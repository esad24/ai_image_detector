import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.base64_encoder import encode_image
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
            raise ValueError("API Key must be provided")

class OpenAIModel():

    def __init__(self, model_name = "gpt-5.2", temp = None, reasoning = "none"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.reasoning = reasoning
        self.temp = temp

    def send_image(self, image_path: str, prompt: dict) -> dict:
        base64_image = encode_image(image_path)

        response = self.client.responses.parse(
        #response = self.client.responses.create(
            model = self.model_name,
            reasoning = {"effort": self.reasoning},
            #temperature = self.temp,
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
