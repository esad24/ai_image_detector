from ollama import chat
from utils.base64_encoder import encode_image
from pydantic import ValidationError
import time

#models = ["qwen3-vl:8b", "llava:7b", 'gemma3:4b',]

class OllamaModel():

    def __init__(self, model_name : str, temp = 0.1, reasoning = False, max_retries=3):
        self.model_name = "qwen3-vl:8b"
        self.temp = temp
        self.reasoning = reasoning
        self.max_retries = max_retries

    def send_image(self, image_path: str, prompt: dict) -> dict:
      base64_image = encode_image(image_path)
      text_prompt = prompt["text"]
      schema = prompt["schema"]

      for attempt in range(1, self.max_retries + 1):
        try:
          response = chat(
            model=self.model_name,
            messages=[
              {
                'role': 'user',
                'content': text_prompt,
                'images': [base64_image]
              },
            ],
            format=schema.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
            # options={
            #   'temperature' : self.temp,
            # }
          )
          content = response.message.content

          if not content or not content.strip():
            raise ValueError("Empty model response")
        
          result = schema.model_validate_json(content)
          return result
         
        except (ValueError, ValidationError) as e:
            print(f"[RETRY {attempt}/{self.max_retries}] {image_path} → {e}")

        except Exception as e:
            print(f"[ERROR {attempt}/{self.max_retries}] {image_path} → {e}")

        time.sleep(1.5) 


        print(f"[FAILED] {image_path} after {self.max_retries} retries")
        return None