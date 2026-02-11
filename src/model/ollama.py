from ollama import chat
from utils.base64_encoder import encode_image

models = ["qwen3-vl:8b", "llava:7b", 'gemma3:4b', 'kimi-k2.5:cloud']

class OllamaModel():

    def __init__(self, model_name : str, temp = 0.1, reasoning = False):
        self.model_name = 'kimi-k2.5:cloud'
        #self.temp = temp
        self.reasoning = reasoning

    def send_image(self, image_path: str, prompt: dict) -> dict:
      base64_image = encode_image(image_path)
      text_prompt = prompt["text"]
      schema = prompt["schema"]


      response = chat(
        model=self.model_name,
        messages=[
          {
            'role': 'user',
            'content': text_prompt,
            'images': [base64_image]
          },
        ],
        format=schema.model_json_schema()  # Use Pydantic to generate the schema or format=schema
        # options={
        #   'temperature' : self.temp,
        # }

      )
      #print('Thinking:\n', response.message.thinking)

      #final_response = prompt["schema"].model_validate_json(response.message.content)

      return response.message.content