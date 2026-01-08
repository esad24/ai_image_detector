# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MODEL_NAME = "gpt-5.2"      

# client = OpenAI(api_key=OPENAI_API_KEY)

# # Function to encode the image
# def encode_image(path):
#     with open(path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# def analyze_image(base64_image):
#     response = client.responses.parse(
#         model=MODEL_NAME,
#         #reasoning={"effort": "medium"},          # somehow this doesnt return a response yet
#         #text={"verbosity": "low"},               # Only want 1-word output
#         input=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "input_text",
#                         "text": PROMPTS["prompt3"]["text"]
#                     },
#                     {
#                         "type": "input_image",
#                         "image_url": f"data:image/jpeg;base64,{base64_image}",
#                     }
#                 ]
#             }
#         ],
#         text_format=ImageAuthenticityResult,        
#         max_output_tokens=1000
#     )
#     return response.output_parsed
