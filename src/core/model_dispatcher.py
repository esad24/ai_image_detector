from model.openai_model import OpenAIModel

def get_model(model_name: str, api_key: str):
    if model_name == "gpt-5.2":
        return OpenAIModel(model_name=model_name, api_key=api_key)

    raise ValueError(f"Unknown model: {model_name}")
