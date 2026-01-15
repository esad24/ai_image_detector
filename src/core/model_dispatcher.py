from model.openai_model import OpenAIModel

def get_model(model_name: str):
    if model_name == "gpt-5.2":
        return OpenAIModel()

    raise ValueError(f"Unknown model: {model_name}")
