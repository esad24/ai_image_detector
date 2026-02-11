from model.openai import OpenAIModel
from model.ollama import OllamaModel

def get_model(model_name: str):
    if model_name == "gpt-5.2":
        return OpenAIModel()
    if model_name == "qwen3-vl":
        return OllamaModel("qwen3-vl:8b")
    if model_name == ("kimi-2.5"):
        return OllamaModel("kimi-2.5")
    if model_name== ("gemma3"):
        return OllamaModel("gemma3:4b")

    raise ValueError(f"Unknown model: {model_name}")
