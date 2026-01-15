import json

def load_prompt(prompt_id: int):
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)

    key = str(prompt_id)
    if key not in prompts:
        raise ValueError("Prompt ID not found")

    return prompts[key]

