import json
from config.prompts import prompt_dict 

def load_prompt(prompt_id: int):

    key = str(prompt_id)
    if key not in prompt_dict:
        raise ValueError("Prompt ID not found")

    #print(prompt_dict[key])
    return prompt_dict[key]

#load_prompt(2)
