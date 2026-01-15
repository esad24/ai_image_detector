PROMPTS = {
  "prompt1": {
    "label": "classification",
    "text": "Analyze the image and decide whether it is real or AI-generated. Answer with only one word: real or fake. Do not provide any explanation."
  },
  "prompt2": {
    "label": "classification + artifact categories",
    "text": "Analyze the image and decide whether it is real or AI-generated. If it is fake identify which artifact types are present using only the following categories: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'. Respond with a JSON object containing exactly two fields: 'classification' (value: 'real' or 'fake') and 'artifacts' (value: a list of artifact labels). Do not provide any explanation."
  },
  "prompt3": {
    "label": "classification + artifacts with reasoning and location",
    "text": "Analyze the image and determine whether it is real or AI-generated. If the image is fake, identify all visible artifacts using only the following categories: 'structural', 'physical', 'semantic', 'stylistic'. For each artifact, specify its type, describe the anomaly, and indicate its spatial location in the image."
  },
  "prompt4": {
    "label": "classification + free-form reasoning",
    "text": """Analyze the image and determine whether it is real or AI-generated. 
              If the Image is fake explain all visual cues relevant to authenticity. 
              Return ONLY valid JSON with the fields:
                  - classification: "real" or "fake"
                  - explanation
            """
  },

  "prompt5": {
    "label": "classification + step-by-step reasoning",
    "text": "Analyze the image step by step, identifying any visual inconsistencies, artifacts, or anomalies. After completing the analysis, conclude whether the image is real or AI-generated."
  },
  "prompt6": {
      "label": "explanation in json",
      "text": """
                Analyze the image and decide whether it is real or AI-generated.Return ONLY valid JSON with the fields:
                  - classification: "real" or "fake"
                  - artifacts: a list of objects with:
                    - type: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'
                    - description: explanation of the artifact including its location in the image
                If the image is real, return an empty artifacts list.
                Do not include any text outside the JSON.
              """
  }
}
