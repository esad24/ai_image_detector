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
    "text": "Analyze the image and determine whether it is real or AI-generated. If the image is fake, identify all visible artifacts using only the following categories: 'structural', 'physics', 'semantic', 'stylistic'. For each artifact, specify its type, describe the anomaly, and indicate its spatial location in the image."
  },
  "prompt4": {
    "label": "classification + free-form reasoning",
    "text": "Analyze the image in detail and explain all visual cues relevant to authenticity. Do not use predefined artifact categories. Provide a comprehensive forensic reasoning and conclude whether the image is real or AI-generated."
  },
  "prompt5": {
    "label": "classification + step-by-step reasoning",
    "text": "Analyze the image step by step, identifying any visual inconsistencies, artifacts, or anomalies. After completing the analysis, conclude whether the image is real or AI-generated."
  }
}
