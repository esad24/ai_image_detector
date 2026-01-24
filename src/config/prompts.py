from config.response_schema import artifacts

prompt_dict = {
  "1": "Is this image real or fake? Return only 'real' or 'fake'.",
  "2": {
        "text": "Analyze the image and decide whether it is real or AI-generated. If it is fake identify which artifact types are present using only the following categories: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'. Do not provide any explanation.",
        "schema": artifacts
      }
}
