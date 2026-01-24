from config.response_schema import artifacts, classification, reasoning, explain

prompt_dict = {
      "1": {
        "text": "Tell me if this is an AI-generated image. Answer real or fake. Respond ONLY with valid JSON",
        "schema": classification
        },
      "2": {
        "text": "Tell me if this is an AI-generated image. If it is fake identify which artifact types are present using only the following categories: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'. Do not provide any explanation. Respond ONLY with valid JSON",
        "schema": artifacts
        },
      "3": {
        "text": "Tell me if this is an AI-generated image. If the image is fake, identify all visible artifacts using only the following categories: 'structural' (Structural & Component Defects), 'physics' (violation of the laws of physics), 'semantic' (Semantic & Functional Implausibilities), 'stylistic'. For each artifact, specify its type, describe the anomaly and indicate its spatial location in the image. Respond ONLY with valid JSON",
        "schema": reasoning
        },
      "4": {
        "text": "Tell me if synthesis artifacts are in the this image. Must return with 1) fake or real; 2) if real, explain why and where the artifacts exist. Respond ONLY with valid JSON",
        "schema": explain
      },
      "5": {
        "text": "This is a fake image. Explain why and where the artifacts exist. Respond ONLY with valid JSON",
        "schema": explain
      },
      "6": {
        "text": "This is a real image. Explain why. Respond ONLY with valid JSON",
        "schema": explain
      }
}
