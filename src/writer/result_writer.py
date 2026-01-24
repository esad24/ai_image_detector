import json
import os
from datetime import datetime


class ResultWriter:

    def __init__(self, image_folder, model_name, prompt_id):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if(image_folder == "train" or image_folder == "test"):
            self.run_dir = f"/home/usluesyr/ai_image_detector/data/{image_folder}/results/{model_name}/prompt{prompt_id}/{timestamp}"
        else:
            self.run_dir = f"{image_folder}/results/{model_name}/prompt{prompt_id}/{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)

        self.results = []
        self.meta = {
            "model": model_name,
            "prompt_id": prompt_id
        }

    def write(self, image_path, result):
        # Parse result if it's a JSON string
        if isinstance(result, str):
            result = json.loads(result)
        
        # Merge filename with result fields, filename first
        if isinstance(result, dict):
            result = {"filename": os.path.basename(image_path), **result}
        
        self.results.append(result)

    def save_summary(self, image_count):
        output = {
            "meta": self.meta,
            "image_count": image_count,
            "results": self.results
        }

        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(output, f, indent=2)
        print("result saved in " + self.run_dir)
