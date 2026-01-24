import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

class ResultWriter:

    def __init__(self, image_folder, model_name, temp, reasoning, prompt_id):

        # german_tz = ZoneInfo("Europe/Berlin")
        # now_germany = datetime.now(german_tz)
        # timestamp = now_germany.strftime("%Y-%m-%d_%H-%M-%S")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if(image_folder == "real_train"):
            self.run_dir = f"/home/usluesyr/ai_image_detector/data/real/train/results/{model_name}/prompt{prompt_id}/{timestamp}"
        elif(image_folder == "real_test"):
            self.run_dir = f"/home/usluesyr/ai_image_detector/data/real/test/results/{model_name}/prompt{prompt_id}/{timestamp}"
        elif(image_folder == "fake_train"):
            self.run_dir = f"/home/usluesyr/ai_image_detector/data/fake/train/results/{model_name}/prompt{prompt_id}/{timestamp}"
        elif(image_folder == "fake_test"):
            self.run_dir = f"/home/usluesyr/ai_image_detector/data/fake/test/results/{model_name}/prompt{prompt_id}/{timestamp}"
        else:
            self.run_dir = f"{image_folder}/results/{model_name}/prompt{prompt_id}/{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)

        self.results = []
        self.fake = 0
        self.real = 0
        self.meta = {
            "model": model_name,
            "prompt_id": prompt_id,
            "temperature": temp,
            "reasoning": reasoning
        }

    def write(self, image_path, result):
        # Parse result if it's a JSON string
        if isinstance(result, str):
            result = json.loads(result)
        
        # Merge filename with result fields, filename first
        if isinstance(result, dict):
            result = {"filename": os.path.basename(image_path), **result}
        
        classification = result["classification"]
        if(classification == "fake"): 
            self.fake += 1
        elif(classification == "real"):
            self.real += 1

        
        self.results.append(result)

    def save_summary(self, image_count):
        output = {
            "meta": self.meta,
            "image_count": image_count,
            "total_fakes": self.fake,
            "total_real": self.real,
            "results": self.results
        }

        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(output, f, indent=2)
        print("result saved in " + self.run_dir)
