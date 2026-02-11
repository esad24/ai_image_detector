import json
import re
import os
from datetime import datetime

class ResultWriter:

    def __init__(self, image_folder, model_name, reasoning, prompt_id, resume=False):
        """
        resume: if True, try to resume the latest previous run for this folder/model/prompt_id
        """
        base_dir = ""
        if image_folder.startswith("real_train"):
            base_dir = f"/home/usluesyr/ai_image_detector/data/real/train/results/{model_name}/prompt{prompt_id}"
        elif image_folder.startswith("real_test"):
            base_dir = f"/home/usluesyr/ai_image_detector/data/real/test/results/{model_name}/prompt{prompt_id}"
        elif image_folder.startswith("fake_train"):
            base_dir = f"/home/usluesyr/ai_image_detector/data/fake/train/results/{model_name}/prompt{prompt_id}"
        elif image_folder.startswith("fake_test"):
            base_dir = f"/home/usluesyr/ai_image_detector/data/fake/test/results/{model_name}/prompt{prompt_id}"
        else:
            base_dir = f"{image_folder}/results/{model_name}/prompt{prompt_id}"

        os.makedirs(base_dir, exist_ok=True)

        # Resume logic
        if resume:
            previous_runs = sorted(os.listdir(base_dir))
            if previous_runs:
                self.run_dir = os.path.join(base_dir, previous_runs[-1])
                print(f"Resuming previous run: {self.run_dir}")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.run_dir = os.path.join(base_dir, timestamp)
                os.makedirs(self.run_dir, exist_ok=True)
                print(f"No previous run found. Starting new run: {self.run_dir}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_dir = os.path.join(base_dir, timestamp)
            os.makedirs(self.run_dir, exist_ok=True)
            print(f"Starting new run: {self.run_dir}")

        self.file_path = os.path.join(self.run_dir, "results.json")

        self.fake = 0
        self.real = 0
        self.meta = {
            "model": model_name,
            "prompt_id": prompt_id,
            #"temperature": temp,
            "reasoning": reasoning
        }

        if os.path.exists(self.file_path):
            self._load_counters()
        else:
            self._init_file()

    def _init_file(self):
        base = {
            "meta": self.meta,
            "image_count": 0,
            "total_fakes": 0,
            "total_real": 0,
            "results": []
        }
        with open(self.file_path, "w") as f:
            json.dump(base, f, indent=2)

    def _load(self):
        with open(self.file_path, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_counters(self):
        data = self._load()
        self.fake = data.get("total_fakes", 0)
        self.real = data.get("total_real", 0)

    def get_processed_filenames(self):
        if not os.path.exists(self.file_path):
            return set()
        data = self._load()
        return set(r["filename"] for r in data.get("results", []))

    def write(self, image_path, result):
        # --- Sicherstellen, dass result ein Dict ist ---
        if isinstance(result, str):
            # Extrahiere nur den JSON-Teil zwischen { und }
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    print(f"JSON parse error for {image_path}: {e}")
                    return
            else:
                print(f"No JSON found in result for {image_path}")
                return

        elif not isinstance(result, dict):
            print(f"Invalid result type for {image_path}: {type(result)}")
            return

        # Füge Dateiname hinzu
        result = {"filename": os.path.basename(image_path), **result}

        classification = result.get("classification")
        if classification == "fake":
            self.fake += 1
        elif classification == "real":
            self.real += 1

        # load current results
        data = self._load()
        data["results"].append(result)
        data["image_count"] = len(data["results"])
        data["total_fakes"] = self.fake
        data["total_real"] = self.real

        self._save(data)

    def save_summary(self, image_count):
        print("Final summary already saved incrementally in:", self.run_dir)
