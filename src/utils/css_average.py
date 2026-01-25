import json
import sys

def calculate_averages(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    image_scores = data.get("image_scores", [])

    mean_scores = []
    max_scores = []
    min_scores = []
    artifact_scores = []

    for item in image_scores:
        # image-level scores
        if isinstance(item.get("mean_cosine_similarity"), (int, float)):
            mean_scores.append(item["mean_cosine_similarity"])

        if isinstance(item.get("max_cosine_similarity"), (int, float)):
            max_scores.append(item["max_cosine_similarity"])

        if isinstance(item.get("min_cosine_similarity"), (int, float)):
            min_scores.append(item["min_cosine_similarity"])

        # artifact-level scores
        for artifact in item.get("matched_artifacts", []):
            if isinstance(artifact.get("cosine_score"), (int, float)):
                artifact_scores.append(artifact["cosine_score"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("===== Cosine Similarity Averages =====")
    print(f"Images processed: {len(image_scores)}")
    print(f"Avg mean_cosine_similarity: {avg(mean_scores)}")
    print(f"Avg max_cosine_similarity:  {avg(max_scores)}")
    print(f"Avg min_cosine_similarity:  {avg(min_scores)}")
    print(f"Avg artifact cosine_score:  {avg(artifact_scores)}")
    print(f"Total artifact scores:      {len(artifact_scores)}")

if __name__ == "__main__":


    json_path = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/semantic_evaluation.json"
    #json_path = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/semantic_evaluation_2.json"
    calculate_averages(json_path)
