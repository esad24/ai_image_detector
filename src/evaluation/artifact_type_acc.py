import json

evaluation_json = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/semantic_evaluation.json"

# Load your JSON file
with open(evaluation_json, "r", encoding="utf-8") as f:
    data = json.load(f)

total_score = 0
files_with_matches = 0

for image in data.get("image_scores", []):
    artifact_scores = []

    for match in image.get("matched_artifacts", []):
        result_artifact = match.get("result_artifact")
        gt_artifact = match.get("matched_gt_artifact")

        # Only consider matches where both sides exist and have a type
        if result_artifact and isinstance(gt_artifact, dict) and "type" in gt_artifact:
            if result_artifact.get("type") == gt_artifact.get("type"):
                artifact_scores.append(1)  # same type
            else:
                artifact_scores.append(0)  # different type (normalized)

    # Calculate normalized file score
    if artifact_scores:
        file_score = sum(artifact_scores) / len(artifact_scores)
        image["artifact_type_acc"] = file_score
        total_score += file_score
        files_with_matches += 1
    else:
        image["artifact_type_acc"] = None  # No valid matches

# Calculate dataset average
dataset_average = total_score / files_with_matches if files_with_matches > 0 else 0

# Ensure dataset_average exists and add the metric
if "dataset_average" not in data or not isinstance(data["dataset_average"], dict):
    data["dataset_average"] = {}
data["dataset_average"]["mean_artifact_type_acc"] = dataset_average

# Save the updated JSON
output_file = "artifact_type_acc.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Processed {len(data['image_scores'])} files.")
print(f"Dataset average normalized artifact type match score: {dataset_average}")
