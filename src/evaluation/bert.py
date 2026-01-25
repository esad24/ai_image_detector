import json
from bert_score import score
from tqdm import tqdm
import torch
import numpy as np
import os

result = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/results.json"
ground_truth = "/home/usluesyr/ai_image_detector/data/fake/test/results/ground_truth/ground_truth.json"

# -------------------------------
# Helper functions
# -------------------------------

def artifact_to_text(artifact):
    """
    Combine type, reasoning and location into one comparable text string.
    """
    parts = []
    
    if "type" in artifact and artifact["type"]:
        parts.append(f"type: {artifact['type']}")
        
    if "reasoning" in artifact and artifact["reasoning"]:
        parts.append(f"reasoning: {artifact['reasoning']}")
        
    if "location" in artifact and artifact["location"]:
        parts.append(f"location: {artifact['location']}")
        
    return " | ".join(parts)

def extract_combined_texts(artifacts):
    """Return list of combined artifact texts."""
    return [artifact_to_text(a) for a in artifacts]

def compute_best_match_scores(result_texts, gt_texts):
    """
    Compute pairwise BERTScore matrix and return best-match F1s.
    Returns:
        recall: mean of best matches per GT artifact
        precision: mean of best matches per result artifact
        f1: harmonic mean of precision and recall
    """
    if not result_texts and not gt_texts:
        return 1.0, 1.0, 1.0
    elif not result_texts or not gt_texts:
        return 0.0, 0.0, 0.0
    
    # Pairwise F1 matrix
    f1_matrix = torch.zeros(len(result_texts), len(gt_texts))
    for i, r_text in enumerate(result_texts):
        for j, g_text in enumerate(gt_texts):
            _, _, f1 = score([r_text], [g_text], lang="en", verbose=False)
            f1_matrix[i, j] = f1.item()
    
    # Recall: for each GT, take best-matching result
    best_per_gt, _ = f1_matrix.max(dim=0)
    recall = best_per_gt.mean().item()
    
    # Precision: for each result, take best-matching GT
    best_per_result, _ = f1_matrix.max(dim=1)
    precision = best_per_result.mean().item()
    
    # F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return recall, precision, f1_score

# -------------------------------
# Main evaluation
# -------------------------------

# Load JSON files
with open(result, "r") as f:
    result_json = json.load(f)

with open(ground_truth, "r") as f:
    gt_json = json.load(f)

# Build lookup table for ground truth by filename
gt_map = {item["filename"]: item for item in gt_json}

# Store per-file results
file_scores = []

# Loop through each result image
for image in tqdm(result_json["results"][:5], desc="Evaluating images"):
    filename = image["filename"]
    gt_item = gt_map.get(filename)
    
    if gt_item is None:
        continue

    # Step 1: build combined texts
    result_texts = extract_combined_texts(image["artifacts"])
    gt_texts     = extract_combined_texts(gt_item["artifacts"])
    
    # Step 2: Global BERTScore (concatenate all combined texts)
    result_concat = " ".join(result_texts) if result_texts else ""
    gt_concat     = " ".join(gt_texts) if gt_texts else ""
    
    _, _, F1_global = score([result_concat], [gt_concat], lang="en", verbose=False)
    global_f1 = F1_global.item()
    
    # Step 3: Pairwise artifact-level BERTScore
    recall, precision, f1 = compute_best_match_scores(result_texts, gt_texts)
    
    # Store results for this file
    file_scores.append({
        "filename": filename,
        "global_f1": global_f1,
        "artifact_recall": recall,
        "artifact_precision": precision,
        "artifact_f1": f1,
        "num_result_artifacts": len(result_texts),
        "num_gt_artifacts": len(gt_texts)
    })

# -------------------------------
# Dataset-level summary
# -------------------------------
summary = {
    "global_f1_mean": float(np.mean([f["global_f1"] for f in file_scores])),
    "artifact_recall_mean": float(np.mean([f["artifact_recall"] for f in file_scores])),
    "artifact_precision_mean": float(np.mean([f["artifact_precision"] for f in file_scores])),
    "artifact_f1_mean": float(np.mean([f["artifact_f1"] for f in file_scores]))
}

# -------------------------------
# Output results
# -------------------------------
output = {
    "per_file": file_scores,
    "dataset_summary": summary
}

# -------------------------------
# Save output in same folder as results.json
# -------------------------------

results_dir = os.path.dirname(result)
output_path = os.path.join(results_dir, "bert_evaluation_results.json")

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Evaluation complete! Results saved to {output_path}")

