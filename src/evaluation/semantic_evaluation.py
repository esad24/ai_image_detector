import json
from tqdm import tqdm
import torch
import numpy as np
import os
import warnings
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer, util

# Suppress warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# Paths
result = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/results.json"
ground_truth = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt4/2026-01-23_08-50-39/results.json"

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Threshold for matching
THRESHOLD = 0.5


# def artifact_to_text(artifact):
#     """Combine type, reasoning, and location for semantic comparison."""
#     parts = []
#     if "type" in artifact and artifact["type"]:
#         parts.append(f"type: {artifact['type']}")
#     if "reasoning" in artifact and artifact["reasoning"]:
#         parts.append(f"reasoning: {artifact['reasoning']}")
#     if "location" in artifact and artifact["location"]:
#         parts.append(f"location: {artifact['location']}")
#     return " | ".join(parts)

def artifact_to_text(artifact):
    parts = []
    if artifact.get("reasoning"):
        parts.append(artifact["reasoning"])
    if artifact.get("location"):
        parts.append(artifact["location"])
    return ". ".join(parts)


def compute_semantic_matches(result_artifacts, gt_artifacts, threshold=THRESHOLD):
    """Compute semantic similarity matches using SBERT."""
    r_texts = [artifact_to_text(a) for a in result_artifacts]
    g_texts = [artifact_to_text(a) for a in gt_artifacts]

    if not r_texts and not g_texts:
        return {
            "mean_cosine": 1.0,
            "max_cosine": 1.0,
            "min_cosine": 1.0,
            "matches": []
        }

    if not r_texts or not g_texts:
        matches = []
        for r in r_texts:
            matches.append({"result_artifact": r, "matched_gt_artifact": None, "cosine_score": 0.0})
        for g in g_texts:
            matches.append({"result_artifact": None, "matched_gt_artifact": g, "cosine_score": 0.0})
        return {
            "mean_cosine": 0.0,
            "max_cosine": 0.0,
            "min_cosine": 0.0,
            "matches": matches
        }

    # Encode embeddings
    r_emb = model.encode(r_texts, convert_to_tensor=True)
    g_emb = model.encode(g_texts, convert_to_tensor=True)

    # Compute cosine similarity matrix
    cos_sim_matrix = util.cos_sim(r_emb, g_emb)

    matches = []
    gt_matched_flags = [False] * len(gt_artifacts)

    for i, r_art in enumerate(result_artifacts):
        best_j = torch.argmax(cos_sim_matrix[i]).item()
        score_val = cos_sim_matrix[i, best_j].item()
        if score_val >= threshold:
            matches.append({
                "result_artifact": r_art,
                "matched_gt_artifact": gt_artifacts[best_j],
                "cosine_score": score_val
            })
            gt_matched_flags[best_j] = True
        else:
            matches.append({
                "result_artifact": r_art,
                "matched_gt_artifact": None,
                "cosine_score": 0.0,
            })

    # Add unmatched GT artifacts
    for j, matched in enumerate(gt_matched_flags):
        if not matched:
            matches.append({
                "result_artifact": None,
                "matched_gt_artifact": gt_artifacts[j],
                "cosine_score": 0.0
            })

    all_scores = [m["cosine_score"] for m in matches] #if m["cosine_score"] > 0]

    mean_cosine = float(np.mean(all_scores)) if all_scores else 0.0
    max_cosine = float(np.max(all_scores)) if all_scores else 0.0
    min_cosine = float(np.min(all_scores)) if all_scores else 0.0

    return {
        "mean_cosine": mean_cosine,
        "max_cosine": max_cosine,
        "min_cosine": min_cosine,
        "matches": matches
    }



# Load JSONs
with open(result, "r") as f:
    result_json = json.load(f)

with open(ground_truth, "r") as f:
    gt_json = json.load(f)

# Build lookup table for ground truth by filename
gt_map = {item["filename"]: item for item in gt_json["results"]}

file_scores = []

for image in tqdm(result_json["results"], desc="Evaluating images"):
    filename = image["filename"]
    classification = image["classification"]
    gt_item = gt_map.get(filename)
    if gt_item is None:
        continue

    # Compute semantic artifact matches
    semantic_result = compute_semantic_matches(
        image["artifacts"], gt_item["artifacts"], threshold=THRESHOLD
    )


    file_scores.append({
        "filename": filename,
        "classification": classification,
        "mean_cosine_similarity": semantic_result["mean_cosine"],
        "max_cosine_similarity": semantic_result["max_cosine"],
        "min_cosine_similarity": semantic_result["min_cosine"],
        "num_result_artifacts": len(image["artifacts"]),
        "num_gt_artifacts": len(gt_item["artifacts"]),
        "matched_artifacts": semantic_result["matches"]
    })


# Summary
summary = {
    "mean_cosine_similarity_avg": float(np.mean([f["mean_cosine_similarity"] for f in file_scores])),
    "max_cosine_similarity_avg": float(np.mean([f["max_cosine_similarity"] for f in file_scores])),
}


# Final output
output = {
    "meta": result_json.get("meta", {}),
    "threshold": THRESHOLD,
    "image_count": result_json.get("image_count", len(result_json["results"])),
    "total_fakes": result_json.get("total_fakes", 0),
    "total_real": result_json.get("total_real", 0),
    "image_scores": file_scores,
    "dataset_average": summary
}

# Save JSON
results_dir = os.path.dirname(result)
output_path = os.path.join(results_dir, "semantic_evaluation_gpt_prompt3_and_prompt4.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Semantic evaluation complete! Results saved to {output_path}")
