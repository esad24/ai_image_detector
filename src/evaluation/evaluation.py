import json
import os
import warnings
import numpy as np
from tqdm import tqdm
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment

# -----------------------------
# SETTINGS
# -----------------------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

RESULT_JSON = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/results_final.json"

GT = "gt_2"
GT_JSON = f"/home/usluesyr/ai_image_detector/data/ground_truth/{GT}/json/{GT}.json"

MODEL_NAME = "all-mpnet-base-v2"
THRESHOLD = 0.5

OUTPUT_FILE = os.path.join(
    os.path.dirname(RESULT_JSON),
    f"semantic_evaluation_hungarian_{GT}_{THRESHOLD}.json"
)

model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# HELPER
# -----------------------------
def artifact_to_text(artifact):
    parts = []
    if artifact.get("reasoning"):
        parts.append(artifact["reasoning"])
    if artifact.get("location"):
        parts.append(str(artifact["location"]))
    return ". ".join(parts)


def compute_semantic_matches(result_artifacts, gt_artifacts, threshold=THRESHOLD):
    r_texts = [artifact_to_text(a) for a in result_artifacts]
    g_texts = [artifact_to_text(a) for a in gt_artifacts]

    # Edge case: both empty
    if not r_texts and not g_texts:
        return {
            "mean_cosine": None,          
            "max_cosine": None,           
            "min_cosine": None,           
            "mean_cosine_non_zero": None, 
            "max_cosine_non_zero": None,  
            "min_cosine_non_zero": None,  
            "matches": [],
            "conditional_type_acc": None,
            "unmatched_gt_artifacts": [],
        }

    # Edge case: one empty
    if not r_texts or not g_texts:
        matches = []
        for r in result_artifacts:
            matches.append({
                "result_artifact": r,
                "matched_gt_artifact": None,
                "cosine_score": 0.0
            })
        unmatched_gt = gt_artifacts.copy()
        for g in unmatched_gt:
            matches.append({
                "result_artifact": None,
                "matched_gt_artifact": g,
                "cosine_score": 0.0
            })
        return {
            "mean_cosine": 0.0,
            "max_cosine": 0.0,
            "min_cosine": 0.0,
            "mean_cosine_non_zero": 0.0,
            "max_cosine_non_zero": 0.0,
            "min_cosine_non_zero": 0.0,
            "matches": matches,
            "artifact_type_acc": None,
            "unmatched_gt_artifacts": unmatched_gt
        }

    # Encode embeddings
    r_emb = model.encode(r_texts, convert_to_tensor=True)
    g_emb = model.encode(g_texts, convert_to_tensor=True)

    cos_sim_matrix = util.cos_sim(r_emb, g_emb).cpu().numpy()
    cost_matrix = -cos_sim_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    used_r = set()
    used_g = set()
    type_scores = []

    # Hungarian matching
    for r_idx, g_idx in zip(row_ind, col_ind):
        score_val = cos_sim_matrix[r_idx, g_idx]

        if score_val >= threshold:
            used_r.add(r_idx)
            used_g.add(g_idx)

            match_entry = {
                "result_artifact": result_artifacts[r_idx],
                "matched_gt_artifact": gt_artifacts[g_idx],
                "cosine_score": float(score_val)
            }

            # Manual review mismatch flag
            manual_review = result_artifacts[r_idx].get("manual_review", "")

            if isinstance(manual_review, str):
                review_lower = manual_review.lower()
                if "invalid" in review_lower or "uncertain" in review_lower: #or "new artifact" in review_lower and "valid" not in review_lower:
                    match_entry["mismatch"] = True


            matches.append(match_entry)

            # Artifact type accuracy
            r_type = result_artifacts[r_idx].get("type")
            g_type = gt_artifacts[g_idx].get("type")
            if r_type is not None and g_type is not None:
                type_scores.append(int(r_type == g_type))


    # Add unmatched result artifacts
    for i in range(len(result_artifacts)):
        if i not in used_r:
            matches.append({
                "result_artifact": result_artifacts[i],
                "matched_gt_artifact": None,
                "cosine_score": 0.0
            })

    # Add unmatched GT artifacts
    unmatched_gt = []
    for j in range(len(gt_artifacts)):
        if j not in used_g:
            unmatched_gt.append(gt_artifacts[j])
            matches.append({
                "result_artifact": None,
                "matched_gt_artifact": gt_artifacts[j],
                "cosine_score": 0.0
            })

    # Cosine statistics
    all_scores = [m["cosine_score"] for m in matches]
    non_zero_scores = [s for s in all_scores if s > 0.0]

    artifact_type_acc = float(np.mean(type_scores)) if type_scores else None

    return {
        "mean_cosine": float(np.mean(all_scores)) if all_scores else 0.0,
        "mean_cosine_non_zero": float(np.mean(non_zero_scores)) if non_zero_scores else 0.0,
        "matches": matches,
        "artifact_type_acc": artifact_type_acc,
        "unmatched_gt_artifacts": unmatched_gt
    }


# -----------------------------
# LOAD JSONS
# -----------------------------
with open(RESULT_JSON, "r", encoding="utf-8") as f:
    result_json = json.load(f)

with open(GT_JSON, "r", encoding="utf-8") as f:
    gt_json = json.load(f)

gt_map = {item["filename"]: item for item in gt_json["results"]}

# -----------------------------
# EVALUATION
# -----------------------------
file_scores = []

for image in tqdm(result_json["results"], desc="Evaluating images"):
    filename = image["filename"]
    classification = image["classification"]
    gt_item = gt_map.get(filename)
    if gt_item is None:
        continue

    semantic_result = compute_semantic_matches(
        image["artifacts"],
        gt_item["artifacts"],
        threshold=THRESHOLD
    )

    file_scores.append({
        "filename": filename,
        "classification": classification,
        "mean_cosine_similarity": semantic_result["mean_cosine"],
        "mean_cosine_similarity_non_zero": semantic_result["mean_cosine_non_zero"],
        "num_result_artifacts": len(image["artifacts"]),
        "num_gt_artifacts": len(gt_item["artifacts"]),
        "matched_artifacts": semantic_result["matches"],
        "artifact_type_acc": semantic_result["artifact_type_acc"],
        "unmatched_gt_artifacts": semantic_result["unmatched_gt_artifacts"]
    })


# -----------------------------
# DATASET AVERAGE (ROBUST)
# -----------------------------
if file_scores:
    # 1. ALL FILES
    all_per_image_means = [f["mean_cosine_similarity"] for f in file_scores]
    
    # 2. FAKE-CLASSIFIED FILES ONLY
    fake_per_image_means = [
        f["mean_cosine_similarity"] 
        for f in file_scores 
        if f["classification"] == "fake"
    ]
    
    # 3. FAKE-CLASSIFIED FILES WITH NON-ZERO SCORES
    fake_nonzero_per_image_means = [
        f["mean_cosine_similarity_non_zero"]  # This already excludes 0.0 matches
        for f in file_scores 
        if f["classification"] == "fake" 
        and f["mean_cosine_similarity_non_zero"] > 0.0  # Extra safety check
    ]
    
    dataset_average = {
        # 1. All files
        "all_mean_cosine": float(np.mean(all_per_image_means)) if all_per_image_means else 0.0,
        "all_std_cosine": float(np.std(all_per_image_means)) if all_per_image_means else 0.0,
        "all_count": len(all_per_image_means),
        
        # 2. Fake-classified only
        "fake_mean_cosine": float(np.mean(fake_per_image_means)) if fake_per_image_means else 0.0,
        "fake_std_cosine": float(np.std(fake_per_image_means)) if fake_per_image_means else 0.0,
        "fake_count": len(fake_per_image_means),
        
        # 3. Fake with non-zero matches only
        "fake_nonzero_mean_cosine": float(np.mean(fake_nonzero_per_image_means)) if fake_nonzero_per_image_means else 0.0,
        "fake_nonzero_std_cosine": float(np.std(fake_nonzero_per_image_means)) if fake_nonzero_per_image_means else 0.0,
        "fake_nonzero_count": len(fake_nonzero_per_image_means),
    }

    type_acc_values = [
        f["artifact_type_acc"]
        for f in file_scores
        if f["artifact_type_acc"] is not None
    ]

    dataset_average["mean_artifact_type_acc"] = (
        float(np.mean(type_acc_values)) if type_acc_values else None
    )

    dataset_average["total_unmatched_gt_artifacts"] = sum(
        len(f["unmatched_gt_artifacts"]) for f in file_scores
    )

else:
    dataset_average = {
        "all_mean_cosine": 0.0,
        "all_std_cosine": 0.0,
        "fake_mean_cosine": 0.0,
        "fake_std_cosine": 0.0,
        "fake_nonzero_mean_cosine": 0.0,
        "fake_nonzero_std_cosine": 0.0,
        "mean_artifact_type_acc": None,
        "total_unmatched_gt_artifacts": 0
    }


# -----------------------------
# SAVE JSON
# -----------------------------
output = {
    "meta": result_json.get("meta", {}),
    "semantic_model": MODEL_NAME,
    "comparison": GT_JSON,
    "threshold": THRESHOLD,
    "image_count": result_json.get("image_count", len(result_json["results"])),
    "total_fakes": result_json.get("total_fakes", 0),
    "total_real": result_json.get("total_real", 0),
    "image_scores": file_scores,
    "dataset_average": dataset_average
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Semantic evaluation complete! Results saved to {OUTPUT_FILE}")
