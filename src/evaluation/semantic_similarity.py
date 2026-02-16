# Authot: Claude Opus 4.6

import json
import os
import warnings
import numpy as np
from tqdm import tqdm
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

RESULT_JSON = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/results.json"

GT = "gt_2"
GT_JSON = f"/home/usluesyr/ai_image_detector/data/ground_truth/{GT}/json/{GT}.json"

MODEL_NAME = "all-mpnet-base-v2"
THRESHOLD = 0.6

OUTPUT_FILE = os.path.join(
    os.path.dirname(RESULT_JSON),
    f"semantic_similarity_evaluation_{GT}_{THRESHOLD}.json"
)


model = SentenceTransformer(MODEL_NAME)



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
            "mean_cosine_all": None,
            "mean_cosine_matched": None,
            "mean_cosine_valid": None,
            "matches": [],
            "artifact_type_acc": None,
            "unmatched_gt_artifacts": [],
            "unmatched_result_artifacts": []
        }

    # Edge case: one empty
    if not r_texts or not g_texts:
        matches = []
        unmatched_gt = gt_artifacts.copy()
        unmatched_results = result_artifacts.copy()
        # Don't add unmatched to matches list anymore
        return {
            "mean_cosine_all": 0.0,
            "mean_cosine_matched": 0.0,
            "mean_cosine_valid": 0.0,
            "matches": matches,
            "artifact_type_acc": None,
            "unmatched_gt_artifacts": unmatched_gt,
            "unmatched_result_artifacts": unmatched_results
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

        # CHANGED: Always mark as used regardless of threshold
        # Previously: only marked as used if score_val >= threshold
        # This caused artifacts below threshold to be treated as unmatched with 0.0 scores
        used_r.add(r_idx)
        used_g.add(g_idx)

        # Manual review mismatch flag
        manual_review = result_artifacts[r_idx].get("manual_review", "")
        is_invalid = False
        
        if isinstance(manual_review, str):
            review_lower = manual_review.lower()
            if "invalid" in review_lower: # or "new artifact" in review_lower and "valid" not in review_lower:
                is_invalid = True

        match_entry = {
            "result_artifact": result_artifacts[r_idx],
            "matched_gt_artifact": gt_artifacts[g_idx],
            "cosine_score": float(score_val),
            # CHANGED: is_match is true only if score >= threshold AND not manually flagged as invalid
            "valid_match": bool(score_val >= threshold)
        }

        # CHANGED: Only add mismatch flag if it's actually invalid
        if is_invalid and bool(score_val >= threshold):
            match_entry["mismatch"] = True

        matches.append(match_entry)

        # CHANGED: Only count artifact type accuracy for valid matches (above threshold and not invalid)
        # Previously: counted all Hungarian assignments
        if score_val >= threshold and not is_invalid:  # ADDED not is_invalid CONDITION
            r_type = result_artifacts[r_idx].get("type")
            g_type = gt_artifacts[g_idx].get("type")
            if r_type is not None and g_type is not None:
                type_scores.append(int(r_type == g_type))


    # Add unmatched GT artifacts (but NOT to matches list)
    unmatched_gt = []
    for j in range(len(gt_artifacts)):
        if j not in used_g:
            unmatched_gt.append(gt_artifacts[j])
    
    # Add unmatched result artifacts (hallucinations/extra detections)
    unmatched_results = []
    for i in range(len(result_artifacts)):
        if i not in used_r:
            unmatched_results.append(result_artifacts[i])

    # CHANGED: Cosine statistics calculation - Three distinct metrics
    # Note: matches now only contains actual Hungarian-assigned pairs (both result and GT exist)
    # Unmatched artifacts are in separate lists: unmatched_results and unmatched_gt
    
    # All matched pair scores (all have actual cosine values, no 0.0 from unmatched)
    all_scores = [m["cosine_score"] for m in matches]
    
    # Scores only for matches that pass the threshold (and not manually invalid)
    valid_match_scores = [
        m["cosine_score"] for m in matches 
        if m.get("valid_match", False)
    ]
    
    # For mean_cosine_all, we need to include unmatched artifacts as 0.0
    # Total pairs = matched pairs + unmatched results + unmatched GTs
    total_scores = all_scores + [0.0] * (len(unmatched_results) + len(unmatched_gt))

    artifact_type_acc = float(np.mean(type_scores)) if type_scores else None

    return {
        # 1. Mean of ALL pairs including unmatched (0.0)
        "mean_cosine_all": float(np.mean(total_scores)) if total_scores else 0.0,
        # 2. Mean of matched pairs only (excludes unmatched)
        "mean_cosine_matched": float(np.mean(all_scores)) if all_scores else 0.0,
        # 3. Mean of only valid matches (above threshold and not invalid)
        "mean_cosine_valid": float(np.mean(valid_match_scores)) if valid_match_scores else 0.0,
        "matches": matches,
        "artifact_type_acc": artifact_type_acc,
        "unmatched_gt_artifacts": unmatched_gt,
        "unmatched_result_artifacts": unmatched_results
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
        # 1. Mean of all pairs including unmatched (0.0)
        "mean_cosine_all": semantic_result["mean_cosine_all"],
        # 2. Mean of matched pairs only (excludes unmatched)
        "mean_cosine_matched": semantic_result["mean_cosine_matched"],
        # 3. Mean of valid matches only (above threshold)
        "mean_cosine_valid": semantic_result["mean_cosine_valid"],
        "num_result_artifacts": len(image["artifacts"]),
        "num_gt_artifacts": len(gt_item["artifacts"]),
        "matched_artifacts": semantic_result["matches"],
        "artifact_type_acc": semantic_result["artifact_type_acc"],
        "unmatched_gt_artifacts": semantic_result["unmatched_gt_artifacts"],
        "unmatched_result_artifacts": semantic_result["unmatched_result_artifacts"]
    })


# -----------------------------
# DATASET AVERAGE (ROBUST)
# -----------------------------
if file_scores:
    # CHANGED: Updated to calculate three distinct metrics across the dataset
    # 1. ALL pairs (including unmatched with 0.0)
    # 2. MATCHED pairs only (excludes unmatched, includes below-threshold)
    # 3. VALID matches only (above threshold)
    
    # === ALL FILES ===
    all_mean_all = [f["mean_cosine_all"] for f in file_scores if f["mean_cosine_all"] is not None]
    all_mean_matched = [f["mean_cosine_matched"] for f in file_scores if f["mean_cosine_matched"] is not None]
    all_mean_valid = [f["mean_cosine_valid"] for f in file_scores if f["mean_cosine_valid"] is not None]
    
    # === FAKE-CLASSIFIED FILES ONLY ===
    # 1. All pairs including unmatched
    fake_mean_all = [f["mean_cosine_all"] for f in file_scores if f["classification"] == "fake" and f["mean_cosine_all"] is not None]
    # 2. Matched pairs only
    fake_mean_matched = [f["mean_cosine_matched"] for f in file_scores if f["classification"] == "fake" and f["mean_cosine_matched"] is not None and f["mean_cosine_matched"] > 0.0]
    # 3. Valid matches only
    fake_mean_valid = [f["mean_cosine_valid"] for f in file_scores if f["classification"] == "fake" and f["mean_cosine_valid"] is not None and f["mean_cosine_valid"] > 0.0]
    
    # artifact type accuracy per image
    type_acc_values = [
        f["artifact_type_acc"]
        for f in file_scores
        if f["artifact_type_acc"] is not None
    ]

    # artifact type accuracy global weighted
    total_type_correct = 0
    total_type_eligible = 0

    for f in file_scores:
        for m in f["matched_artifacts"]:
            if m.get("valid_match") and not m.get("mismatch"):
                r_type = m["result_artifact"].get("type")
                g_type = m["matched_gt_artifact"].get("type")
                if r_type is not None and g_type is not None:
                    total_type_eligible += 1
                    if r_type == g_type:
                        total_type_correct += 1
    
    
    dataset_average = {
        # ALL FILES
        "all_files": {
            # 1. All pairs (including unmatched)
            "all_mean_cosine_all": float(np.mean(all_mean_all)) if all_mean_all else 0.0,
            "all_std_cosine_all": float(np.std(all_mean_all)) if all_mean_all else 0.0,
            "all_count_all": len(all_mean_all),
            
            # 2. Matched pairs only
            "all_mean_cosine_matched": float(np.mean(all_mean_matched)) if all_mean_matched else 0.0,
            "all_std_cosine_matched": float(np.std(all_mean_matched)) if all_mean_matched else 0.0,
            "all_count_matched": len(all_mean_matched),
            
            # 3. Valid matches only
            "all_mean_cosine_valid": float(np.mean(all_mean_valid)) if all_mean_valid else 0.0,
            "all_std_cosine_valid": float(np.std(all_mean_valid)) if all_mean_valid else 0.0,
            "all_count_valid": len(all_mean_valid),
        },
        # FAKE-CLASSIFIED FILES ONLY
        "fake_classified_only": {
            # 1. All pairs (including unmatched)
            "fake_mean_cosine_all": float(np.mean(fake_mean_all)) if fake_mean_all else 0.0,
            "fake_std_cosine_all": float(np.std(fake_mean_all)) if fake_mean_all else 0.0,
            "fake_count_all": len(fake_mean_all),
            
            # 2. Matched pairs only
            "fake_mean_cosine_matched": float(np.mean(fake_mean_matched)) if fake_mean_matched else 0.0,
            "fake_std_cosine_matched": float(np.std(fake_mean_matched)) if fake_mean_matched else 0.0,
            "fake_count_matched": len(fake_mean_matched),
            
            # 3. Valid matches only
            "fake_mean_cosine_valid": float(np.mean(fake_mean_valid)) if fake_mean_valid else 0.0,
            "fake_std_cosine_valid": float(np.std(fake_mean_valid)) if fake_mean_valid else 0.0,
            "fake_count_valid": len(fake_mean_valid),
        },

        "artifact_type_acc": {    
            "mean_artifact_type_acc_per_image": float(np.mean(type_acc_values)) if type_acc_values else None,
            "mean_artifact_type_acc_per_image_std": float(np.std(type_acc_values)) if type_acc_values else None,
            "mean_artifact_type_acc_per_image_count": len(type_acc_values),
            "mean_artifact_type_acc_global": total_type_correct / total_type_eligible if total_type_eligible > 0 else None,
            "mean_artifact_type_acc_global_count": total_type_eligible
        },


        "total_unmatched_gt_artifacts": sum(len(f["unmatched_gt_artifacts"]) for f in file_scores),
            
        "total_unmatched_result_artifacts": sum(len(f["unmatched_result_artifacts"]) for f in file_scores)
    }

    # Add to dataset_average computation
    total_gt = sum(f["num_gt_artifacts"] for f in file_scores)
    total_unmatched_gt = dataset_average["total_unmatched_gt_artifacts"]
    total_valid_matches = sum(
        sum(1 for m in f["matched_artifacts"] if m.get("valid_match") and not m.get("mismatch"))
        for f in file_scores
    )

    # Rename to reflect what it actually measures
    dataset_average["artifact_assignment_recall"] = (    # any Hungarian assignment, including below-threshold
        (total_gt - total_unmatched_gt) / total_gt if total_gt > 0 else 0.0
    )
    dataset_average["valid_match_recall"] = (            # only above-threshold, non-invalid matches
        total_valid_matches / total_gt if total_gt > 0 else 0.0
    )

    # ADDED: Total result artifacts for precision calculation
    total_result_artifacts = sum(f["num_result_artifacts"] for f in file_scores)

    # ADDED: Precision - fraction of result artifacts that are valid matches
    dataset_average["valid_match_precision"] = (
        total_valid_matches / total_result_artifacts if total_result_artifacts > 0 else 0.0
    )

    # ADDED: F1 score - harmonic mean of precision and recall
    _precision = dataset_average["valid_match_precision"]
    _recall = dataset_average["valid_match_recall"]
    dataset_average["valid_match_f1"] = (
        2 * (_precision * _recall) / (_precision + _recall)
        if (_precision + _recall) > 0 else 0.0
    )

    # ADDED: Count of valid matches for context (helps interpret mean_cosine_valid)
    dataset_average["total_valid_matches"] = total_valid_matches
    dataset_average["total_gt_artifacts"] = total_gt
    dataset_average["total_result_artifacts"] = total_result_artifacts

    # ADDED: Mismatches count - valid_match=True but manually flagged as invalid
    total_mismatches = sum(
        sum(1 for m in f["matched_artifacts"] if m.get("mismatch"))
        for f in file_scores
    )
    dataset_average["mismatches"] = total_mismatches

else:
    dataset_average = {
        # All files -> in case a model classified an image as real but still has valid artifacts
        "all_files": {
            "all_mean_cosine_all": 0.0,
            "all_std_cosine_all": 0.0,
            "all_mean_cosine_matched": 0.0,
            "all_std_cosine_matched": 0.0,
            "all_mean_cosine_valid": 0.0,
            "all_std_cosine_valid": 0.0,
        },
        # Fake-classified files -> onyl if the model classified the image as fake 
        "fake_classified_only": {
            "fake_mean_cosine_all": 0.0,
            "fake_std_cosine_all": 0.0,
            "fake_mean_cosine_matched": 0.0,
            "fake_std_cosine_matched": 0.0,
            "fake_mean_cosine_valid": 0.0,
            "fake_std_cosine_valid": 0.0,
        },
        "artifact_type_acc": {
            "mean_artifact_type_acc_per_image": None,
            "mean_artifact_type_acc_per_image_std": None,
            "mean_artifact_type_acc_per_image_count": 0,
            "mean_artifact_type_acc_global": None,
            "mean_artifact_type_acc_global_count": 0,
        },
        "total_unmatched_gt_artifacts": 0,
        "total_unmatched_result_artifacts": 0,
        "artifact_assignment_recall": 0.0,
        "valid_match_recall": 0.0,
        "valid_match_precision": 0.0,       # ADDED
        "valid_match_f1": 0.0,              # ADDED
        "total_valid_matches": 0,           # ADDED
        "total_gt_artifacts": 0,            # ADDED
        "total_result_artifacts": 0,        # ADDED
        "mismatches": 0                     # ADDED
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
    "detected_fake": result_json["detected_fake"],
    "detected_real": result_json["detected_real"],
    "acc": result_json["acc"],

    "image_scores": file_scores,
    "dataset_average": dataset_average
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Semantic evaluation complete! Results saved to {OUTPUT_FILE}")