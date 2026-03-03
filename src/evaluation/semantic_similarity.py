"""
Semantic similarity evaluation for AI-generated image detection.

Uses sentence embeddings + Hungarian matching to compare predicted artifacts
against ground truth at the individual artifact level. Requires predictions
with reasoning text (not just type labels).

For real image evaluation (no ground truth), use evaluate.py instead.
"""

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



# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
RESULT_JSON = "/home/usluesyr/ai_image_detector/data/genArtifact/fake/test/results/qwen3-vl/prompt3/2026-01-22_21-28-42/results_manual_review.json"
GT = "gt_1"
GT_JSON = f"/home/usluesyr/ai_image_detector/data/genArtifact/ground_truth/{GT}/json/{GT}.json"

MODEL_NAME = "all-mpnet-base-v2"
THRESHOLD = 0.6

OUTPUT_FILE = os.path.join(
    os.path.dirname(RESULT_JSON),
    f"semantic_similarity_evaluation_{GT}_{THRESHOLD}.json"
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def safe_div(a, b):
    return a / b if b > 0 else 0.0


def print_header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def artifact_to_text(artifact):
    parts = []
    if artifact.get("reasoning"):
        parts.append(artifact["reasoning"])
    if artifact.get("location"):
        parts.append(str(artifact["location"]))
    return ". ".join(parts)


# ═══════════════════════════════════════════════════════════════
# SEMANTIC MATCHING (used in fake mode only)
# ═══════════════════════════════════════════════════════════════
def compute_semantic_matches(model, result_artifacts, gt_artifacts, threshold):
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
            "unmatched_result_artifacts": [],
        }

    # Edge case: one side empty
    if not r_texts or not g_texts:
        return {
            "mean_cosine_all": 0.0,
            "mean_cosine_matched": 0.0,
            "mean_cosine_valid": 0.0,
            "matches": [],
            "artifact_type_acc": None,
            "unmatched_gt_artifacts": gt_artifacts.copy(),
            "unmatched_result_artifacts": result_artifacts.copy(),
        }

    # Encode and compute similarity
    r_emb = model.encode(r_texts, convert_to_tensor=True)
    g_emb = model.encode(g_texts, convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(r_emb, g_emb).cpu().numpy()

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix)

    matches = []
    used_r = set()
    used_g = set()
    type_scores = []

    for r_idx, g_idx in zip(row_ind, col_ind):
        score_val = cos_sim_matrix[r_idx, g_idx]
        used_r.add(r_idx)
        used_g.add(g_idx)

        #Manual review mismatch flag
        manual_review = result_artifacts[r_idx].get("manual_review", "")
        is_invalid = False
        if isinstance(manual_review, str) and "invalid" in manual_review.lower():
            is_invalid = True

        match_entry = {
            "result_artifact": result_artifacts[r_idx],
            "matched_gt_artifact": gt_artifacts[g_idx],
            "cosine_score": float(score_val),
            "valid_match": bool(score_val >= threshold),
        }

        if is_invalid and score_val >= threshold:
            match_entry["mismatch"] = True

        matches.append(match_entry)

        # Type accuracy only on valid, non-invalid matches
        if score_val >= threshold and not is_invalid:
            r_type = result_artifacts[r_idx].get("type")
            g_type = gt_artifacts[g_idx].get("type")
            if r_type is not None and g_type is not None:
                type_scores.append(int(r_type == g_type))

    # Unmatched artifacts
    unmatched_gt = [gt_artifacts[j] for j in range(len(gt_artifacts)) if j not in used_g]
    unmatched_results = [result_artifacts[i] for i in range(len(result_artifacts)) if i not in used_r]

    # Cosine statistics
    all_scores = [m["cosine_score"] for m in matches]
    valid_scores = [m["cosine_score"] for m in matches if m.get("valid_match")]
    total_scores = all_scores + [0.0] * (len(unmatched_results) + len(unmatched_gt))

    return {
        "mean_cosine_all": float(np.mean(total_scores)) if total_scores else 0.0,
        "mean_cosine_matched": float(np.mean(all_scores)) if all_scores else 0.0,
        "mean_cosine_valid": float(np.mean(valid_scores)) if valid_scores else 0.0,
        "matches": matches,
        "artifact_type_acc": float(np.mean(type_scores)) if type_scores else None,
        "unmatched_gt_artifacts": unmatched_gt,
        "unmatched_result_artifacts": unmatched_results,
    }


# ═══════════════════════════════════════════════════════════════
# FAKE MODE
# ═══════════════════════════════════════════════════════════════
def evaluate_fake(result_json, gt_json, model, threshold, output_path):
    gt_map = {item["filename"]: item for item in gt_json["results"]}

    # Detect whether manual_review annotations are present in the data
    has_manual_review = any(
        "manual_review" in a
        for image in result_json["results"]
        for a in image.get("artifacts", [])
    )

    # ── Classification ──
    tp = 0
    fn = 0
    fn_files = []

    for image in result_json["results"]:
        if image["classification"] == "fake":
            tp += 1
        else:
            fn += 1
            fn_files.append(image["filename"])

    total = len(result_json["results"])
    classification_metrics = {
        "total_images":        total,
        "true_positives":      tp,
        "false_negatives":     fn,
        "recall_sensitivity":  round(safe_div(tp, total), 4),
        "false_negative_rate": round(safe_div(fn, total), 4),
        "false_negative_files": fn_files,
    }

    # ── Per-image semantic evaluation ──
    image_scores = []

    for image in tqdm(result_json["results"], desc="Evaluating images"):
        filename = image["filename"]
        gt_item = gt_map.get(filename)
        if gt_item is None:
            continue

        sem = compute_semantic_matches(model, image["artifacts"], gt_item["artifacts"], threshold)

        image_scores.append({
            "filename":                    filename,
            "classification":              image["classification"],
            "mean_cosine_all":             sem["mean_cosine_all"],
            "mean_cosine_matched":         sem["mean_cosine_matched"],
            "mean_cosine_valid":           sem["mean_cosine_valid"],
            "num_result_artifacts":        len(image["artifacts"]),
            "num_gt_artifacts":            len(gt_item["artifacts"]),
            "matched_artifacts":           sem["matches"],
            "artifact_type_acc":           sem["artifact_type_acc"],
            "unmatched_gt_artifacts":      sem["unmatched_gt_artifacts"],
            "unmatched_result_artifacts":  sem["unmatched_result_artifacts"],
        })

    # ── Dataset averages ──
    if image_scores:
        # Cosine averages - all files
        all_mean_all     = [f["mean_cosine_all"] for f in image_scores if f["mean_cosine_all"] is not None]
        all_mean_matched = [f["mean_cosine_matched"] for f in image_scores if f["mean_cosine_matched"] is not None]
        all_mean_valid   = [f["mean_cosine_valid"] for f in image_scores if f["mean_cosine_valid"] is not None]

        # Cosine averages - fake-classified only
        fake_mean_all     = [f["mean_cosine_all"] for f in image_scores if f["classification"] == "fake" and f["mean_cosine_all"] is not None]
        fake_mean_matched = [f["mean_cosine_matched"] for f in image_scores if f["classification"] == "fake" and f["mean_cosine_matched"] is not None and f["mean_cosine_matched"] > 0.0]
        fake_mean_valid   = [f["mean_cosine_valid"] for f in image_scores if f["classification"] == "fake" and f["mean_cosine_valid"] is not None and f["mean_cosine_valid"] > 0.0]

        # Artifact type accuracy - per image
        type_acc_values = [f["artifact_type_acc"] for f in image_scores if f["artifact_type_acc"] is not None]

        # Artifact type accuracy - global weighted
        total_type_correct = 0
        total_type_eligible = 0
        for f in image_scores:
            for m in f["matched_artifacts"]:
                if m.get("valid_match") and not m.get("mismatch"):
                    r_type = m["result_artifact"].get("type")
                    g_type = m["matched_gt_artifact"].get("type")
                    if r_type is not None and g_type is not None:
                        total_type_eligible += 1
                        if r_type == g_type:
                            total_type_correct += 1

        # Counts
        total_gt              = sum(f["num_gt_artifacts"] for f in image_scores)
        total_result          = sum(f["num_result_artifacts"] for f in image_scores)
        total_unmatched_gt    = sum(len(f["unmatched_gt_artifacts"]) for f in image_scores)
        total_unmatched_res   = sum(len(f["unmatched_result_artifacts"]) for f in image_scores)
        total_valid_matches   = sum(
            sum(1 for m in f["matched_artifacts"] if m.get("valid_match") and not m.get("mismatch"))
            for f in image_scores
        )
        total_mismatches      = sum(
            sum(1 for m in f["matched_artifacts"] if m.get("mismatch"))
            for f in image_scores
        ) if has_manual_review else "unknown"

        # P / R / F1 on valid matches
        _precision = safe_div(total_valid_matches, total_result)
        _recall    = safe_div(total_valid_matches, total_gt)
        _f1        = safe_div(2 * _precision * _recall, _precision + _recall)

        dataset_average = {
            "all_files": {
                "all_mean_cosine_all":     float(np.mean(all_mean_all)) if all_mean_all else 0.0,
                "all_std_cosine_all":      float(np.std(all_mean_all)) if all_mean_all else 0.0,
                "all_count_all":           len(all_mean_all),
                "all_mean_cosine_matched": float(np.mean(all_mean_matched)) if all_mean_matched else 0.0,
                "all_std_cosine_matched":  float(np.std(all_mean_matched)) if all_mean_matched else 0.0,
                "all_count_matched":       len(all_mean_matched),
                "all_mean_cosine_valid":   float(np.mean(all_mean_valid)) if all_mean_valid else 0.0,
                "all_std_cosine_valid":    float(np.std(all_mean_valid)) if all_mean_valid else 0.0,
                "all_count_valid":         len(all_mean_valid),
            },
            "fake_classified_only": {
                "fake_mean_cosine_all":     float(np.mean(fake_mean_all)) if fake_mean_all else 0.0,
                "fake_std_cosine_all":      float(np.std(fake_mean_all)) if fake_mean_all else 0.0,
                "fake_count_all":           len(fake_mean_all),
                "fake_mean_cosine_matched": float(np.mean(fake_mean_matched)) if fake_mean_matched else 0.0,
                "fake_std_cosine_matched":  float(np.std(fake_mean_matched)) if fake_mean_matched else 0.0,
                "fake_count_matched":       len(fake_mean_matched),
                "fake_mean_cosine_valid":   float(np.mean(fake_mean_valid)) if fake_mean_valid else 0.0,
                "fake_std_cosine_valid":    float(np.std(fake_mean_valid)) if fake_mean_valid else 0.0,
                "fake_count_valid":         len(fake_mean_valid),
            },
            "artifact_type_acc": {
                "mean_artifact_type_acc_per_image":       float(np.mean(type_acc_values)) if type_acc_values else None,
                "mean_artifact_type_acc_per_image_std":   float(np.std(type_acc_values)) if type_acc_values else None,
                "mean_artifact_type_acc_per_image_count": len(type_acc_values),
                "mean_artifact_type_acc_global":          safe_div(total_type_correct, total_type_eligible) if total_type_eligible > 0 else None,
                "mean_artifact_type_acc_global_count":    total_type_eligible,
            },
            "total_gt_artifacts":              total_gt,
            "total_result_artifacts":          total_result,
            "total_valid_matches":             total_valid_matches,
            "total_unmatched_gt_artifacts":    total_unmatched_gt,
            "total_unmatched_result_artifacts": total_unmatched_res,
            "artifact_assignment_recall":      safe_div(total_gt - total_unmatched_gt, total_gt),
            "valid_match_recall":              _recall,
            "valid_match_precision":           _precision,
            "valid_match_f1":                  _f1,
            "mismatches":                      total_mismatches,
        }
    else:
        dataset_average = _empty_dataset_average()

    # ── Assemble output ──
    output = {
        "meta":            result_json.get("meta", {}),
        "semantic_model":  MODEL_NAME,
        "comparison":      GT_JSON,
        "threshold":       threshold,
        "split":           "fake",
        "image_count":     result_json.get("image_count", len(result_json["results"])),
        "detected_fake":   result_json["detected_fake"],
        "detected_real":   result_json["detected_real"],
        "acc":             result_json["acc"],
        "classification_metrics": classification_metrics,
        "dataset_average": dataset_average,
        "image_scores":    image_scores,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    txt_output_path = os.path.splitext(output_path)[0] + ".txt"

    # Redirect print to TXT file
    with open(txt_output_path, "w", encoding="utf-8") as f:
        def printer(*args, **kwargs):
            print(*args, **kwargs, file=f)

        # ── Print report ──
        printer("="*60)
        printer("FAKE IMAGES — SEMANTIC SIMILARITY EVALUATION")
        printer("="*60)

        printer(f"\nClassification:")
        printer(f"  TP (correctly detected fake): {tp}")
        printer(f"  FN (missed fakes):            {fn}")
        printer(f"  Recall / Sensitivity:         {classification_metrics['recall_sensitivity']}")
        printer(f"  False Negative Rate:          {classification_metrics['false_negative_rate']}")

        printer(f"\nSemantic Matching (threshold={threshold}):")
        printer(f"  Valid matches:    {dataset_average['total_valid_matches']}")
        printer(f"  Unmatched GT:     {dataset_average['total_unmatched_gt_artifacts']}")
        printer(f"  Unmatched pred:   {dataset_average['total_unmatched_result_artifacts']}")
        printer(f"  Mismatches:       {dataset_average['mismatches']}" if dataset_average['mismatches'] is not None else "  Mismatches:       N/A (no manual_review in data)")
        printer(f"  Precision:        {dataset_average['valid_match_precision']:.4f}")
        printer(f"  Recall:           {dataset_average['valid_match_recall']:.4f}")
        printer(f"  F1:               {dataset_average['valid_match_f1']:.4f}")

        printer(f"\nCosine Similarity (all files):")
        af = dataset_average["all_files"]
        printer(f"  Mean (all pairs):     {af['all_mean_cosine_all']:.4f} ± {af['all_std_cosine_all']:.4f}")
        printer(f"  Mean (matched only):  {af['all_mean_cosine_matched']:.4f} ± {af['all_std_cosine_matched']:.4f}")
        printer(f"  Mean (valid only):    {af['all_mean_cosine_valid']:.4f} ± {af['all_std_cosine_valid']:.4f}")

        printer(f"\nCosine Similarity (fake-classified only):")
        ff = dataset_average["fake_classified_only"]
        printer(f"  Mean (all pairs):     {ff['fake_mean_cosine_all']:.4f} ± {ff['fake_std_cosine_all']:.4f}")
        printer(f"  Mean (matched only):  {ff['fake_mean_cosine_matched']:.4f} ± {ff['fake_std_cosine_matched']:.4f}")
        printer(f"  Mean (valid only):    {ff['fake_mean_cosine_valid']:.4f} ± {ff['fake_std_cosine_valid']:.4f}")

        printer(f"\nArtifact Type Accuracy (on valid matches):")
        ta = dataset_average["artifact_type_acc"]
        per_img = ta["mean_artifact_type_acc_per_image"]
        per_img_std = ta["mean_artifact_type_acc_per_image_std"]
        glob = ta["mean_artifact_type_acc_global"]
        printer(f"  Per-image mean:  {per_img:.4f} ± {per_img_std:.4f} (n={ta['mean_artifact_type_acc_per_image_count']})" if per_img is not None else "  Per-image mean:  N/A")
        printer(f"  Global weighted: {glob:.4f} (n={ta['mean_artifact_type_acc_global_count']})" if glob is not None else "  Global weighted: N/A")

        print(f"\nWrote {output_path}")
        print(f"TXT report written to {txt_output_path}")


# ═══════════════════════════════════════════════════════════════
# EMPTY DEFAULTS
# ═══════════════════════════════════════════════════════════════
def _empty_dataset_average():
    return {
        "all_files": {
            "all_mean_cosine_all": 0.0, "all_std_cosine_all": 0.0, "all_count_all": 0,
            "all_mean_cosine_matched": 0.0, "all_std_cosine_matched": 0.0, "all_count_matched": 0,
            "all_mean_cosine_valid": 0.0, "all_std_cosine_valid": 0.0, "all_count_valid": 0,
        },
        "fake_classified_only": {
            "fake_mean_cosine_all": 0.0, "fake_std_cosine_all": 0.0, "fake_count_all": 0,
            "fake_mean_cosine_matched": 0.0, "fake_std_cosine_matched": 0.0, "fake_count_matched": 0,
            "fake_mean_cosine_valid": 0.0, "fake_std_cosine_valid": 0.0, "fake_count_valid": 0,
        },
        "artifact_type_acc": {
            "mean_artifact_type_acc_per_image": None, "mean_artifact_type_acc_per_image_std": None,
            "mean_artifact_type_acc_per_image_count": 0, "mean_artifact_type_acc_global": None,
            "mean_artifact_type_acc_global_count": 0,
        },
        "total_gt_artifacts": 0, "total_result_artifacts": 0, "total_valid_matches": 0,
        "total_unmatched_gt_artifacts": 0, "total_unmatched_result_artifacts": 0,
        "artifact_assignment_recall": 0.0, "valid_match_recall": 0.0,
        "valid_match_precision": 0.0, "valid_match_f1": 0.0, "mismatches": None,
    }

# ═══════════════════════════════════════════════════════════════
# Testing 
# ═══════════════════════════════════════════════════════════════
def test_two_pairs():
    model = SentenceTransformer("all-mpnet-base-v2")

    text_a = "The left eye has a realistic natural reflection."
    text_b = "The left eye has an unrealistic artificial reflection."

    emb = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1])

    print(float(score))



# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        result_json = json.load(f)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_json = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    evaluate_fake(result_json, gt_json, model, THRESHOLD, OUTPUT_FILE)

if __name__ == "__main__":
    main()
    #test_two_pairs()