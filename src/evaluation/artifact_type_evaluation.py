"""
Evaluate predictions on fake or real image datasets.

Produces a standardized result JSON with consistent metric names,
compatible with the semantic similarity evaluation pipeline.

Modes:
  fake  - Compares predictions against ground truth.
          Classification metrics (TP, FN, recall).
          Artifact type detection (set-level per-type P/R/F1).
          Artifact counts & matching summary.

  real  - No ground truth needed (all images are truly real).
          Classification metrics (TN, FP, specificity).
          False positive analysis (hallucinated artifact type distribution).
"""

import json
import os

ALL_TYPES = ["structural", "semantic", "stylistic", "physical"]


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
MODE = "real"  # "fake" or "real"

GT = "gt_2"

GT_JSON = None
RESULT_JSON  = "/home/usluesyr/ai_image_detector/data/real2gen/real/results/qwen3-vl/prompt3/2026-02-21_13-15-29/results.json"
if MODE == "fake":
    OUTPUT_FILE = os.path.join(
        os.path.dirname(RESULT_JSON),
        f"artifact_type_evaluation_{GT}.json"
    )
else:
    OUTPUT_FILE = os.path.join(
        os.path.dirname(RESULT_JSON),
        f"artifact_type_evaluation.json"
    )




# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════


def print_header(title, printer):
    printer("=" * 60)
    printer(title)
    printer("=" * 60)

def redirect_print_to_file(output_path):
    """Redirects prints to a .txt file next to the JSON output."""
    txt_output_path = os.path.splitext(output_path)[0] + ".txt"
    f = open(txt_output_path, "w", encoding="utf-8")
    def printer(*args, **kwargs):
        print(*args, **kwargs, file=f)
    return f, printer, txt_output_path



def get_type_set(artifacts):
    """Extract the unique set of artifact types from a list of artifacts."""
    return set(a["type"] for a in artifacts)


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def compute_per_type_metrics(type_tp, type_fp, type_fn):
    """Compute precision, recall, F1 for each artifact type."""
    per_type = {}
    for t in ALL_TYPES:
        prec = safe_div(type_tp[t], type_tp[t] + type_fp[t])
        rec  = safe_div(type_tp[t], type_tp[t] + type_fn[t])
        f1   = safe_div(2 * prec * rec, prec + rec)
        per_type[t] = {
            "true_positives":  type_tp[t],
            "false_positives": type_fp[t],
            "false_negatives": type_fn[t],
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        }
    return per_type

# ═══════════════════════════════════════════════════════════════
# FAKE MODE
# ═══════════════════════════════════════════════════════════════
def evaluate_fake(predictions, ground_truth, output_path):
    gt_lookup = {r["filename"]: r for r in ground_truth["results"]}
    total = len(predictions["results"])

    # ── Classification ──
    tp = 0
    fn = 0
    fn_files = []

    for p in predictions["results"]:
        if p["classification"] == "fake":
            tp += 1
        else:
            fn += 1
            fn_files.append(p["filename"])

    classification_metrics = {
        "total_images":        total,
        "true_positives":      tp,
        "false_negatives":     fn,
        "recall_sensitivity":  round(safe_div(tp, total), 4),
        "false_negative_rate": round(safe_div(fn, total), 4),
        "false_negative_files": fn_files,
    }

    # ── Artifact type metrics (set-level per image) ──
    type_tp = {t: 0 for t in ALL_TYPES}
    type_fp = {t: 0 for t in ALL_TYPES}
    type_fn = {t: 0 for t in ALL_TYPES}
    exact_matches = 0
    total_gt_artifacts   = 0
    total_pred_artifacts = 0
    per_image_results = []

    for p in predictions["results"]:
        fname = p["filename"]
        gt = gt_lookup[fname]
        gt_types   = get_type_set(gt["artifacts"])
        pred_types = get_type_set(p["artifacts"]) if p["classification"] == "fake" else set()

        matched      = gt_types & pred_types
        hallucinated = pred_types - gt_types
        missed       = gt_types - pred_types

        if gt_types == pred_types:
            exact_matches += 1

        for t in ALL_TYPES:
            if t in gt_types and t in pred_types:
                type_tp[t] += 1
            if t in pred_types and t not in gt_types:
                type_fp[t] += 1
            if t in gt_types and t not in pred_types:
                type_fn[t] += 1

        total_gt_artifacts   += len(gt["artifacts"])
        total_pred_artifacts += len(p["artifacts"])

        per_image_results.append({
            "filename":                  fname,
            "predicted_classification":  p["classification"],
            "ground_truth_types":        sorted(gt_types),
            "predicted_types":           sorted(pred_types),
            "matched_types":             sorted(matched),
            "hallucinated_types":        sorted(hallucinated),
            "missed_types":              sorted(missed),
            "num_gt_artifacts":          len(gt["artifacts"]),
            "num_pred_artifacts":        len(p["artifacts"]),
        })

    per_type = compute_per_type_metrics(type_tp, type_fp, type_fn)

    total_matched_types = sum(type_tp.values())
    total_hallucinated  = sum(type_fp.values())
    total_missed        = sum(type_fn.values())

    artifact_type_metrics = {
        "note": "Set-level comparison: unique artifact types per image. "
                "FN images (classified as real) have an empty predicted set, "
                "so all their GT types count as missed.",
        "exact_type_set_matches":            exact_matches,
        "exact_type_set_match_rate":         round(safe_div(exact_matches, total), 4),
        "total_gt_artifacts":                total_gt_artifacts,
        "total_pred_artifacts":              total_pred_artifacts,
        "avg_gt_artifacts_per_image":        round(safe_div(total_gt_artifacts, total), 2),
        "avg_pred_artifacts_per_image":      round(safe_div(total_pred_artifacts, total), 2),
        "total_matched_type_instances":      total_matched_types,
        "total_hallucinated_type_instances": total_hallucinated,
        "total_missed_type_instances":       total_missed,
        "per_type": per_type,
    }

    # ── Assemble output ──
    result = {
        "meta": {
            "model":       predictions["meta"]["model"],
            "prompt_id":   predictions["meta"]["prompt_id"],
            "reasoning":   predictions["meta"]["reasoning"],
            "split":       "fake",
            "total_images": total,
        },
        "comparison": GT_JSON,
        "classification_metrics": classification_metrics,
        "artifact_type_metrics":  artifact_type_metrics,
        "per_image_results":      per_image_results,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


    f, printer, txt_output_path = redirect_print_to_file(output_path)


    # ── Print summary to TXT ──
    print_header("FAKE IMAGES EVALUATION", printer)
    printer(f"\nClassification:")
    printer(f"  TP (correctly detected fake): {tp}")
    printer(f"  FN (missed fakes):            {fn}")
    printer(f"  Recall / Sensitivity:         {classification_metrics['recall_sensitivity']}")
    printer(f"  False Negative Rate:          {classification_metrics['false_negative_rate']}")

    printer(f"\nArtifact Type Detection (set-level):")
    printer(f"  Exact type-set matches: {exact_matches}/{total} "
            f"({artifact_type_metrics['exact_type_set_match_rate']*100:.1f}%)")
    printer(f"  Matched / Hallucinated / Missed type instances: "
            f"{total_matched_types} / {total_hallucinated} / {total_missed}")

    printer(f"\n  {'Type':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    printer(f"  {'-'*44}")
    for t in ALL_TYPES:
        d = per_type[t]
        printer(f"  {t:<12} {d['true_positives']:>4} {d['false_positives']:>4} "
                f"{d['false_negatives']:>4} {d['precision']:>6.2f} {d['recall']:>6.2f} {d['f1']:>6.2f}")

    printer(f"\n  Avg GT artifacts/image:   {artifact_type_metrics['avg_gt_artifacts_per_image']}")
    printer(f"  Avg pred artifacts/image: {artifact_type_metrics['avg_pred_artifacts_per_image']}")
    print(f"\nWrote {output_path}")
    print(f"TXT report written to {txt_output_path}")

    f.close()



def evaluate_fake_no_gt(predictions, output_path):
    total = len(predictions["results"])

    tp = 0
    fn = 0
    fn_files = []

    for p in predictions["results"]:
        if p["classification"] == "fake":
            tp += 1
        else:
            fn += 1
            fn_files.append(p["filename"])

    classification_metrics = {
        "total_images":        total,
        "true_positives":      tp,
        "false_negatives":     fn,
        "recall_sensitivity":  round(safe_div(tp, total), 4),
        "false_negative_rate": round(safe_div(fn, total), 4),
        "false_negative_files": fn_files,
    }

    # ── Detected artifact type distribution (no GT to compare against) ──
    detected_type_distribution = {t: 0 for t in ALL_TYPES}
    total_artifacts = 0
    per_image_results = []

    for p in predictions["results"]:
        if p["classification"] == "fake":
            types = get_type_set(p["artifacts"])
            num_artifacts = len(p["artifacts"])
            total_artifacts += num_artifacts
            for t in types:
                detected_type_distribution[t] += 1
            per_image_results.append({
                "filename":                 p["filename"],
                "predicted_classification": "fake",
                "correct":                  True,
                "detected_types":           sorted(types),
                "num_detected_artifacts":   num_artifacts,
            })
        else:
            per_image_results.append({
                "filename":                 p["filename"],
                "predicted_classification": "real",
                "correct":                  False,
                "detected_types":           [],
            })

    detected_artifact_analysis = {
        "note": "No ground truth available. Type distribution reflects predictions only.",
        "total_detected_artifacts":      total_artifacts,
        "avg_artifacts_per_tp":          round(safe_div(total_artifacts, tp), 2),
        "detected_type_distribution":    detected_type_distribution,
    }

    result = {
        "meta": {
            "model":        predictions["meta"]["model"],
            "prompt_id":    predictions["meta"]["prompt_id"],
            "reasoning":    predictions["meta"]["reasoning"],
            "split":        "fake_no_gt",
            "total_images": total,
        },
        "comparison": None,
        "classification_metrics":    classification_metrics,
        "detected_artifact_analysis": detected_artifact_analysis,
        "per_image_results":          per_image_results,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    f, printer, txt_output_path = redirect_print_to_file(output_path)

    print_header("FAKE IMAGES EVALUATION (NO GROUND TRUTH)", printer)
    printer(f"\nClassification:")
    printer(f"  TP (correctly detected fake): {tp}")
    printer(f"  FN (missed fakes):            {fn}")
    printer(f"  Recall / Sensitivity:         {classification_metrics['recall_sensitivity']}")
    printer(f"  False Negative Rate:          {classification_metrics['false_negative_rate']}")

    printer(f"\nDetected Artifact Type Distribution (TPs only):")
    printer(f"  Total detected artifacts: {total_artifacts}")
    printer(f"  Avg per TP:               {detected_artifact_analysis['avg_artifacts_per_tp']}")
    for t in ALL_TYPES:
        count = detected_type_distribution[t]
        pct = safe_div(count, tp) * 100
        printer(f"    {t:<12} {count:>3}  ({pct:.1f}% of TPs contain this type)")

    print(f"\nWrote {output_path}")
    print(f"TXT report written to {txt_output_path}")

    f.close()




# ═══════════════════════════════════════════════════════════════
# REAL MODE
# ═══════════════════════════════════════════════════════════════
def evaluate_real(predictions, output_path):
    total = len(predictions["results"])

    # ── Classification ──
    tn = 0
    fp = 0
    fp_files = []

    for p in predictions["results"]:
        if p["classification"] == "real":
            tn += 1
        else:
            fp += 1
            fp_files.append(p["filename"])

    classification_metrics = {
        "total_images":                   total,
        "true_negatives":                 tn,
        "false_positives":                fp,
        "specificity_true_negative_rate": round(safe_div(tn, total), 4),
        "false_positive_rate":            round(safe_div(fp, total), 4),
        "false_positive_files":           fp_files,
    }

    # ── False positive analysis ──
    fp_type_distribution = {t: 0 for t in ALL_TYPES}
    fp_total_artifacts = 0
    per_image_results = []

    for p in predictions["results"]:
        if p["classification"] == "real":
            per_image_results.append({
                "filename":                 p["filename"],
                "predicted_classification": "real",
                "correct":                  True,
                "hallucinated_types":       [],
            })
        else:
            types = get_type_set(p["artifacts"])
            num_artifacts = len(p["artifacts"])
            fp_total_artifacts += num_artifacts
            for t in types:
                fp_type_distribution[t] += 1
            per_image_results.append({
                "filename":                  p["filename"],
                "predicted_classification":  "fake",
                "correct":                   False,
                "hallucinated_types":        sorted(types),
                "num_hallucinated_artifacts": num_artifacts,
            })

    false_positive_analysis = {
        "total_false_positives":             fp,
        "total_hallucinated_artifacts":      fp_total_artifacts,
        "avg_hallucinated_artifacts_per_fp": round(safe_div(fp_total_artifacts, fp), 2),
        "hallucinated_type_distribution":    fp_type_distribution,
    }

    # ── Assemble output ──
    result = {
        "meta": {
            "model":       predictions["meta"]["model"],
            "prompt_id":   predictions["meta"]["prompt_id"],
            "reasoning":   predictions["meta"]["reasoning"],
            "split":       "real",
            "total_images": total,
        },
        "comparison": GT_JSON,
        "classification_metrics":  classification_metrics,
        "false_positive_analysis": false_positive_analysis,
        "per_image_results":       per_image_results,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    f, printer, txt_output_path = redirect_print_to_file(output_path)


    print_header("REAL IMAGES EVALUATION", printer)
    printer(f"\nClassification:")
    printer(f"  TN (correctly identified real): {tn}")
    printer(f"  FP (real wrongly called fake):  {fp}")
    printer(f"  Specificity / TNR:              {classification_metrics['specificity_true_negative_rate']}")
    printer(f"  False Positive Rate:            {classification_metrics['false_positive_rate']}")

    printer(f"\nFalse Positive Analysis:")
    printer(f"  Total hallucinated artifacts: {fp_total_artifacts}")
    printer(f"  Avg per false positive:       {false_positive_analysis['avg_hallucinated_artifacts_per_fp']}")
    printer(f"\n  Hallucinated type distribution:")
    for t in ALL_TYPES:
        count = fp_type_distribution[t]
        pct = safe_div(count, fp) * 100
        printer(f"    {t:<12} {count:>3}  ({pct:.1f}% of FPs contain this type)")

    print(f"\nWrote {output_path}")
    print(f"TXT report written to {txt_output_path}")

    f.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    with open(RESULT_JSON) as f:
        predictions = json.load(f)

    if MODE == "fake":
        if GT_JSON is not None:
            with open(GT_JSON) as f:
                ground_truth = json.load(f)
            evaluate_fake(predictions, ground_truth, OUTPUT_FILE)
        else:
            evaluate_fake_no_gt(predictions, OUTPUT_FILE)
    elif MODE == "real":
        evaluate_real(predictions, OUTPUT_FILE)
    else:
        print(f"Error: MODE must be 'fake' or 'real', got '{MODE}'")


if __name__ == "__main__":
    main()