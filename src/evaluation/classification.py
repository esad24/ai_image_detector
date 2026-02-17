"""
Classification-only evaluation for AI-generated image detection.

For result JSONs that only contain classification labels (no artifact types,
no reasoning). Computes classification metrics only.

Modes:
  fake  - All images are truly fake. Measures TP, FN, recall.
  real  - All images are truly real. Measures TN, FP, specificity.
"""

import json
import os


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
MODE = "fake"  # "fake" or "real"

RESULT_JSON = "/home/usluesyr/ai_image_detector/data/fake/test/results/AIDE/results.json"
OUTPUT_FILE = os.path.join(
    os.path.dirname(RESULT_JSON),
    f"classification_evaluation.json"
)

# Example for real mode:
# MODE = "real"
# PREDICTIONS_PATH = "pred_real.json"
# OUTPUT_PATH      = "result_classification_real.json"


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def print_header(title, printer):
    printer("=" * 60)
    printer(title)
    printer("=" * 60)

def redirect_print_to_file(output_path):
    """Redirect prints to a .txt file next to the JSON output."""
    txt_output_path = os.path.splitext(output_path)[0] + ".txt"
    f = open(txt_output_path, "w", encoding="utf-8")
    def printer(*args, **kwargs):
        print(*args, **kwargs, file=f)
    return f, printer, txt_output_path


def safe_div(a, b):
    return a / b if b > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# FAKE MODE
# ═══════════════════════════════════════════════════════════════
def evaluate_fake(predictions, output_path):
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

    result = {
        "meta": predictions.get("meta", {}),
        "split": "fake",
        "image_count": predictions.get("image_count", total),
        "detected_fake": predictions.get("detected_fake"),
        "detected_real": predictions.get("detected_real"),
        "acc": predictions.get("acc"),
        "classification_metrics": classification_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print report to TXT
    f_txt, printer, txt_output_path = redirect_print_to_file(output_path)
    print_header("FAKE IMAGES — CLASSIFICATION EVALUATION", printer)
    printer(f"\nClassification:")
    printer(f"  TP (correctly detected fake): {tp}")
    printer(f"  FN (missed fakes):            {fn}")
    printer(f"  Recall / Sensitivity:         {classification_metrics['recall_sensitivity']}")
    printer(f"  False Negative Rate:          {classification_metrics['false_negative_rate']}")
    # printer(f"\nFalse Negatives ({fn} files):")
    # for fname in fn_files:
    #     printer(f"    {fname}")
    print(f"\nWrote {output_path}")
    print(f"TXT report written to {txt_output_path}")
    f_txt.close()


# ═══════════════════════════════════════════════════════════════
# REAL MODE
# ═══════════════════════════════════════════════════════════════
def evaluate_real(predictions, output_path):
    total = len(predictions["results"])

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

    result = {
        "meta": predictions.get("meta", {}),
        "split": "real",
        "image_count": predictions.get("image_count", total),
        "detected_fake": predictions.get("detected_fake"),
        "detected_real": predictions.get("detected_real"),
        "acc": predictions.get("acc"),
        "classification_metrics": classification_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print report to TXT
    f_txt, printer, txt_output_path = redirect_print_to_file(output_path)
    print_header("REAL IMAGES — CLASSIFICATION EVALUATION", printer)
    printer(f"\nClassification:")
    printer(f"  TN (correctly identified real): {tn}")
    printer(f"  FP (real wrongly called fake):  {fp}")
    printer(f"  Specificity / TNR:              {classification_metrics['specificity_true_negative_rate']}")
    printer(f"  False Positive Rate:            {classification_metrics['false_positive_rate']}")
    #printer(f"\nFalse Positives ({fp} files):")
    #for fname in fp_files:
    #    printer(f"    {fname}")
    printer(f"\nWrote {output_path}")
    printer(f"TXT report written to {txt_output_path}")
    f_txt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    with open(RESULT_JSON) as f:
        predictions = json.load(f)

    if MODE == "fake":
        evaluate_fake(predictions, OUTPUT_FILE)
    elif MODE == "real":
        evaluate_real(predictions, OUTPUT_FILE)
    else:
        print(f"Error: MODE must be 'fake' or 'real', got '{MODE}'")


if __name__ == "__main__":
    main()