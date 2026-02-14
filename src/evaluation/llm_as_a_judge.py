#author: claude sonnet 4.5

import json
from tqdm import tqdm
import numpy as np
import os
import warnings
from typing import List, Dict, Optional
import time
from openai import OpenAI
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
RESULT_PATH = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/results_final.json"
GT = "gt_1"
GROUND_TRUTH_PATH = "/home/usluesyr/ai_image_detector/data/ground_truth/gt_1/gt_1.json"

# Configuration
LLM_MODEL = "gpt-5.2"  # or "gpt-4o-mini" for faster/cheaper, "claude-3-5-sonnet-20241022" for Anthropic
MATCH_THRESHOLD = 0.6  # Minimum score to consider a match (0-1 scale)
USE_BATCH_API = False  # Set to True for cheaper batch processing (slower)
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# System prompt for the LLM judge
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator comparing AI-generated image artifacts. 
Your task is to determine if two artifact descriptions refer to the same underlying image artifact.

Consider:
1. **Type consistency**: Do both artifacts refer to the same type of manipulation/feature?
2. **Semantic similarity**: Do the descriptions refer to the same visual phenomenon?
3. **Location alignment**: Do they describe the same region of the image?
4. **Reasoning overlap**: Is the underlying reasoning for flagging the artifact the same?

An artifact consists of:
- type: The category of artifact (e.g., "inconsistent_lighting", "unnatural_texture", "anatomical_error")
- reasoning: Why this artifact was flagged
- location: Where in the image the artifact appears

Respond in JSON format only with this structure:
{
  "is_match": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your decision",
  "type_match": true/false,
  "semantic_similarity": 0.0-1.0,
  "location_similarity": 0.0-1.0
}

Be strict but fair. Minor wording differences don't matter if the core meaning is the same.
"""


def create_comparison_prompt(result_artifact: Dict, gt_artifact: Dict) -> str:
    """Create a prompt for comparing two artifacts."""
    return f"""Compare these two image artifacts:

RESULT ARTIFACT:
- Type: {result_artifact.get('type', 'N/A')}
- Reasoning: {result_artifact.get('reasoning', 'N/A')}
- Location: {result_artifact.get('location', 'N/A')}

GROUND TRUTH ARTIFACT:
- Type: {gt_artifact.get('type', 'N/A')}
- Reasoning: {gt_artifact.get('reasoning', 'N/A')}
- Location: {gt_artifact.get('location', 'N/A')}

Do these artifacts refer to the same image manipulation/feature? Provide your evaluation in JSON format."""


def query_llm_judge(result_artifact: Dict, gt_artifact: Dict, retries: int = MAX_RETRIES) -> Optional[Dict]:
    """Query the LLM to judge if two artifacts match."""
    user_prompt = create_comparison_prompt(result_artifact, gt_artifact)
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                #max_tokens=500,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate response structure
            required_keys = ["is_match", "confidence", "reasoning"]
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys in LLM response: {result}")
            
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"All retries failed for artifact comparison")
                return None
    
    return None


def compute_similarity_matrix(result_artifacts: List[Dict], gt_artifacts: List[Dict]) -> np.ndarray:
    """
    Compute similarity matrix using LLM judge.
    Returns matrix of shape (len(result_artifacts), len(gt_artifacts))
    """
    n_result = len(result_artifacts)
    n_gt = len(gt_artifacts)
    
    similarity_matrix = np.zeros((n_result, n_gt))
    judgments = [[None for _ in range(n_gt)] for _ in range(n_result)]
    
    total_comparisons = n_result * n_gt
    pbar = tqdm(total=total_comparisons, desc="LLM judgments", leave=False)
    
    for i, r_art in enumerate(result_artifacts):
        for j, g_art in enumerate(gt_artifacts):
            judgment = query_llm_judge(r_art, g_art)
            
            if judgment:
                # Use confidence score as similarity
                similarity_matrix[i, j] = judgment['confidence'] if judgment['is_match'] else 0.0
                judgments[i][j] = judgment
            else:
                similarity_matrix[i, j] = 0.0
                judgments[i][j] = {
                    "is_match": False,
                    "confidence": 0.0,
                    "reasoning": "LLM query failed",
                    "error": True
                }
            
            pbar.update(1)
            
            # Rate limiting (adjust based on your API tier)
            time.sleep(0.1)
    
    pbar.close()
    return similarity_matrix, judgments


def compute_llm_matches(result_artifacts: List[Dict], gt_artifacts: List[Dict], 
                        threshold: float = MATCH_THRESHOLD) -> Dict:
    """
    Compute optimal artifact matches using LLM-as-a-Judge with Hungarian algorithm.
    """
    # Handle edge cases
    if not result_artifacts and not gt_artifacts:
        return create_perfect_match_result()
    
    if not result_artifacts or not gt_artifacts:
        return create_no_match_result(result_artifacts, gt_artifacts)
    
    # Get similarity matrix from LLM
    similarity_matrix, judgments = compute_similarity_matrix(result_artifacts, gt_artifacts)
    
    # Convert similarity to cost for Hungarian algorithm
    cost_matrix = 1 - similarity_matrix
    
    # Pad matrix if sizes differ
    max_size = max(len(result_artifacts), len(gt_artifacts))
    padded_cost = np.ones((max_size, max_size))
    padded_cost[:len(result_artifacts), :len(gt_artifacts)] = cost_matrix
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Build matches
    matches = []
    matched_gt_indices = set()
    
    for r_idx, g_idx in zip(row_ind, col_ind):
        if r_idx < len(result_artifacts) and g_idx < len(gt_artifacts):
            score = similarity_matrix[r_idx, g_idx]
            judgment = judgments[r_idx][g_idx]
            
            if score >= threshold:
                matches.append({
                    "result_artifact": result_artifacts[r_idx],
                    "matched_gt_artifact": gt_artifacts[g_idx],
                    "match_score": float(score),
                    "llm_judgment": judgment
                })
                matched_gt_indices.add(g_idx)
            else:
                matches.append({
                    "result_artifact": result_artifacts[r_idx],
                    "matched_gt_artifact": None,
                    "match_score": 0.0,
                    "llm_judgment": judgment
                })
    
    # Add unmatched GT artifacts
    for g_idx in range(len(gt_artifacts)):
        if g_idx not in matched_gt_indices:
            matches.append({
                "result_artifact": None,
                "matched_gt_artifact": gt_artifacts[g_idx],
                "match_score": 0.0,
                "llm_judgment": {"is_match": False, "confidence": 0.0, "reasoning": "No result artifact matched"}
            })
    
    # Calculate metrics
    all_scores = [m["match_score"] for m in matches]
    non_zero_scores = [s for s in all_scores if s > 0.0]
    
    true_positives = sum(1 for m in matches if m["match_score"] >= threshold and m["matched_gt_artifact"] is not None)
    false_positives = sum(1 for m in matches if m["result_artifact"] is not None and m["matched_gt_artifact"] is None)
    false_negatives = sum(1 for m in matches if m["result_artifact"] is None and m["matched_gt_artifact"] is not None)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "mean_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "max_score": float(np.max(all_scores)) if all_scores else 0.0,
        "min_score": float(np.min(all_scores)) if all_scores else 0.0,
        "mean_score_non_zero": float(np.mean(non_zero_scores)) if non_zero_scores else 0.0,
        "max_score_non_zero": float(np.max(non_zero_scores)) if non_zero_scores else 0.0,
        "min_score_non_zero": float(np.min(non_zero_scores)) if non_zero_scores else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": matches
    }


def create_perfect_match_result() -> Dict:
    """Return perfect score when both lists are empty."""
    return {
        "mean_score": 1.0,
        "max_score": 1.0,
        "min_score": 1.0,
        "mean_score_non_zero": 1.0,
        "max_score_non_zero": 1.0,
        "min_score_non_zero": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "matches": []
    }


def create_no_match_result(result_artifacts: List[Dict], gt_artifacts: List[Dict]) -> Dict:
    """Return no-match result when one list is empty."""
    matches = []
    for r in result_artifacts:
        matches.append({
            "result_artifact": r,
            "matched_gt_artifact": None,
            "match_score": 0.0,
            "llm_judgment": {"is_match": False, "confidence": 0.0, "reasoning": "No ground truth to compare"}
        })
    for g in gt_artifacts:
        matches.append({
            "result_artifact": None,
            "matched_gt_artifact": g,
            "match_score": 0.0,
            "llm_judgment": {"is_match": False, "confidence": 0.0, "reasoning": "No result artifact to compare"}
        })
    
    return {
        "mean_score": 0.0,
        "max_score": 0.0,
        "min_score": 0.0,
        "mean_score_non_zero": 0.0,
        "max_score_non_zero": 0.0,
        "min_score_non_zero": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "true_positives": 0,
        "false_positives": len(result_artifacts),
        "false_negatives": len(gt_artifacts),
        "matches": matches
    }


def main():
    print(f"Using LLM: {LLM_MODEL}")
    print(f"Match threshold: {MATCH_THRESHOLD}")
    
    # Load JSONs
    with open(RESULT_PATH, "r") as f:
        result_json = json.load(f)
    
    with open(GROUND_TRUTH_PATH, "r") as f:
        gt_json = json.load(f)
    
    # Build lookup table for ground truth by filename
    gt_map = {item["filename"]: item for item in gt_json["results"]}
    
    file_scores = []
    
    for image in tqdm(result_json["results"], desc="Evaluating images"):
        filename = image["filename"]
        classification = image["classification"]
        gt_item = gt_map.get(filename)
        
        if gt_item is None:
            print(f"Warning: No ground truth found for {filename}")
            continue
        
        # Compute LLM-based artifact matches
        llm_result = compute_llm_matches(
            image["artifacts"], 
            gt_item["artifacts"], 
            threshold=MATCH_THRESHOLD
        )
        
        file_scores.append({
            "filename": filename,
            "classification": classification,
            "mean_match_score": llm_result["mean_score"],
            "max_match_score": llm_result["max_score"],
            "min_match_score": llm_result["min_score"],
            "mean_match_score_non_zero": llm_result["mean_score_non_zero"],
            "max_match_score_non_zero": llm_result["max_score_non_zero"],
            "min_match_score_non_zero": llm_result["min_score_non_zero"],
            "precision": llm_result["precision"],
            "recall": llm_result["recall"],
            "f1_score": llm_result["f1_score"],
            "true_positives": llm_result["true_positives"],
            "false_positives": llm_result["false_positives"],
            "false_negatives": llm_result["false_negatives"],
            "num_result_artifacts": len(image["artifacts"]),
            "num_gt_artifacts": len(gt_item["artifacts"]),
            "matched_artifacts": llm_result["matches"]
        })
    
    # Summary statistics
    summary = {
        "mean_match_score_avg": float(np.mean([f["mean_match_score"] for f in file_scores])),
        "max_match_score_avg": float(np.mean([f["max_match_score"] for f in file_scores])),
        "mean_match_score_non_zero_avg": float(np.mean([f["mean_match_score_non_zero"] for f in file_scores])),
        "max_match_score_non_zero_avg": float(np.mean([f["max_match_score_non_zero"] for f in file_scores])),
        "avg_precision": float(np.mean([f["precision"] for f in file_scores])),
        "avg_recall": float(np.mean([f["recall"] for f in file_scores])),
        "avg_f1_score": float(np.mean([f["f1_score"] for f in file_scores])),
        "total_true_positives": sum(f["true_positives"] for f in file_scores),
        "total_false_positives": sum(f["false_positives"] for f in file_scores),
        "total_false_negatives": sum(f["false_negatives"] for f in file_scores),
    }
    
    # Final output
    output = {
        "meta": result_json.get("meta", {}),
        "evaluation_method": "llm_as_judge",
        "llm_model": LLM_MODEL,
        "comparison": GROUND_TRUTH_PATH,
        "match_threshold": MATCH_THRESHOLD,
        "image_count": result_json.get("image_count", len(result_json["results"])),
        "total_fakes": result_json.get("total_fakes", 0),
        "total_real": result_json.get("total_real", 0),
        "image_scores": file_scores,
        "dataset_average": summary
    }
    
    # Save JSON
    results_dir = os.path.dirname(RESULT_PATH)
    output_path = os.path.join(results_dir, f"llm_judge_evaluation_{GT}_{LLM_MODEL}_{MATCH_THRESHOLD}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"LLM-as-a-Judge evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    print(f"\nDataset Summary:")
    print(f"  Average Precision: {summary['avg_precision']:.3f}")
    print(f"  Average Recall: {summary['avg_recall']:.3f}")
    print(f"  Average F1 Score: {summary['avg_f1_score']:.3f}")
    print(f"  Mean Match Score: {summary['mean_match_score_avg']:.3f}")
    print(f"  Total True Positives: {summary['total_true_positives']}")
    print(f"  Total False Positives: {summary['total_false_positives']}")
    print(f"  Total False Negatives: {summary['total_false_negatives']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()