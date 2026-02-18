# Project File Structure

> AI-generated image detection experiment — 57 files organized by dataset split and model.

---

## 1. Reference & Configuration Files

Shared reference and configuration files used across all experiments.

| File | Type | Description |
|------|------|-------------|
| `images_metadata.csv` | CSV | Master lookup table: filename, classification (real/fake), and generator model for 100 test images (50 fake, 50 real images). |
| `prompts.json` | Config | Defines the 6 prompt variants (prompt1–prompt6) used to instruct LLM detectors, including full prompt text and strategy metadata. |
| `new_artifacts.json` | Dataset | Lists only the new artifacts for each image, that were discovered after manually reviewing the prompt3 results of gpt-5.2 and qwen-3. The artifacts were not in ground truth 1 |
| `gt_1.json` | Ground Truth | First round of human annotations — was used in the frist evaluations (noted by gt_2 in filename or see 'comparison' path included the json). |
| `gt_2.json` | Ground Truth | Second / updated annotation set. New artifacts find by Gpt-5.2 and Qwen-3 were added to ground truth 1. This was used in most later evaluation rounds and semantic similarity comparisons (noted by gt_2 in filename or see 'comparison' path included the json). |

---

# Fake Image Dataset

Results and evaluations run on AI-generated (fake) images, organized by model.

---

## 2. Baseline Detectors — Fake Images

Traditional/non-LLM detectors used as baselines. Each produces a raw results file and a classification evaluation.

### AIDE (GenImage_train)

| File | Description |
|------|-------------|
| `results_fake_AIDE_results.json` | Raw AIDE predictions on fake images: per-image decisions, overall accuracy, detected fake/real counts. |
| `results_fake_AIDE_classification_evaluation.json` | Classification metrics (accuracy, precision, recall, F1) for AIDE on the fake-image split. |

### UnivFD (CLIP:ViT-L/14)

| File | Description |
|------|-------------|
| `results_fake_UnivFD_results.json` | Raw UnivFD predictions on fake images with per-image and aggregate scores. |
| `results_fake_UnivFD_classification_evaluation.json` | Classification metrics for UnivFD on the fake-image split. |

---

## 3. GPT-5.2 — Fake Images

GPT-5.2 was tested with prompt variants 1–4 and 6. Prompt3 includes a pre/post manual review split. Evaluation types vary by prompt strategy.

| File | Eval Type | Description |
|------|-----------|-------------|
| `results_fake_gpt-5_2_prompt1_2026-01-18_20-39-12_results.json` | Classification | Raw GPT-5.2 predictions on fake images using prompt1 (binary classification). |
| `results_fake_gpt-5_2_prompt1_2026-01-18_20-39-12_classification_evaluation.json` | Classification | Accuracy / F1 metrics for prompt1 on fake images. |
| `results_fake_gpt-5_2_prompt2_2026-01-19_18-41-02_results.json` | Artifact Type | Raw GPT-5.2 outputs using prompt2 (artifact-type detection strategy). |
| `results_fake_gpt-5_2_prompt2_2026-01-19_18-41-02_artifact_type_evaluation_gt_1.json` | Artifact Type | Artifact-type evaluation against ground-truth set #1. |
| `results_fake_gpt-5_2_prompt2_2026-01-19_18-41-02_artifact_type_evaluation_gt_2.json` | Artifact Type | Artifact-type evaluation against ground-truth set #2. |
| `results_fake_gpt-5_2_prompt3_2026-01-12_17-13-35_results_pre_manual_review.json` | Sem. Similarity | Prompt3 raw outputs before human review (pre-human-review snapshot). |
| `results_fake_gpt-5_2_prompt3_2026-01-12_17-13-35_semantic_similarity_evaluation_gt_1_0_6.json` | Sem. Similarity | Semantic similarity at threshold 0.6 vs. gt_1 (pre-human-review). |
| `results_fake_gpt-5_2_prompt3_2026-01-20_18-46-16_results_manual_review.json` | Sem. Similarity | Prompt3 results after manual human review / correction. |
| `results_fake_gpt-5_2_prompt3_2026-01-20_18-46-16_semantic_similarity_evaluation_gt_2_0_6.json` | Sem. Similarity | Semantic similarity at threshold 0.6 vs. gt_2 (post-review). |
| `results_fake_gpt-5_2_prompt4_2026-01-23_08-50-39_results.json` | Sem. Similarity | Raw GPT-5.2 outputs using prompt4. |
| `results_fake_gpt-5_2_prompt4_2026-01-23_08-50-39_semantic_similarity_evaluation_gt_2_0_6.json` | Sem. Similarity | Semantic similarity vs. gt_2 for prompt4. |
| `results_fake_gpt-5_2_prompt6_2026-01-23_09-46-58_results.json` | Sem. Similarity | Raw GPT-5.2 outputs using prompt6. |
| `results_fake_gpt-5_2_prompt6_2026-01-23_09-46-58_semantic_similarity_evaluation_gt_2_0_6.json` | Sem. Similarity | Semantic similarity vs. gt_2 for prompt6. |

---

## 4. Qwen3-VL — Fake Images

Qwen3-VL was tested with the same prompt variants as GPT-5. Prompt3 also includes a pre/post manual review split.

| File | Eval Type | Description |
|------|-----------|-------------|
| `results_fake_qwen3-vl_prompt1_2026-01-22_21-17-46_results.json` | Classification | Raw Qwen3-VL predictions using prompt1 on fake images. |
| `results_fake_qwen3-vl_prompt1_2026-01-22_21-17-46_classification_evaluation.json` | Classification | Accuracy / F1 metrics for prompt1. |
| `results_fake_qwen3-vl_prompt2_2026-01-24_13-54-27_results.json` | Artifact Type | Raw outputs using prompt2 (artifact-type detection). |
| `results_fake_qwen3-vl_prompt2_2026-01-24_13-54-27_artifact_type_evaluation_gt_1.json` | Artifact Type | Artifact-type evaluation against gt_1. |
| `results_fake_qwen3-vl_prompt2_2026-01-24_13-54-27_artifact_type_evaluation_gt_2.json` | Artifact Type | Artifact-type evaluation against gt_2. |
| `results_fake_qwen3-vl_prompt3_2026-01-21_13-23-15_results_pre_manual_review.json` | Sem. Similarity | Prompt3 raw outputs before human review. |
| `results_fake_qwen3-vl_prompt3_2026-01-21_13-23-15_semantic_similarity_evaluation_gt_1_0_6.json` | Sem. Similarity | Semantic similarity vs. gt_1 at threshold 0.6 (pre-human-review). |
| `results_fake_qwen3-vl_prompt3_2026-01-22_21-28-42_results_manual_review.json` | Sem. Similarity | Prompt3 results after manual human review. |
| `results_fake_qwen3-vl_prompt3_2026-01-22_21-28-42_semantic_similarity_evaluation_gt_2_0_6.json` | Sem. Similarity | Semantic similarity vs. gt_2 (post-human-review). |
| `results_fake_qwen3-vl_prompt4_2026-01-24_15-31-29_results.json` | Sem. Similarity | Raw outputs using prompt4. |
| `results_fake_qwen3-vl_prompt4_2026-01-24_15-31-29_semantic_similarity_evaluation_gt_2_0_6.json` | Sem. Similarity | Semantic similarity vs. gt_2 for prompt4. |
| `results_fake_qwen3-vl_prompt6_2026-01-24_15-59-12_results.json` | Sem. Similarity | Raw outputs using prompt6. |
| `results_fake_qwen3-vl_prompt6_2026-01-24_15-59-12_semantic_similarity_evaluation_gt_2_0_6.json` | Sem. Similarity | Semantic similarity vs. gt_2 for prompt6. |

---

# Real Image Dataset

Results and evaluations run on real (authentic) images, organized by model.

---

## 5. Baseline Detectors — Real Images

Same baseline detectors applied to the real-image split.

### AIDE

| File | Description |
|------|-------------|
| `results_real_AIDE_results.json` | Raw AIDE predictions on real images. |
| `results_real_AIDE_classification_evaluation.json` | Classification metrics (accuracy, precision, recall, F1) for AIDE on the real-image split. |

### UnivFD

| File | Description |
|------|-------------|
| `results_real_UnivFD_results.json` | Raw UnivFD predictions on real images. |
| `results_real_UnivFD_classification_evaluation.json` | Classification metrics for UnivFD on the real-image split. |

---

## 6. GPT-5.2 — Real Images

GPT-5.2 was tested with prompt variants 1–5 on real images. Note: prompt5 exists only on the real split, and real-image experiments do not include a manual review step. The artifact types, reasoning and location could obviously not be compared to a ground truth. The focus was rather on the hallucination.

| File | Eval Type | Description |
|------|-----------|-------------|
| `results_real_gpt-5_2_prompt1_2026-01-20_18-56-12_results.json` | Classification | Raw GPT-5.2 predictions using prompt1 on real images. |
| `results_real_gpt-5_2_prompt1_2026-01-20_18-56-12_classification_evaluation.json` | Classification | Accuracy / F1 metrics for prompt1. |
| `results_real_gpt-5_2_prompt2_2026-01-20_19-03-04_results.json` | Artifact Type | Raw outputs using prompt2 (artifact-type detection). |
| `results_real_gpt-5_2_prompt2_2026-01-20_19-03-04_artifact_type_evaluation.json` | Artifact Type | Artifact-type evaluation for prompt2 on real images. |
| `results_real_gpt-5_2_prompt3_2026-01-20_20-09-43_results.json` | Artifact Type | Raw outputs using prompt3 on real images. |
| `results_real_gpt-5_2_prompt3_2026-01-20_20-09-43_artifact_type_evaluation.json` | Artifact Type | Artifact-type evaluation for prompt3 on real images. |
| `results_real_gpt-5_2_prompt4_2026-02-04_19-03-02_results.json` | Classification | Raw outputs using prompt4 on real images. |
| `results_real_gpt-5_2_prompt4_2026-02-04_19-03-02_classification_evaluation.json` | Classification | Classification metrics for prompt4. |
| `results_real_gpt-5_2_prompt5_2026-01-23_09-35-54_results.json` | Classification | Raw outputs using prompt5 on real images. |
| `results_real_gpt-5_2_prompt5_2026-01-23_09-35-54_classification_evaluation.json` | Classification | Classification metrics for prompt5. |

---

## 7. Qwen3-VL — Real Images

Qwen3-VL was tested with prompt variants 1–5 on real images. Note: prompt5 exists only on the real split, and real-image experiments do not include a manual review step. The artifact types, reasoning and location could obviously not be compared to a ground truth. The focus was rather on the hallucination.

| File | Eval Type | Description |
|------|-----------|-------------|
| `results_real_qwen3-vl_prompt1_2026-01-23_08-17-57_results.json` | Classification | Raw Qwen3-VL predictions using prompt1 on real images. |
| `results_real_qwen3-vl_prompt1_2026-01-23_08-17-57_classification_evaluation.json` | Classification | Classification metrics for prompt1. |
| `results_real_qwen3-vl_prompt2_2026-01-24_17-55-36_results.json` | Artifact Type | Raw outputs using prompt2. |
| `results_real_qwen3-vl_prompt2_2026-01-24_17-55-36_artifact_type_evaluation.json` | Artifact Type | Artifact-type evaluation for prompt2. |
| `results_real_qwen3-vl_prompt3_2026-01-24_16-37-23_results.json` | Artifact Type | Raw outputs using prompt3. |
| `results_real_qwen3-vl_prompt3_2026-01-24_16-37-23_artifact_type_evaluation.json` | Artifact Type | Artifact-type evaluation for prompt3. |
| `results_real_qwen3-vl_prompt4_2026-01-24_18-27-42_results.json` | Classification | Raw outputs using prompt4. |
| `results_real_qwen3-vl_prompt4_2026-01-24_18-27-42_classification_evaluation.json` | Classification | Classification metrics for prompt4. |
| `results_real_qwen3-vl_prompt5_2026-02-16_09-08-13_results.json` | Classification | Raw outputs using prompt5 (most recent run, Feb 2026). |
| `results_real_qwen3-vl_prompt5_2026-02-16_09-08-13_classification_evaluation.json` | Classification | Classification metrics for prompt5. |