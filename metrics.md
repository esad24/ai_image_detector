# AI Image Detection Metrics Reference

## 1. Classification Fundamentals

In the context of AI detection, every prediction the model makes falls into one of four categories based on the **True Label** (Reality) and the **Predicted Label** (Model's Guess).

| Outcome | Definition |
|---|---|
| **True Positive (TP)** | The image is actually AI-generated, and the model correctly flags it as AI. |
| **True Negative (TN)** | The image is a real photograph, and the model correctly flags it as Real. |
| **False Positive (FP)** | The image is real, but the model incorrectly flags it as AI (False Alarm). |
| **False Negative (FN)** | The image is AI-generated, but the model incorrectly flags it as Real (Miss). |

## 2. Evaluation Formulas

These metrics transform the raw counts above into actionable performance scores.

### Accuracy

Measures the overall percentage of correct predictions.

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision

Measures the reliability of AI flags. High precision means that if the model says it's AI, it is very likely correct.

$$Precision = \frac{TP}{TP + FP}$$

### Recall (Sensitivity)

Measures the catch rate. High recall means the model finds most AI-generated images, even if it creates some false alarms.

$$Recall = \frac{TP}{TP + FN}$$

### Specificity

Measures the ability to correctly identify real photographs. It is the "Recall" for the negative class.

$$Specificity = \frac{TN}{TN + FP}$$

### F1 Score

The harmonic mean of Precision and Recall. Best for balanced performance evaluation, especially with uneven datasets.

$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

## 3. Similarity and Set Metrics

### Jaccard Similarity Index

Used to measure the overlap between two sets (e.g., truly AI images vs. predicted AI images). It is calculated as the intersection divided by the union.

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

In classification terms:

$$J = \frac{TP}{TP + FP + FN}$$

## 4. Business Interpretation Guide

| Metric | Business Question | Priority |
|---|---|---|
| **Precision** | "Can I trust this 'AI' label?" | High if you want to avoid accusing real artists. |
| **Recall** | "Did we miss any deepfakes?" | High if filtering harmful misinformation. |
| **Accuracy** | "How many did we get right total?" | High only if dataset is balanced (50/50 real & AI). |
| **F1 Score** | "What is the overall grade?" | High for general model comparison. |