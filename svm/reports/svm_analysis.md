# SVM Analysis Report — PlacePredictor
---

## Executive Summary

This report documents the complete Support Vector Machine (SVM) implementation for
student placement prediction. Both a baseline SVM and a GridSearchCV-tuned SVM were
trained with SMOTE-based class imbalance handling applied exclusively on training data.

### Tuned Model Performance (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.8140 |
| Precision | 0.9231 |
| Recall | 0.8000 |
| F1-Score | 0.8571 |
| AUC | 0.9308 |

### Baseline vs Tuned Comparison

| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 0.8605 | 0.8140 |
| Precision | 0.9286 | 0.9231 |
| Recall | 0.8667 | 0.8000 |
| F1-Score | 0.8966 | 0.8571 |

### Best Hyperparameters (GridSearchCV)

```json
{
  "classifier__C": 0.1,
  "classifier__gamma": "scale",
  "classifier__kernel": "linear"
}
```

**Best CV F1 Score:** 0.9022

---

## Detailed Result Analysis — Baseline SVM

### 1. Accuracy Analysis

**Value:** 0.8605 (86.05%)

Accuracy measures the proportion of all placement predictions that were correct. An accuracy of 86.05% means the SVM correctly classified 86.05% of students in the held-out test set.

**Why it matters:** Accuracy gives a quick overall picture of model reliability. However, with class imbalance (more placed than not placed students), accuracy alone can hide poor performance on the minority class.

**Impact on placement prediction:** High accuracy suggests the model captures general patterns in academic and profile data, but placement teams should pair it with recall and precision before making counselling decisions.

### 2. Precision Analysis

**Value:** 0.9286 (92.86%)

Precision answers: *Of students predicted as Placed, how many were actually placed?* A precision of 92.86% means that when the model predicts placement, it is correct roughly that often.

**Why it matters:** High precision reduces false alarms — incorrectly telling a student they will be placed when they will not. This protects student confidence and institutional credibility.

**Impact on placement prediction:** In campus recruitment, high precision helps career services prioritize resources for students the model flags with high confidence, avoiding wasted effort on overly optimistic predictions.

### 3. Recall Analysis

**Value:** 0.8667 (86.67%)

Recall answers: *Of all students who were actually placed, how many did the model identify?* A recall of 86.67% indicates the model captures that share of truly placed students.

**Why it matters:** Missing placed students (false negatives) means lost opportunities for targeted mentoring. In an imbalanced dataset, recall for the Placed class is critical.

**Impact on placement prediction:** Strong recall ensures employable students are not overlooked by the counselling system, supporting equitable placement support.

### 4. F1-Score Analysis

**Value:** 0.8966

F1 is the harmonic mean of precision and recall, balancing both concerns. An F1 of 0.8966 reflects how well the model trades off false positives against false negatives.

**Why it matters:** F1 was used as the GridSearchCV scoring metric because it penalizes models that excel on only one dimension. It is especially suitable for imbalanced placement data.

**Impact on placement prediction:** A balanced F1 indicates the SVM is useful for both identifying likely placements and avoiding misleading predictions.

### 5. Confusion Matrix Analysis

| | Predicted Not Placed | Predicted Placed |
|---|---|---|
| **Actual Not Placed** | 11 (TN) | 2 (FP) |
| **Actual Placed** | 4 (FN) | 26 (TP) |

- **True Negatives (11):** Correctly identified students who were not placed.
- **True Positives (26):** Correctly identified students who were placed.
- **False Positives (2):** Predicted placed but actually not placed — optimistic errors.
- **False Negatives (4):** Predicted not placed but actually placed — missed opportunities.

### 6. Error Analysis

The model made **6** total errors on the test set (2 false positives, 4 false negatives).

False negatives dominate errors: the model is conservative, failing to identify some students who were placed. These students may have non-linear profile patterns that the RBF kernel boundary does not fully capture.

**Common error drivers:**
- Students with moderate MBA/degree percentages but strong employability test scores
- Borderline work experience profiles (No workex but high etest_p)
- Specialisation mismatches (Mkt&HR vs Mkt&Fin) affecting recruiter preferences

### 7. Business Interpretation

**AUC Score:** 0.9128

The ROC-AUC summarizes the model's ability to rank placed students above not-placed students across all decision thresholds. Higher AUC means better separability.

#### Feature Influence on Placement (Domain Context)

**Work Experience (`workex`):**
Prior work experience is a strong employability signal. Recruiters often prefer candidates with industry exposure. Students with `workex=Yes` typically show higher placement rates, and the SVM learns this boundary in combination with test scores.

**Employability Score (`etest_p`):**
The employability test score directly measures job-readiness skills. High `etest_p` values strongly correlate with placement success because they reflect aptitude assessed by campus recruiters and training partners.

**Academic Performance (`ssc_p`, `hsc_p`, `degree_p`, `mba_p`):**
Consistent academic performance builds recruiter confidence. However, marks alone rarely guarantee placement — the SVM combines percentages with board type, stream, and specialisation to form a holistic decision boundary.

**Recommendations for Career Services:**
1. Use probability outputs (not just binary labels) to tier intervention intensity.
2. Focus mentoring on high-recall segments — students with low predicted probability but strong individual features (e.g., low MBA score but high etest_p).
3. Combine SVM predictions with SHAP explanations to justify counselling advice to students.


---

## Detailed Result Analysis — Tuned SVM

### 1. Accuracy Analysis

**Value:** 0.8140 (81.40%)

Accuracy measures the proportion of all placement predictions that were correct. An accuracy of 81.40% means the SVM correctly classified 81.40% of students in the held-out test set.

**Why it matters:** Accuracy gives a quick overall picture of model reliability. However, with class imbalance (more placed than not placed students), accuracy alone can hide poor performance on the minority class.

**Impact on placement prediction:** High accuracy suggests the model captures general patterns in academic and profile data, but placement teams should pair it with recall and precision before making counselling decisions.

### 2. Precision Analysis

**Value:** 0.9231 (92.31%)

Precision answers: *Of students predicted as Placed, how many were actually placed?* A precision of 92.31% means that when the model predicts placement, it is correct roughly that often.

**Why it matters:** High precision reduces false alarms — incorrectly telling a student they will be placed when they will not. This protects student confidence and institutional credibility.

**Impact on placement prediction:** In campus recruitment, high precision helps career services prioritize resources for students the model flags with high confidence, avoiding wasted effort on overly optimistic predictions.

### 3. Recall Analysis

**Value:** 0.8000 (80.00%)

Recall answers: *Of all students who were actually placed, how many did the model identify?* A recall of 80.00% indicates the model captures that share of truly placed students.

**Why it matters:** Missing placed students (false negatives) means lost opportunities for targeted mentoring. In an imbalanced dataset, recall for the Placed class is critical.

**Impact on placement prediction:** Strong recall ensures employable students are not overlooked by the counselling system, supporting equitable placement support.

### 4. F1-Score Analysis

**Value:** 0.8571

F1 is the harmonic mean of precision and recall, balancing both concerns. An F1 of 0.8571 reflects how well the model trades off false positives against false negatives.

**Why it matters:** F1 was used as the GridSearchCV scoring metric because it penalizes models that excel on only one dimension. It is especially suitable for imbalanced placement data.

**Impact on placement prediction:** A balanced F1 indicates the SVM is useful for both identifying likely placements and avoiding misleading predictions.

### 5. Confusion Matrix Analysis

| | Predicted Not Placed | Predicted Placed |
|---|---|---|
| **Actual Not Placed** | 11 (TN) | 2 (FP) |
| **Actual Placed** | 6 (FN) | 24 (TP) |

- **True Negatives (11):** Correctly identified students who were not placed.
- **True Positives (24):** Correctly identified students who were placed.
- **False Positives (2):** Predicted placed but actually not placed — optimistic errors.
- **False Negatives (6):** Predicted not placed but actually placed — missed opportunities.

### 6. Error Analysis

The model made **8** total errors on the test set (2 false positives, 6 false negatives).

False negatives dominate errors: the model is conservative, failing to identify some students who were placed. These students may have non-linear profile patterns that the RBF kernel boundary does not fully capture.

**Common error drivers:**
- Students with moderate MBA/degree percentages but strong employability test scores
- Borderline work experience profiles (No workex but high etest_p)
- Specialisation mismatches (Mkt&HR vs Mkt&Fin) affecting recruiter preferences

### 7. Business Interpretation

**AUC Score:** 0.9308

The ROC-AUC summarizes the model's ability to rank placed students above not-placed students across all decision thresholds. Higher AUC means better separability.

#### Feature Influence on Placement (Domain Context)

**Work Experience (`workex`):**
Prior work experience is a strong employability signal. Recruiters often prefer candidates with industry exposure. Students with `workex=Yes` typically show higher placement rates, and the SVM learns this boundary in combination with test scores.

**Employability Score (`etest_p`):**
The employability test score directly measures job-readiness skills. High `etest_p` values strongly correlate with placement success because they reflect aptitude assessed by campus recruiters and training partners.

**Academic Performance (`ssc_p`, `hsc_p`, `degree_p`, `mba_p`):**
Consistent academic performance builds recruiter confidence. However, marks alone rarely guarantee placement — the SVM combines percentages with board type, stream, and specialisation to form a holistic decision boundary.

**Recommendations for Career Services:**
1. Use probability outputs (not just binary labels) to tier intervention intensity.
2. Focus mentoring on high-recall segments — students with low predicted probability but strong individual features (e.g., low MBA score but high etest_p).
3. Combine SVM predictions with SHAP explanations to justify counselling advice to students.


---

## SHAP Explainability Analysis

SHAP (SHapley Additive exPlanations) quantifies how each feature pushes a student's predicted placement probability above or below the baseline.

### Top Features Influencing Placement Prediction

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | ssc_p | 0.2237 |
| 2 | mba_p | 0.0869 |
| 3 | hsc_p | 0.0757 |
| 4 | degree_p | 0.0672 |
| 5 | workex_Yes | 0.0193 |
| 6 | degree_t_Comm&Mgmt | 0.0192 |
| 7 | workex_No | 0.0172 |
| 8 | gender_M | 0.0162 |
| 9 | degree_t_Sci&Tech | 0.0150 |
| 10 | gender_F | 0.0139 |

### Interpretation

The most influential feature is **ssc_p**. Features related to employability testing (`etest_p`), work experience (`workex`), and MBA performance (`mba_p`) typically appear among top contributors because recruiters weight job-readiness and specialization fit heavily.

**Why predictions are made:**
- High `etest_p` SHAP values push predictions toward *Placed* — strong employability scores.
- `workex_Yes` typically increases placement probability — prior experience signals readiness.
- Low `mba_p` or `degree_p` SHAP contributions may push toward *Not Placed* unless offset by strong test scores or experience.

SHAP summary plots show directionality: red (high feature value) vs blue (low) indicates whether high values increase or decrease placement probability.


---

## Classification Reports

### Baseline SVM

```
{
  "Not Placed": {
    "precision": 0.7333333333333333,
    "recall": 0.8461538461538461,
    "f1-score": 0.7857142857142857,
    "support": 13.0
  },
  "Placed": {
    "precision": 0.9285714285714286,
    "recall": 0.8666666666666667,
    "f1-score": 0.896551724137931,
    "support": 30.0
  },
  "accuracy": 0.8604651162790697,
  "macro avg": {
    "precision": 0.8309523809523809,
    "recall": 0.8564102564102565,
    "f1-score": 0.8411330049261083,
    "support": 43.0
  },
  "weighted avg": {
    "precision": 0.869545957918051,
    "recall": 0.8604651162790697,
    "f1-score": 0.8630427311261313,
    "support": 43.0
  }
}
```

### Tuned SVM

```
{
  "Not Placed": {
    "precision": 0.6470588235294118,
    "recall": 0.8461538461538461,
    "f1-score": 0.7333333333333333,
    "support": 13.0
  },
  "Placed": {
    "precision": 0.9230769230769231,
    "recall": 0.8,
    "f1-score": 0.8571428571428571,
    "support": 30.0
  },
  "accuracy": 0.813953488372093,
  "macro avg": {
    "precision": 0.7850678733031675,
    "recall": 0.823076923076923,
    "f1-score": 0.7952380952380952,
    "support": 43.0
  },
  "weighted avg": {
    "precision": 0.8396295906555825,
    "recall": 0.813953488372093,
    "f1-score": 0.8197120708748615,
    "support": 43.0
  }
}
```

---

## Generated Artifacts

| Artifact | Path |
|---|---|
| Baseline model | `results/models/svm_baseline.joblib` |
| Tuned model | `results/models/svm_tuned.joblib` |
| Metrics | `results/metrics/` |
| Predictions | `results/predictions/` |
| Plots | `results/plots/` |
