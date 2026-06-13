# PlacePredictor — SVM Module

---

## Problem Description

Campus placement is a critical outcome for MBA students and institutions alike.
PlacePredictor aims to predict whether a student will be **Placed** or **Not Placed**
during campus recruitment using historical academic and profile data.

The SVM module provides:

- Binary placement classification with probability estimates
- Hyperparameter-tuned model via GridSearchCV
- Class imbalance handling with SMOTE
- Explainable AI via SHAP
- Comprehensive evaluation and visualization

---

## Dataset Description

**File:** `data/Placement_Data_Full_Class.csv`

| Column | Type | Description |
|---|---|---|
| `gender` | Categorical | Student gender (M/F) |
| `ssc_p` | Numerical | Secondary School Certificate percentage |
| `ssc_b` | Categorical | SSC board |
| `hsc_p` | Numerical | Higher Secondary Certificate percentage |
| `hsc_b` | Categorical | HSC board |
| `hsc_s` | Categorical | HSC stream |
| `degree_p` | Numerical | Degree percentage |
| `degree_t` | Categorical | Degree type |
| `workex` | Categorical | Work experience (Yes/No) |
| `etest_p` | Numerical | Employability test percentage |
| `specialisation` | Categorical | MBA specialisation |
| `mba_p` | Numerical | MBA percentage |
| `status` | Target | Placed / Not Placed |

**Removed columns:**

- `salary` — post-placement outcome (data leakage)
- `sl_no` — row identifier

---

## Project Structure

```
PlacePredictor/
├── data/
│   └── Placement_Data_Full_Class.csv
├── models/
│   ├── svm_pipeline.py      # Preprocessing + SVM pipeline
│   ├── train_svm.py           # Training orchestration
│   ├── evaluate_svm.py        # Metrics & interpretation
│   └── shap_analysis.py       # SHAP explainability
├── reports/
│   ├── svm_analysis.md        # Auto-generated analysis
│   ├── contribution_report.md
│   └── model_comparison.md
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   └── models/
├── notebooks/
│   └── svm_experiment.ipynb
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Start

```bash
# 1. Clone and checkout branch
git checkout svm_0112331000

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full SVM pipeline
py models/train_svm.py
# or: python models/train_svm.py
```

Outputs are written to `results/` and `reports/svm_analysis.md`.

---

## Methodology

### Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### Preprocessing Pipeline

```
Raw Features
    → ColumnTransformer
        → OneHotEncoder (categorical)
        → StandardScaler (numerical)
    → SMOTE (training only)
    → SVC (RBF/Linear kernel)
```

### Class Imbalance

`SMOTE(random_state=42)` is embedded in an `imblearn` Pipeline so oversampling
occurs only during `fit()` on training folds — never on test data.

---

## SVM Architecture

**Baseline model:**

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
)
```

The RBF kernel maps features into a higher-dimensional space, enabling non-linear
decision boundaries between placed and not-placed student profiles.

---

## Hyperparameter Tuning

GridSearchCV searches:

| Parameter | Values |
|---|---|
| `C` | 0.1, 1, 10, 100 |
| `gamma` | scale, 0.1, 0.01, 0.001 |
| `kernel` | rbf, linear |

| Setting | Value |
|---|---|
| CV folds | 5 |
| Scoring | F1-Score |

Both baseline and tuned models are saved to `results/models/`.

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness |
| **Precision** | Reliability of positive (Placed) predictions |
| **Recall** | Coverage of actually placed students |
| **F1-Score** | Harmonic balance of precision and recall |
| **AUC** | Ranking quality across thresholds |

Generated artifacts:

- Classification report (JSON)
- Confusion matrix plot
- ROC curve plot
- Detailed interpretation in `reports/svm_analysis.md`

---

## Explainable AI

SHAP KernelExplainer analyzes the tuned SVM to identify:

- Which features most influence placement predictions
- Direction of influence (high vs low feature values)
- Feature importance ranking (`results/metrics/shap_feature_importance.csv`)

Key domain features:

- **Work experience** — prior industry exposure
- **Employability test score (`etest_p`)** — job-readiness signal
- **Academic scores** — SSC, HSC, degree, and MBA percentages

---

## Result Analysis

After training, see `reports/svm_analysis.md` for:

1. Accuracy Analysis
2. Precision Analysis
3. Recall Analysis
4. F1 Analysis
5. Confusion Matrix Analysis
6. Error Analysis
7. Business Interpretation

---

## Model Comparison

Team members can populate `reports/model_comparison.md` with metrics from:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM (this module)
- KNN
- Naive Bayes

---

## Future Improvements

1. **Probability calibration** — Platt scaling or isotonic regression for better probability estimates
2. **Feature engineering** — Academic consistency score, employability index composites
3. **Alternative kernels** — Polynomial kernel exploration
4. **Ensemble methods** — Stack SVM with tree-based models
5. **Real-time API** — Deploy tuned pipeline as a FastAPI inference service
6. **Cross-validation reporting** — Nested CV for unbiased performance estimates
7. **Fairness analysis** — Audit predictions across gender and board categories

---


