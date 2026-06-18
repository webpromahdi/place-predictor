# Contribution Report — PlacePredictor SVM Module


## Contribution Summary

Complete SVM implementation, tuning, evaluation, visualization, and analysis for the
PlacePredictor — Machine Learning-Based Student Placement Prediction System.

## Deliverables

### 1. Core Implementation

| Component | File | Description |
|---|---|---|
| Configuration | `config.py` | Central paths, feature lists, hyperparameters |
| Preprocessing Pipeline | `models/svm_pipeline.py` | ColumnTransformer, SMOTE, SVC pipeline |
| Training | `models/train_svm.py` | End-to-end training orchestration |
| Evaluation | `models/evaluate_svm.py` | Metrics, plots, detailed interpretation |
| Explainability | `models/shap_analysis.py` | SHAP summary and feature importance |

### 2. Models Trained

- **Baseline SVM:** `SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)`
- **Tuned SVM:** GridSearchCV over `C`, `gamma`, and `kernel` with 5-fold CV and F1 scoring

### 3. Preprocessing

- Removed `salary` (data leakage) and `sl_no` (identifier)
- OneHotEncoder (`handle_unknown="ignore"`) for categorical features
- StandardScaler for numerical features
- SMOTE (`random_state=42`) applied only on training data

### 4. Evaluation Outputs

- Accuracy, Precision, Recall, F1-Score
- Classification Report
- Confusion Matrix (plot + analysis)
- ROC Curve and AUC Score
- Detailed business and error interpretation

### 5. Visualizations

All saved under `results/plots/`:

1. Correlation Heatmap
2. Placement Distribution Chart
3. Placement Rate by Category
4. Score Distribution Boxplots
5. Confusion Matrix (baseline & tuned)
6. ROC Curve (baseline & tuned)
7. SHAP Summary Plot

### 6. Documentation

- `README.md` — Project overview and methodology
- `reports/svm_analysis.md` — Auto-generated detailed analysis
- `reports/model_comparison.md` — Team comparison table template
- `notebooks/svm_experiment.ipynb` — Interactive experiment notebook

### 7. Reproducibility

- Fixed random seed (`42`) throughout
- Modular, typed Python code with logging and exception handling
- `requirements.txt` for dependency management
- `.gitignore` for GitHub-ready repository

## How to Run

```bash
pip install -r requirements.txt
python models/train_svm.py
```

## Branch Workflow

All SVM work is developed on branch **`svm_0112331000`** for team integration.
