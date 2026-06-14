# Model Comparison — PlacePredictor

This table supports cross-model comparison across the PlacePredictor team project.
Teammates can insert their model metrics after evaluation on the same held-out test split
(`test_size=0.2`, `random_state=42`, `stratify=y`).

## Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Decision Tree | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Random Forest | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **SVM (Baseline)** | 0.8605 | 0.9286 | 0.8667 | 0.8966 |
| **SVM (Tuned)** | 0.8140 | 0.9231 | 0.8000 | 0.8571 |
| KNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Naive Bayes | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## SVM Results (Auto-Generated Reference)

After running `python models/train_svm.py`, copy values from:

- **Baseline SVM:** `results/metrics/baseline_svm_metrics.json`
- **Tuned SVM:** `results/metrics/tuned_svm_metrics.json`

## Notes for Team Integration

1. Use the same preprocessing rules: drop `sl_no` and `salary`; one-hot encode categoricals; scale numerics.
2. Apply SMOTE only on training data to handle class imbalance consistently.
3. Report metrics on the **Placed** positive class for precision, recall, and F1.
4. For fair comparison, use identical train/test split parameters defined in `config.py`.

## Evaluation Protocol

| Setting | Value |
|---|---|
| Test size | 0.2 |
| Random state | 42 |
| Stratify | Yes (by `status`) |
| Primary tuning metric (SVM) | F1-Score |
| Cross-validation folds (SVM) | 5 |
