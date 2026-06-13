# ============================================================
#  PlacePredictor — Member 1: Logistic Regression
#  Author  : Jannatul Maowa Maliha
#  ID      : 0112230252
#  Branch  : logistic_regression_0112230252
#  Course  : Artificial Intelligence Lab — United International University
#
#  Responsibilities:
#    1. Data Loading & Inspection         (shared foundation)
#    2. Data Cleaning & Leakage Prevention (shared foundation)
#    3. Preprocessing Pipeline            (shared foundation)
#    4. SMOTE Class Balancing             (shared foundation)
#    5. Logistic Regression Training
#    6. Model Evaluation (4 metrics)
#    7. Output files for Member 3
# ============================================================
#
#  CHANGELOG
# ------------------------------------------------------------
#  v1.0 — 15 May 2026 — Initial implementation
#          Created project folder structure (data/raw, models, notebooks)
#          Loaded Placement_Data_Full_Class.csv and inspected all 15 columns
#
#  v1.1 — 19 May 2026 — Data cleaning and leakage prevention
#          Dropped sl_no (row index, no predictive value)
#          Dropped salary (post-placement variable, causes data leakage)
#          Encoded target column: Placed=1, Not Placed=0
#
#  v1.2 — 22 May 2026 — Preprocessing pipeline
#          Identified 5 numerical and 7 categorical columns
#          Built ColumnTransformer with StandardScaler + OneHotEncoder
#          Applied stratified train-test split (80/20, random_state=42)
#
#  v1.3 — 26 May 2026 — SMOTE integration
#          Replaced sklearn.Pipeline with imblearn.Pipeline
#          Added SMOTE inside pipeline to prevent synthetic data leakage
#          Confirmed SMOTE only runs on training data during .fit()
#
#  v1.4 — 29 May 2026 — Logistic Regression training
#          Configured LogisticRegression (max_iter=1000, solver=lbfgs)
#          Trained full pipeline on X_train, y_train
#          Verified X_test never touched during training
#
#  v1.5 — 02 Jun 2026 — Model evaluation
#          Computed Accuracy, Precision, Recall, F1-Score on X_test
#          Generated full classification report and confusion matrix
#          Result: Accuracy 86.05%, F1-Score 0.8980
#
#  v1.6 — 06 Jun 2026 — Output files
#          Saved logistic_regression_results.csv for Member 3 comparison
#          Saved logistic_regression_pipeline.pkl using joblib
#          Verified reload: loaded model produces identical F1-Score
#
#  v1.7 — 13 Jun 2026 — Final review and cleanup
#          Removed all emoji characters for Windows cp1252 compatibility
#          Added sys.stdout.reconfigure(encoding='utf-8') at top
#          Polished comments and structure for final submission
#          Pushed final version to branch logistic_regression_0112230252
# ============================================================


import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("All imports successful.")
print("-" * 50)




DATA_PATH        = "D:\place-predictor\place-predictor\data\Placement_Data_Full_Class.csv"
RESULTS_PATH     = "D:\place-predictor\place-predictor\models\logistic_regression_results.csv"
PIPELINE_PATH    = "D:\place-predictor\place-predictor\models\logistic_regression_pipeline.pkl"
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
LR_MAX_ITER      = 1000
LR_C             = 1.0
LR_SOLVER        = "lbfgs"




print("STEP 1: Loading dataset")
print("-" * 50)

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nMissing values per column:\n{df.isnull().sum().to_string()}")
# Note: salary has 67 nulls — only placed students receive a salary.
# This is expected and further confirms salary must be dropped (see Step 2).




print("\nSTEP 2: Data cleaning and leakage prevention")
print("-" * 50)

df = df.drop(columns=["sl_no", "salary"])

print(f"Dropped  : sl_no (row index), salary (post-placement leak)")
print(f"Shape now: {df.shape[0]} rows x {df.shape[1]} columns")




print("\nSTEP 3: Encoding target variable")
print("-" * 50)

df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

class_counts = df["status"].value_counts()
print(f"Encoding  : Placed=1, Not Placed=0")
print(f"Class distribution:\n{class_counts.to_string()}")
print(f"\nImbalance ratio: {class_counts[1]} Placed vs {class_counts[0]} Not Placed")
print("This imbalance is why we apply SMOTE during training (Step 7).")




print("\nSTEP 4: Splitting features and target")
print("-" * 50)

X = df.drop(columns=["status"])   # feature matrix  (215 x 12)
y = df["status"]                   # target vector   (215,)

print(f"Feature matrix X : {X.shape}")
print(f"Target vector  y : {y.shape}")
print(f"Features used    : {X.columns.tolist()}")




print("\nSTEP 5: Identifying column types")
print("-" * 50)

numerical_cols   = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

print(f"Numerical   ({len(numerical_cols)}) : {numerical_cols}")
print(f"Categorical ({len(categorical_cols)}) : {categorical_cols}")




print("\nSTEP 6: Train-test split")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")
print(f"\nTraining class distribution:\n{y_train.value_counts().to_string()}")
print(f"\nTest class distribution:\n{y_test.value_counts().to_string()}")




print("\nSTEP 7: Building ColumnTransformer")
print("-" * 50)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ],
    remainder="drop"
)

print("ColumnTransformer configured:")
print(f"  StandardScaler  -> {numerical_cols}")
print(f"  OneHotEncoder   -> {categorical_cols}")




print("\nSTEP 8: Building imblearn Pipeline")
print("-" * 50)

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=RANDOM_STATE)),
    ("classifier",   LogisticRegression(
                         max_iter=LR_MAX_ITER,
                         random_state=RANDOM_STATE,
                         C=LR_C,
                         solver=LR_SOLVER
                     )),
])

print("Pipeline steps:")
print("  Step 1 : ColumnTransformer (scale + encode)")
print("  Step 2 : SMOTE             (balance training data only)")
print("  Step 3 : LogisticRegression (train on balanced data)")




print("\nSTEP 9: Training Logistic Regression pipeline")
print("-" * 50)
print("Training in progress...")

pipeline.fit(X_train, y_train)

print("Training complete.")




print("\nSTEP 10: Evaluating model performance")
print("-" * 50)

y_pred = pipeline.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "=" * 52)
print("  LOGISTIC REGRESSION  --  EVALUATION RESULTS")
print("=" * 52)
print(f"  Accuracy  : {accuracy:.4f}   ({accuracy * 100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}   <- primary metric")
print("=" * 52)

print("\nDetailed Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Placed", "Placed"],
    digits=4
))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  {cm}")
print(f"  Rows = Actual class | Columns = Predicted class")
print(f"  True Negatives  (TN) : {cm[0][0]:3d}  -- Correctly predicted Not Placed")
print(f"  False Positives (FP) : {cm[0][1]:3d}  -- Predicted Placed, actually Not Placed")
print(f"  False Negatives (FN) : {cm[1][0]:3d}  -- Predicted Not Placed, actually Placed")
print(f"  True Positives  (TP) : {cm[1][1]:3d}  -- Correctly predicted Placed")




print("\nSTEP 11: Saving evaluation results")
print("-" * 50)

os.makedirs("../models", exist_ok=True)

results = {
    "model"     : "Logistic Regression",
    "accuracy"  : round(accuracy,  4),
    "precision" : round(precision, 4),
    "recall"    : round(recall,    4),
    "f1_score"  : round(f1,        4),
}

results_df = pd.DataFrame([results])
results_df.to_csv(RESULTS_PATH, index=False)

print(f"Saved : {RESULTS_PATH}")
print(results_df.to_string(index=False))




print("\nSTEP 12: Saving trained pipeline")
print("-" * 50)

joblib.dump(pipeline, PIPELINE_PATH)
print(f"Saved : {PIPELINE_PATH}")



print("\nSTEP 13: Verifying saved pipeline")
print("-" * 50)

loaded_pipeline = joblib.load(PIPELINE_PATH)
verify_pred     = loaded_pipeline.predict(X_test)
verify_f1       = f1_score(y_test, verify_pred)

assert round(verify_f1, 4) == round(f1, 4), \
    f"F1 mismatch after reload: {verify_f1:.4f} vs {f1:.4f}"

print(f"Reload verified. F1-Score: {verify_f1:.4f} (matches original)")




print("\n" + "=" * 52)
print("  MEMBER 1 -- SUBMISSION COMPLETE")
print("  Jannatul Maowa Maliha  |  0112230252")
print("=" * 52)
print("Output files:")
print(f"  {RESULTS_PATH}")
print(f"  {PIPELINE_PATH}")
print("\nHandoff to Member 3:")
print("  Push both output files on branch logistic_regression_0112230252")
print("  Member 3 loads logistic_regression_results.csv alongside")
print("  the other 5 models to select the champion by F1-Score.")
print("=" * 52)
