# ── STEP 1: IMPORTS ─────────────────────────────────────────
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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
import os

print("All imports successful.")


# ── STEP 2: LOAD DATA ────────────────────────────────────────


DATA_PATH = "D:\place-predictor\place-predictor\data\Placement_Data_Full_Class.csv"

df = pd.read_csv(DATA_PATH)

print(f"\n Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())


# ── STEP 3: DATA CLEANING & LEAKAGE PREVENTION ───────────────
#
# WHY we drop these two columns:
#
#   sl_no  → Just a row number (1, 2, 3...). It has zero predictive
#             power and would confuse the model.
#
#   salary → This is a POST-PLACEMENT variable. A student only gets
#             a salary AFTER they are placed. If we include it, the
#             model sees the answer while learning the question.
#             This causes fake ~100% accuracy (data leakage).
#             In a viva, saying this earns marks.

df = df.drop(columns=["sl_no", "salary"])

print("\n Dropped 'sl_no' and 'salary' (leakage prevention).")
print(f"Dataset shape after dropping: {df.shape}")


# ── STEP 4: ENCODE THE TARGET VARIABLE ───────────────────────
#
# The target column is 'status' with values "Placed" / "Not Placed".
# ML models need numbers, not strings.
# We encode: Placed = 1, Not Placed = 0

df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

print("\n Target encoded: Placed=1, Not Placed=0")
print("Class distribution:")
print(df["status"].value_counts())
print(f"\nClass balance: {df['status'].value_counts().to_dict()}")
print("(Notice the imbalance — this is why we use SMOTE)")


# ── STEP 5: SPLIT FEATURES AND TARGET ────────────────────────

X = df.drop(columns=["status"])   # everything except the target
y = df["status"]                   # the target column only

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape:  {y.shape}")
print(f"\nFeatures used: {X.columns.tolist()}")


# ── STEP 6: IDENTIFY COLUMN TYPES ────────────────────────────
#
# We need to apply different transformations to different columns:
#   Numerical  → StandardScaler  (brings all numbers to same scale)
#   Categorical → OneHotEncoder  (converts text categories to 0/1 columns)

numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumerical columns  ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")


# ── STEP 7: TRAIN-TEST SPLIT ──────────────────────────────────
#
# IMPORTANT: Split BEFORE any preprocessing or SMOTE.
# test_size=0.2  → 80% train, 20% test
# stratify=y     → keeps the same Placed/Not Placed ratio in both sets
# random_state=42 → makes results reproducible (same split every run)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set size:  {X_train.shape[0]} rows")
print(f"Test set size:      {X_test.shape[0]} rows")
print(f"\nTraining class distribution:\n{y_train.value_counts()}")
print(f"\nTest class distribution:\n{y_test.value_counts()}")


# ── STEP 8: BUILD THE PREPROCESSING PIPELINE ─────────────────
#
# ColumnTransformer applies:
#   - StandardScaler to numerical columns
#   - OneHotEncoder to categorical columns
#
# StandardScaler: transforms each number to have mean=0, std=1
#   Example: SSC% of 75 becomes a z-score relative to the class average
#   Why: Logistic Regression is sensitive to feature scale. Without
#        scaling, a feature ranging 0-100 dominates one ranging 0-1.
#
# OneHotEncoder: converts e.g. gender = ["M","F"] into two columns:
#   gender_M = [1,0,1] and gender_F = [0,1,0]
#   handle_unknown='ignore' → if test has unseen category, it won't crash

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

print("\n ColumnTransformer built.")
print("   - StandardScaler applied to:", numerical_cols)
print("   - OneHotEncoder applied to:", categorical_cols)


# ── STEP 9: BUILD THE FULL IMBLEARN PIPELINE ─────────────────
#
# We use imblearn.Pipeline (NOT sklearn.Pipeline) because it correctly
# handles SMOTE — sklearn's Pipeline does not support resamplers.
#
# Pipeline order:
#   Step 1 → preprocessor  : scale + encode the data
#   Step 2 → smote         : generate synthetic minority samples
#                            (ONLY runs during .fit(), never on test data)
#   Step 3 → classifier    : train Logistic Regression on balanced data
#


pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=1000,      # increase from default 100 to ensure convergence
        random_state=42,
        C=1.0,              # regularisation strength (default, good starting point)
        solver="lbfgs"      # efficient solver for binary classification
    )),
])

print("\n imblearn Pipeline built:")
print("   Step 1: ColumnTransformer (preprocessor)")
print("   Step 2: SMOTE (class balancer — training only)")
print("   Step 3: LogisticRegression (classifier)")


# ── STEP 10: TRAIN THE MODEL ──────────────────────────────────
#
# .fit() triggers all three pipeline steps in order:
#   1. Preprocessor fits on X_train (learns mean/std for scaling,
#      learns categories for encoding), then transforms X_train
#   2. SMOTE generates synthetic Not Placed samples until balanced
#   3. Logistic Regression trains on the balanced, preprocessed data
#
# The test set (X_test) is NEVER touched here.

print("\n Training Logistic Regression pipeline...")
pipeline.fit(X_train, y_train)
print(" Training complete.")


# ── STEP 11: EVALUATE THE MODEL ──────────────────────────────
#
# Now we use X_test (unseen data) to evaluate.
# .predict() runs only the preprocessor + classifier steps
# (SMOTE is skipped — it only runs during training).

y_pred = pipeline.predict(X_test)

# Individual metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("LOGISTIC REGRESSION — EVALUATION RESULTS")
print("="*50)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}  * primary metric for imbalanced data")
print("="*50)

# Full classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("  Rows = Actual | Columns = Predicted")
print(f"  True Negatives  (TN): {cm[0][0]}  — Correctly predicted Not Placed")
print(f"  False Positives (FP): {cm[0][1]}  — Predicted Placed but actually Not Placed")
print(f"  False Negatives (FN): {cm[1][0]}  — Predicted Not Placed but actually Placed")
print(f"  True Positives  (TP): {cm[1][1]}  — Correctly predicted Placed")


# ── STEP 12: SAVE RESULTS FOR MEMBER 3 ───────────────────────
#
# Member 3 compares all 6 models and picks the champion.
# We save our results to a shared CSV so they can load it.

os.makedirs("../models", exist_ok=True)

results = {
    "model": "Logistic Regression",
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
}

results_df = pd.DataFrame([results])
results_df.to_csv("../models/logistic_regression_results.csv", index=False)
print("\n Results saved to: models/logistic_regression_results.csv")
print(results_df)


# ── STEP 13: SAVE THE TRAINED PIPELINE ───────────────────────
#
# Save the full pipeline (preprocessor + SMOTE + model) as a .pkl file.
# Member 3 will load all 6 models, compare F1-Scores, and save
# the best one as champion_model.pkl for the Streamlit app.

joblib.dump(pipeline, "../models/logistic_regression_pipeline.pkl")
print(" Pipeline saved to: models/logistic_regression_pipeline.pkl")


# ── STEP 14: VERIFY THE SAVED MODEL WORKS ────────────────────
#
# Load it back and make a test prediction to confirm it saved correctly.

loaded_pipeline = joblib.load("../models/logistic_regression_pipeline.pkl")
verify_pred = loaded_pipeline.predict(X_test)
verify_f1 = f1_score(y_test, verify_pred)
print(f"\n Model reload verified. F1-Score matches: {verify_f1:.4f}")
