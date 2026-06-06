"""
Train and compare student placement prediction models.
Final selected model for this project: Naive Bayes.

Run:
    python train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATASET_PATH = "dataset/Placement_Data_Full_Class.csv"
MODEL_PATH = "placement_model.pkl"
RESULTS_PATH = "model_results.csv"
REPORT_PATH = "naive_bayes_classification_report.txt"


def load_and_prepare_data():
    df = pd.read_csv(DATASET_PATH)

    # sl_no is only an ID column.
    # salary is a post-placement value, so it would cause data leakage.
    df = df.drop(columns=["sl_no", "salary"], errors="ignore")

    df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

    X = df.drop("status", axis=1)
    y = df["status"]

    return X, y


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }, y_pred


def main():
    X, y = load_and_prepare_data()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    naive_bayes_pipeline = None
    naive_bayes_pred = None

    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", clf),
            ]
        )

        result, y_pred = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

        if name == "Naive Bayes":
            naive_bayes_pipeline = pipeline
            naive_bayes_pred = y_pred

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)

    # Save final selected model: Naive Bayes
    joblib.dump(naive_bayes_pipeline, MODEL_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Final Selected Model: Naive Bayes\n")
        file.write("=================================\n\n")
        file.write(classification_report(y_test, naive_bayes_pred, target_names=["Not Placed", "Placed"]))
        file.write("\nConfusion Matrix:\n")
        file.write(str(confusion_matrix(y_test, naive_bayes_pred)))

    print("Training completed successfully.")
    print(f"Saved selected model: {MODEL_PATH}")
    print(f"Saved model comparison: {RESULTS_PATH}")
    print(f"Saved Naive Bayes report: {REPORT_PATH}")
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
