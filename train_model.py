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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = "dataset/Placement_Data_Full_Class.csv"
MODEL_PATH = "placement_model.pkl"
RESULTS_PATH = "model_results.csv"
REPORT_PATH = "naive_bayes_classification_report.txt"


def load_and_prepare_data():
    df = pd.read_csv(DATASET_PATH)

    df = df.drop(columns=["sl_no", "salary"], errors="ignore")

    df["status"] = df["status"].map({
        "Placed": 1,
        "Not Placed": 0
    })

    X = df.drop("status", axis=1)
    y = df["status"]

    return X, y


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    return preprocessor


def main():
    X, y = load_and_prepare_data()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", GaussianNB())
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Naive Bayes Model Performance")
    print("-----------------------------")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print()
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)

    results = pd.DataFrame({
        "Model": ["Naive Bayes"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1]
    })

    results.to_csv(RESULTS_PATH, index=False)

    with open(REPORT_PATH, "w") as file:
        file.write("Naive Bayes Classification Report\n")
        file.write("--------------------------------\n\n")
        file.write(classification_report(y_test, y_pred))
        file.write("\n\nConfusion Matrix\n")
        file.write(str(confusion_matrix(y_test, y_pred)))

    print("Model saved successfully as placement_model.pkl")
    print("Results saved successfully as model_results.csv")
    print("Report saved successfully as naive_bayes_classification_report.txt")


if __name__ == "__main__":
    main()