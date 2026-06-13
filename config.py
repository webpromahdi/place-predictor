"""Central configuration for the PlacePredictor SVM module."""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data paths
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PATH: Path = DATA_DIR / "Placement_Data_Full_Class.csv"

# Results paths
RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"
MODELS_DIR: Path = RESULTS_DIR / "models"

# Reports
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Columns to drop
DROP_COLUMNS: list[str] = ["sl_no", "salary"]

# Target column
TARGET_COLUMN: str = "status"

# Feature groups
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
]

NUMERICAL_FEATURES: list[str] = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p",
]

FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Train/test split
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# SVM baseline hyperparameters
SVM_BASELINE_PARAMS: dict = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,
    "random_state": RANDOM_STATE,
}

# GridSearchCV parameter grid
SVM_PARAM_GRID: dict = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__gamma": ["scale", 0.1, 0.01, 0.001],
    "classifier__kernel": ["rbf", "linear"],
}

# Cross-validation settings
CV_FOLDS: int = 5
# pos_label required because target classes are string labels ("Placed", "Not Placed")
CV_SCORING: str = "f1"  # resolved via get_cv_scorer() in train_svm.py

# Class labels
POSITIVE_CLASS: str = "Placed"
NEGATIVE_CLASS: str = "Not Placed"

# Model filenames
BASELINE_MODEL_FILENAME: str = "svm_baseline.joblib"
TUNED_MODEL_FILENAME: str = "svm_tuned.joblib"

# Plot settings
PLOT_DPI: int = 150
FIGURE_FORMAT: str = "png"
