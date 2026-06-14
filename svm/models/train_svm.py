"""Training script for PlacePredictor SVM module."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

# Ensure project root is on path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.evaluate_svm import run_full_evaluation
from models.shap_analysis import generate_shap_interpretation, run_shap_analysis
from models.svm_pipeline import build_svm_pipeline, get_baseline_svm, get_tuning_param_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_cv_scorer():
    """F1 scorer with explicit positive class for string-encoded labels."""
    return make_scorer(f1_score, pos_label=config.POSITIVE_CLASS)


def ensure_directories() -> None:
    """Create required output directories."""
    for directory in [
        config.METRICS_DIR,
        config.PLOTS_DIR,
        config.PREDICTIONS_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare the placement dataset.

    Drops identifier and leakage columns; returns features and target.
    """
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {config.DATA_PATH}")

    logger.info("Loading dataset from %s", config.DATA_PATH)
    df = pd.read_csv(config.DATA_PATH)

    missing_cols = set(config.DROP_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.warning("Expected drop columns not found: %s", missing_cols)

    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in dataset.")

    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()

    logger.info("Dataset loaded: %d samples, %d features", len(X), len(config.FEATURE_COLUMNS))
    logger.info("Class distribution:\n%s", y.value_counts().to_string())

    return X, y


def generate_eda_plots(df: pd.DataFrame) -> None:
    """Generate exploratory visualizations and save to results/plots."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating EDA visualizations...")

    # 1. Correlation heatmap (numerical features + binary target)
    plot_df = df.copy()
    plot_df["status_binary"] = (plot_df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int)
    numerical_cols = config.NUMERICAL_FEATURES + ["status_binary"]

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = plot_df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap — Numerical Features & Placement Status")
    plt.tight_layout()
    fig.savefig(
        config.PLOTS_DIR / f"correlation_heatmap.{config.FIGURE_FORMAT}",
        dpi=config.PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2. Placement distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    status_counts = df[config.TARGET_COLUMN].value_counts()
    colors = sns.color_palette("Set2", len(status_counts))
    ax.bar(status_counts.index, status_counts.values, color=colors)
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Number of Students")
    ax.set_title("Placement Status Distribution")
    for idx, count in enumerate(status_counts.values):
        ax.text(idx, count + 1, str(count), ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(
        config.PLOTS_DIR / f"placement_distribution.{config.FIGURE_FORMAT}",
        dpi=config.PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. Placement rate by category
    categorical_cols = config.CATEGORICAL_FEATURES
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(categorical_cols):
        rate = (
            df.groupby(col)[config.TARGET_COLUMN]
            .apply(lambda s: (s == config.POSITIVE_CLASS).mean() * 100)
            .sort_values(ascending=False)
        )
        rate.plot(kind="bar", ax=axes[idx], color="steelblue")
        axes[idx].set_title(f"Placement Rate by {col}")
        axes[idx].set_ylabel("Placement Rate (%)")
        axes[idx].set_xlabel(col)
        axes[idx].tick_params(axis="x", rotation=45)

    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(
        config.PLOTS_DIR / f"placement_rate_by_category.{config.FIGURE_FORMAT}",
        dpi=config.PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. Score distribution boxplots
    fig, axes = plt.subplots(1, len(config.NUMERICAL_FEATURES), figsize=(16, 5))
    if len(config.NUMERICAL_FEATURES) == 1:
        axes = [axes]

    for ax, col in zip(axes, config.NUMERICAL_FEATURES):
        sns.boxplot(
            data=df,
            x=config.TARGET_COLUMN,
            y=col,
            hue=config.TARGET_COLUMN,
            ax=ax,
            palette="Set2",
            legend=False,
        )
        ax.set_title(f"{col} by Placement Status")
        ax.set_xlabel("Status")

    plt.tight_layout()
    fig.savefig(
        config.PLOTS_DIR / f"score_distribution_boxplots.{config.FIGURE_FORMAT}",
        dpi=config.PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    logger.info("EDA plots saved to %s", config.PLOTS_DIR)


def train_baseline_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train and save the baseline SVM pipeline."""
    logger.info("Training baseline SVM...")
    pipeline = build_svm_pipeline(get_baseline_svm())
    pipeline.fit(X_train, y_train)

    model_path = config.MODELS_DIR / config.BASELINE_MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    logger.info("Baseline model saved to %s", model_path)
    return pipeline


def train_tuned_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, dict]:
    """Train SVM with GridSearchCV hyperparameter tuning."""
    logger.info("Starting GridSearchCV hyperparameter tuning...")
    pipeline = build_svm_pipeline(get_baseline_svm())

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=get_tuning_param_grid(),
        cv=config.CV_FOLDS,
        scoring=get_cv_scorer(),
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Best parameters: %s", grid_search.best_params_)
    logger.info("Best CV F1 score: %.4f", grid_search.best_score_)

    best_pipeline = grid_search.best_estimator_
    model_path = config.MODELS_DIR / config.TUNED_MODEL_FILENAME
    joblib.dump(best_pipeline, model_path)
    logger.info("Tuned model saved to %s", model_path)

    tuning_results = {
        "best_params": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
        "cv_folds": config.CV_FOLDS,
        "scoring": config.CV_SCORING,
    }

    tuning_path = config.METRICS_DIR / "gridsearch_results.json"
    with tuning_path.open("w", encoding="utf-8") as file:
        json.dump(tuning_results, file, indent=2)

    return best_pipeline, tuning_results


def write_analysis_report(
    baseline_results: dict,
    tuned_results: dict,
    tuning_results: dict,
    shap_interpretation: str,
) -> None:
    """Write comprehensive SVM analysis report to reports/svm_analysis.md."""
    report_path = config.REPORTS_DIR / "svm_analysis.md"

    tuned_metrics = tuned_results["metrics"]
    baseline_metrics = baseline_results["metrics"]

    content = f"""# SVM Analysis Report — PlacePredictor

**Student:** Muntasir Bin Kashem  
**ID:** 0112331000  
**Branch:** svm_0112331000  

---

## Executive Summary

This report documents the complete Support Vector Machine (SVM) implementation for
student placement prediction. Both a baseline SVM and a GridSearchCV-tuned SVM were
trained with SMOTE-based class imbalance handling applied exclusively on training data.

### Tuned Model Performance (Test Set)

| Metric | Value |
|---|---|
| Accuracy | {tuned_metrics['accuracy']:.4f} |
| Precision | {tuned_metrics['precision']:.4f} |
| Recall | {tuned_metrics['recall']:.4f} |
| F1-Score | {tuned_metrics['f1_score']:.4f} |
| AUC | {tuned_metrics.get('auc', float('nan')):.4f} |

### Baseline vs Tuned Comparison

| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | {baseline_metrics['accuracy']:.4f} | {tuned_metrics['accuracy']:.4f} |
| Precision | {baseline_metrics['precision']:.4f} | {tuned_metrics['precision']:.4f} |
| Recall | {baseline_metrics['recall']:.4f} | {tuned_metrics['recall']:.4f} |
| F1-Score | {baseline_metrics['f1_score']:.4f} | {tuned_metrics['f1_score']:.4f} |

### Best Hyperparameters (GridSearchCV)

```json
{json.dumps(tuning_results['best_params'], indent=2)}
```

**Best CV F1 Score:** {tuning_results['best_cv_f1']:.4f}

---

{baseline_results['interpretation']}

---

{tuned_results['interpretation']}

---

{shap_interpretation}

---

## Classification Reports

### Baseline SVM

```
{json.dumps(baseline_results['classification_report'], indent=2)}
```

### Tuned SVM

```
{json.dumps(tuned_results['classification_report'], indent=2)}
```

---

## Generated Artifacts

| Artifact | Path |
|---|---|
| Baseline model | `results/models/{config.BASELINE_MODEL_FILENAME}` |
| Tuned model | `results/models/{config.TUNED_MODEL_FILENAME}` |
| Metrics | `results/metrics/` |
| Predictions | `results/predictions/` |
| Plots | `results/plots/` |
"""

    report_path.write_text(content, encoding="utf-8")
    logger.info("Analysis report saved to %s", report_path)


def main() -> None:
    """Execute the full SVM training, evaluation, and reporting pipeline."""
    try:
        ensure_directories()

        X, y = load_data()
        full_df = X.copy()
        full_df[config.TARGET_COLUMN] = y
        generate_eda_plots(full_df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )
        logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

        baseline_model = train_baseline_svm(X_train, y_train)
        tuned_model, tuning_results = train_tuned_svm(X_train, y_train)

        baseline_results = run_full_evaluation(
            baseline_model,
            X_test,
            y_test,
            model_name="Baseline SVM",
            prefix="baseline_svm",
        )
        tuned_results = run_full_evaluation(
            tuned_model,
            X_test,
            y_test,
            model_name="Tuned SVM",
            prefix="tuned_svm",
        )

        importance_df = run_shap_analysis(
            tuned_model,
            X_train,
            X_test,
        )
        shap_interpretation = generate_shap_interpretation(importance_df)

        write_analysis_report(
            baseline_results,
            tuned_results,
            tuning_results,
            shap_interpretation,
        )

        logger.info("SVM pipeline completed successfully.")

    except Exception as exc:
        logger.exception("SVM pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
