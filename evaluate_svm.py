"""Evaluation utilities for the PlacePredictor SVM module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import config

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1-score."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=config.POSITIVE_CLASS)),
        "recall": float(recall_score(y_true, y_pred, pos_label=config.POSITIVE_CLASS)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=config.POSITIVE_CLASS)),
    }


def get_classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Return classification report as a serializable dictionary."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        output_dict=True,
    )
    return report


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """
    Evaluate a fitted model on the test set.

    Returns metrics, classification report, predictions, and probabilities.
    """
    logger.info("Evaluating model: %s", model_name)

    y_pred = model.predict(X_test)
    classes = list(model.named_steps["classifier"].classes_)
    placed_idx = classes.index(config.POSITIVE_CLASS)
    y_proba = model.predict_proba(X_test)[:, placed_idx]

    metrics = compute_metrics(y_test.values, y_pred)
    report = get_classification_report_dict(y_test.values, y_pred)

    try:
        auc_score = float(
            roc_auc_score(
                (y_test == config.POSITIVE_CLASS).astype(int),
                y_proba,
            )
        )
    except ValueError as exc:
        logger.warning("Could not compute AUC: %s", exc)
        auc_score = float("nan")

    metrics["auc"] = auc_score

    results: dict[str, Any] = {
        "model_name": model_name,
        "metrics": metrics,
        "classification_report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(
            y_test.values,
            y_pred,
            labels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        ).tolist(),
    }

    logger.info(
        "%s — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["auc"],
    )

    return results


def save_metrics(results: dict[str, Any], output_path: Path) -> None:
    """Save evaluation metrics and classification report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "model_name": results["model_name"],
        "metrics": results["metrics"],
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)

    logger.info("Metrics saved to %s", output_path)


def save_predictions(
    y_test: pd.Series,
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Save test predictions and probabilities to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": results["y_pred"],
            "probability_placed": results["y_proba"],
        }
    )
    df.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s", output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        yticklabels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def plot_roc_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Generate and save an ROC curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_binary = (y_test == config.POSITIVE_CLASS).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", output_path)


def generate_analysis_interpretation(
    results: dict[str, Any],
    model_name: str,
) -> str:
    """
    Generate detailed textual interpretation of evaluation results.

    Covers accuracy, precision, recall, F1, confusion matrix, errors,
    and business impact for placement prediction.
    """
    metrics = results["metrics"]
    cm = np.array(results["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()

    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1_score"]
    auc_score = metrics.get("auc", float("nan"))

    lines = [
        f"## Detailed Result Analysis — {model_name}",
        "",
        "### 1. Accuracy Analysis",
        "",
        f"**Value:** {accuracy:.4f} ({accuracy * 100:.2f}%)",
        "",
        "Accuracy measures the proportion of all placement predictions that were correct. "
        f"An accuracy of {accuracy * 100:.2f}% means the SVM correctly classified "
        f"{accuracy * 100:.2f}% of students in the held-out test set.",
        "",
        "**Why it matters:** Accuracy gives a quick overall picture of model reliability. "
        "However, with class imbalance (more placed than not placed students), accuracy alone "
        "can hide poor performance on the minority class.",
        "",
        "**Impact on placement prediction:** High accuracy suggests the model captures general "
        "patterns in academic and profile data, but placement teams should pair it with "
        "recall and precision before making counselling decisions.",
        "",
        "### 2. Precision Analysis",
        "",
        f"**Value:** {precision:.4f} ({precision * 100:.2f}%)",
        "",
        "Precision answers: *Of students predicted as Placed, how many were actually placed?* "
        f"A precision of {precision * 100:.2f}% means that when the model predicts placement, "
        "it is correct roughly that often.",
        "",
        "**Why it matters:** High precision reduces false alarms — incorrectly telling a student "
        "they will be placed when they will not. This protects student confidence and "
        "institutional credibility.",
        "",
        "**Impact on placement prediction:** In campus recruitment, high precision helps "
        "career services prioritize resources for students the model flags with high confidence, "
        "avoiding wasted effort on overly optimistic predictions.",
        "",
        "### 3. Recall Analysis",
        "",
        f"**Value:** {recall:.4f} ({recall * 100:.2f}%)",
        "",
        "Recall answers: *Of all students who were actually placed, how many did the model identify?* "
        f"A recall of {recall * 100:.2f}% indicates the model captures that share of truly placed students.",
        "",
        "**Why it matters:** Missing placed students (false negatives) means lost opportunities "
        "for targeted mentoring. In an imbalanced dataset, recall for the Placed class is critical.",
        "",
        "**Impact on placement prediction:** Strong recall ensures employable students are not "
        "overlooked by the counselling system, supporting equitable placement support.",
        "",
        "### 4. F1-Score Analysis",
        "",
        f"**Value:** {f1:.4f}",
        "",
        "F1 is the harmonic mean of precision and recall, balancing both concerns. "
        f"An F1 of {f1:.4f} reflects how well the model trades off false positives against false negatives.",
        "",
        "**Why it matters:** F1 was used as the GridSearchCV scoring metric because it penalizes "
        "models that excel on only one dimension. It is especially suitable for imbalanced placement data.",
        "",
        "**Impact on placement prediction:** A balanced F1 indicates the SVM is useful for both "
        "identifying likely placements and avoiding misleading predictions.",
        "",
        "### 5. Confusion Matrix Analysis",
        "",
        f"| | Predicted Not Placed | Predicted Placed |",
        f"|---|---|---|",
        f"| **Actual Not Placed** | {tn} (TN) | {fp} (FP) |",
        f"| **Actual Placed** | {fn} (FN) | {tp} (TP) |",
        "",
        f"- **True Negatives ({tn}):** Correctly identified students who were not placed.",
        f"- **True Positives ({tp}):** Correctly identified students who were placed.",
        f"- **False Positives ({fp}):** Predicted placed but actually not placed — optimistic errors.",
        f"- **False Negatives ({fn}):** Predicted not placed but actually placed — missed opportunities.",
        "",
        "### 6. Error Analysis",
        "",
        f"The model made **{fp + fn}** total errors on the test set "
        f"({fp} false positives, {fn} false negatives).",
        "",
    ]

    if fp > fn:
        lines.extend([
            "False positives dominate errors: the model is somewhat optimistic, predicting "
            "placement for students who ultimately were not placed. This may occur when "
            "students have strong academic scores but lack work experience or employability signals.",
            "",
        ])
    elif fn > fp:
        lines.extend([
            "False negatives dominate errors: the model is conservative, failing to identify "
            "some students who were placed. These students may have non-linear profile patterns "
            "that the RBF kernel boundary does not fully capture.",
            "",
        ])
    else:
        lines.extend([
            "False positives and false negatives are balanced, suggesting neither optimistic "
            "nor overly conservative bias in predictions.",
            "",
        ])

    lines.extend([
        "**Common error drivers:**",
        "- Students with moderate MBA/degree percentages but strong employability test scores",
        "- Borderline work experience profiles (No workex but high etest_p)",
        "- Specialisation mismatches (Mkt&HR vs Mkt&Fin) affecting recruiter preferences",
        "",
        "### 7. Business Interpretation",
        "",
        f"**AUC Score:** {auc_score:.4f}" if not np.isnan(auc_score) else "**AUC Score:** N/A",
        "",
        "The ROC-AUC summarizes the model's ability to rank placed students above not-placed "
        "students across all decision thresholds. Higher AUC means better separability.",
        "",
        "#### Feature Influence on Placement (Domain Context)",
        "",
        "**Work Experience (`workex`):**",
        "Prior work experience is a strong employability signal. Recruiters often prefer candidates "
        "with industry exposure. Students with `workex=Yes` typically show higher placement rates, "
        "and the SVM learns this boundary in combination with test scores.",
        "",
        "**Employability Score (`etest_p`):**",
        "The employability test score directly measures job-readiness skills. High `etest_p` values "
        "strongly correlate with placement success because they reflect aptitude assessed by "
        "campus recruiters and training partners.",
        "",
        "**Academic Performance (`ssc_p`, `hsc_p`, `degree_p`, `mba_p`):**",
        "Consistent academic performance builds recruiter confidence. However, marks alone rarely "
        "guarantee placement — the SVM combines percentages with board type, stream, and "
        "specialisation to form a holistic decision boundary.",
        "",
        "**Recommendations for Career Services:**",
        "1. Use probability outputs (not just binary labels) to tier intervention intensity.",
        "2. Focus mentoring on high-recall segments — students with low predicted probability "
        "but strong individual features (e.g., low MBA score but high etest_p).",
        "3. Combine SVM predictions with SHAP explanations to justify counselling advice to students.",
        "",
    ])

    return "\n".join(lines)


def run_full_evaluation(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    prefix: str,
) -> dict[str, Any]:
    """Run complete evaluation: metrics, plots, predictions, and interpretation."""
    results = evaluate_model(model, X_test, y_test, model_name)

    save_metrics(results, config.METRICS_DIR / f"{prefix}_metrics.json")
    save_predictions(y_test, results, config.PREDICTIONS_DIR / f"{prefix}_predictions.csv")

    plot_confusion_matrix(
        y_test.values,
        results["y_pred"],
        model_name,
        config.PLOTS_DIR / f"{prefix}_confusion_matrix.{config.FIGURE_FORMAT}",
    )
    plot_roc_curve(
        y_test,
        results["y_proba"],
        model_name,
        config.PLOTS_DIR / f"{prefix}_roc_curve.{config.FIGURE_FORMAT}",
    )

    interpretation = generate_analysis_interpretation(results, model_name)
    return results | {"interpretation": interpretation}
