"""SHAP explainability analysis for the PlacePredictor SVM module."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline as ImbPipeline

import config
from models.svm_pipeline import get_feature_names_from_pipeline

logger = logging.getLogger(__name__)


def _transform_features(pipeline: ImbPipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the fitted preprocessor to raw features."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.transform(X)


def run_shap_analysis(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path | None = None,
    background_size: int = 50,
    explain_size: int = 100,
) -> pd.DataFrame:
    """
    Run SHAP analysis on the tuned SVM pipeline.

    Uses KernelExplainer with a background sample from training data.
    Generates summary plot and feature importance ranking.

    Parameters
    ----------
    pipeline : ImbPipeline
        Fitted pipeline containing preprocessor and classifier.
    X_train : pd.DataFrame
        Training features (for background distribution).
    X_test : pd.DataFrame
        Test features to explain.
    output_dir : Path, optional
        Directory for saving plots and CSV rankings.
    background_size : int
        Number of background samples for KernelExplainer.
    explain_size : int
        Number of test samples to explain (for performance).

    Returns
    -------
    pd.DataFrame
        Feature importance ranking sorted by mean |SHAP value|.
    """
    if output_dir is None:
        output_dir = config.PLOTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting SHAP analysis...")

    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_pipeline(pipeline)

    X_train_transformed = _transform_features(pipeline, X_train)
    X_test_transformed = _transform_features(pipeline, X_test)

    rng = np.random.default_rng(config.RANDOM_STATE)
    bg_indices = rng.choice(
        len(X_train_transformed),
        size=min(background_size, len(X_train_transformed)),
        replace=False,
    )
    explain_indices = rng.choice(
        len(X_test_transformed),
        size=min(explain_size, len(X_test_transformed)),
        replace=False,
    )

    background = X_train_transformed[bg_indices]
    X_explain = X_test_transformed[explain_indices]

    def predict_proba_wrapper(data: np.ndarray) -> np.ndarray:
        """Predict positive-class probability on preprocessed features."""
        return classifier.predict_proba(data)[:, 1]

    try:
        explainer = shap.KernelExplainer(predict_proba_wrapper, background)
        shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
    except Exception as exc:
        logger.error("SHAP KernelExplainer failed: %s", exc)
        raise

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    summary_path = output_dir / f"shap_summary_plot.{config.FIGURE_FORMAT}"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s", summary_path)

    # Bar plot for ranking
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    bar_path = output_dir / f"shap_feature_importance.{config.FIGURE_FORMAT}"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info("SHAP feature importance plot saved to %s", bar_path)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    ranking_path = config.METRICS_DIR / "shap_feature_importance.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(ranking_path, index=False)
    logger.info("SHAP feature ranking saved to %s", ranking_path)

    return importance_df


def generate_shap_interpretation(importance_df: pd.DataFrame) -> str:
    """Generate textual explanation of SHAP findings."""
    top_features = importance_df.head(10)

    lines = [
        "## SHAP Explainability Analysis",
        "",
        "SHAP (SHapley Additive exPlanations) quantifies how each feature pushes "
        "a student's predicted placement probability above or below the baseline.",
        "",
        "### Top Features Influencing Placement Prediction",
        "",
        "| Rank | Feature | Mean |SHAP| |",
        "|---|---|---|",
    ]

    for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['feature']} | {row['mean_abs_shap']:.4f} |"
        )

    lines.extend([
        "",
        "### Interpretation",
        "",
    ])

    top_feature = top_features.iloc[0]["feature"] if len(top_features) > 0 else "N/A"
    lines.extend([
        f"The most influential feature is **{top_feature}**. Features related to "
        "employability testing (`etest_p`), work experience (`workex`), and MBA performance "
        "(`mba_p`) typically appear among top contributors because recruiters weight "
        "job-readiness and specialization fit heavily.",
        "",
        "**Why predictions are made:**",
        "- High `etest_p` SHAP values push predictions toward *Placed* — strong employability scores.",
        "- `workex_Yes` typically increases placement probability — prior experience signals readiness.",
        "- Low `mba_p` or `degree_p` SHAP contributions may push toward *Not Placed* unless offset "
        "by strong test scores or experience.",
        "",
        "SHAP summary plots show directionality: red (high feature value) vs blue (low) indicates "
        "whether high values increase or decrease placement probability.",
        "",
    ])

    return "\n".join(lines)
