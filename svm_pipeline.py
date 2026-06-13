"""SVM preprocessing pipeline and model factory for PlacePredictor."""

from __future__ import annotations

import logging
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import config

logger = logging.getLogger(__name__)


def get_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer for categorical and numerical features.

    Categorical features are one-hot encoded; numerical features are scaled.
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
            ("num", numerical_transformer, config.NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_baseline_svm() -> SVC:
    """Return a baseline SVC model with project-specified hyperparameters."""
    return SVC(**config.SVM_BASELINE_PARAMS)


def build_svm_pipeline(classifier: SVC | None = None) -> ImbPipeline:
    """
    Build a full training pipeline: preprocess -> SMOTE -> SVC.

    SMOTE is applied only during fit (training folds), preventing leakage
    onto validation or test data when used inside cross-validation.
    """
    if classifier is None:
        classifier = get_baseline_svm()

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )
    return pipeline


def get_tuning_param_grid() -> dict[str, list[Any]]:
    """Return the hyperparameter grid for GridSearchCV."""
    return config.SVM_PARAM_GRID.copy()


def get_feature_names_from_pipeline(pipeline: ImbPipeline) -> list[str]:
    """
    Extract transformed feature names from a fitted pipeline.

    Combines one-hot encoded categorical names with numerical feature names.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat" and hasattr(transformer, "get_feature_names_out"):
            cat_names = transformer.get_feature_names_out(config.CATEGORICAL_FEATURES)
            feature_names.extend(cat_names.tolist())
        elif name == "num":
            feature_names.extend(columns)

    return feature_names
