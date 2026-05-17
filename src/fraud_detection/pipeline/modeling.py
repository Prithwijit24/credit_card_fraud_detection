from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_detection.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from fraud_detection.models import build_model

MODEL_FILENAME = "model.joblib"
LOGGER = logging.getLogger(__name__)
TREE_PARAM_MODELS = {
    "xgboost",
    "random_forest",
    "lightgbm",
    "catboost",
    "stacking_ensemble",
    "tabtransformer",
}
ANOMALY_FEATURE_MODELS = {"isolation_forest", "one_class_svm", "autoencoder"}
STACKED_ANOMALY_MODELS = ["isolation_forest", "one_class_svm", "autoencoder", "tabtransformer"]
STACKED_BOOSTER_MODELS = ["xgboost", "lightgbm"]
STACKED_META_MODEL = "catboost"


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    metadata: dict[str, Any]


@dataclass
class StackedFraudModel:
    anomaly_models: dict[str, Pipeline]
    booster_models: dict[str, Pipeline]
    meta_model: Any
    anomaly_score_columns: list[str]
    booster_score_columns: list[str]

    def _append_scores(self, frame: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
        augmented = frame.copy()
        for name, model in models.items():
            augmented[f"{name}_score"] = _model_score(model, augmented)
        return augmented

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        with_anomaly_scores = self._append_scores(frame, self.anomaly_models)
        with_booster_scores = self._append_scores(with_anomaly_scores, self.booster_models)
        meta_frame = with_booster_scores[self.booster_score_columns]
        if hasattr(self.meta_model, "predict_proba"):
            return np.asarray(self.meta_model.predict_proba(meta_frame), dtype=float)
        score = np.asarray(self.meta_model.predict(meta_frame), dtype=float)
        return np.column_stack([1.0 - score, score])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(extra_numeric_columns: list[str] | None = None) -> ColumnTransformer:
    numeric_columns = [*NUMERIC_COLUMNS, *(extra_numeric_columns or [])]
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", _one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, numeric_columns),
        ],
        remainder="drop",
    )


def build_training_pipeline(
    seed: int,
    max_bins: int,
    max_depth: int,
    num_trees: int,
    model_name: str = "xgboost",
    extra_numeric_columns: list[str] | None = None,
) -> Pipeline:
    del max_bins
    classifier = build_model(
        model_name,
        seed=seed,
        max_depth=max_depth,
        num_trees=num_trees,
        min_samples_leaf=10,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(extra_numeric_columns=extra_numeric_columns)),
            ("classifier", classifier),
        ]
    )


def build_cv_pipeline(
    seed: int,
    max_bins: int,
    max_depths: list[int],
    num_trees_list: list[int],
    model_name: str = "xgboost",
) -> GridSearchCV:
    del max_bins
    pipeline = build_training_pipeline(
        seed=seed,
        max_bins=0,
        max_depth=max_depths[0],
        num_trees=num_trees_list[0],
        model_name=model_name,
    )
    param_grid: dict[str, list[Any]] = {}
    if model_name in TREE_PARAM_MODELS:
        param_grid = {
            "classifier__max_depth": max_depths,
            "classifier__n_estimators": num_trees_list,
        }
    if isinstance(pipeline.named_steps["classifier"], RandomForestClassifier):
        param_grid["classifier__min_samples_leaf"] = [5, 10]
    if not param_grid:
        param_grid = {"classifier__C": [0.5, 1.0, 2.0]}
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        refit=True,
    )


def _fit_pipeline(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    label: pd.Series,
    sample_weight: pd.Series | None,
    unsupervised: bool = False,
) -> Pipeline:
    if unsupervised:
        pipeline.fit(frame)
        return pipeline
    if sample_weight is None:
        pipeline.fit(frame, label)
        return pipeline
    try:
        pipeline.fit(frame, label, classifier__sample_weight=sample_weight)
    except TypeError:
        pipeline.fit(frame, label)
    return pipeline


def _model_score(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(frame))[:, 1]
    if hasattr(model, "decision_function"):
        raw_score = np.asarray(model.decision_function(frame), dtype=float)
    else:
        raw_score = np.asarray(model.predict(frame), dtype=float)
    min_value = float(np.min(raw_score)) if raw_score.size else 0.0
    max_value = float(np.max(raw_score)) if raw_score.size else 0.0
    if max_value == min_value:
        return np.zeros_like(raw_score, dtype=float)
    return (raw_score - min_value) / (max_value - min_value)


def _fit_meta_model(meta_model: Any, meta_frame: pd.DataFrame, labels: pd.Series) -> Any:
    if labels.nunique() < 2:
        fallback = DummyClassifier(strategy="prior")
        fallback.fit(meta_frame, labels)
        return fallback
    if all(meta_frame[column].nunique(dropna=False) <= 1 for column in meta_frame.columns):
        fallback = LogisticRegression(max_iter=1000, class_weight="balanced")
        fallback.fit(meta_frame, labels)
        return fallback
    try:
        meta_model.fit(meta_frame, labels)
        return meta_model
    except Exception as exc:
        LOGGER.warning("Meta-model fit failed; falling back to logistic regression: %s", exc)
        fallback = LogisticRegression(max_iter=1000, class_weight="balanced")
        fallback.fit(meta_frame, labels)
        return fallback


def build_stacked_model(
    train_df: pd.DataFrame,
    seed: int,
    max_depth: int,
    num_trees: int,
    label_column: str = "isFraud",
    sample_weight_column: str = "class_weight",
) -> StackedFraudModel:
    labels = train_df[label_column].astype(int)
    sample_weight = train_df.get(sample_weight_column)

    anomaly_models: dict[str, Pipeline] = {}
    anomaly_score_columns: list[str] = []
    for model_name in STACKED_ANOMALY_MODELS:
        pipeline = build_training_pipeline(
            seed=seed,
            max_bins=0,
            max_depth=max_depth,
            num_trees=num_trees,
            model_name=model_name,
        )
        _fit_pipeline(
            pipeline,
            train_df,
            labels,
            sample_weight,
            unsupervised=model_name in ANOMALY_FEATURE_MODELS,
        )
        anomaly_models[model_name] = pipeline
        anomaly_score_columns.append(f"{model_name}_score")

    augmented = train_df.copy()
    for model_name, model in anomaly_models.items():
        augmented[f"{model_name}_score"] = _model_score(model, augmented)

    booster_models: dict[str, Pipeline] = {}
    booster_score_columns: list[str] = []
    for model_name in STACKED_BOOSTER_MODELS:
        pipeline = build_training_pipeline(
            seed=seed,
            max_bins=0,
            max_depth=max_depth,
            num_trees=num_trees,
            model_name=model_name,
            extra_numeric_columns=anomaly_score_columns,
        )
        _fit_pipeline(pipeline, augmented, labels, sample_weight)
        booster_models[model_name] = pipeline
        augmented[f"{model_name}_score"] = _model_score(pipeline, augmented)
        booster_score_columns.append(f"{model_name}_score")

    meta_model = build_model(
        STACKED_META_MODEL,
        seed=seed,
        max_depth=max_depth,
        num_trees=num_trees,
    )
    meta_model = _fit_meta_model(meta_model, augmented[booster_score_columns], labels)
    return StackedFraudModel(
        anomaly_models=anomaly_models,
        booster_models=booster_models,
        meta_model=meta_model,
        anomaly_score_columns=anomaly_score_columns,
        booster_score_columns=booster_score_columns,
    )


def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)
    artifact_path = path / MODEL_FILENAME if path.is_dir() else path
    artifact = joblib.load(artifact_path)
    if isinstance(artifact, ModelArtifact):
        return artifact.model
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


def load_artifact(model_path: str | Path) -> ModelArtifact:
    path = Path(model_path)
    artifact_path = path / MODEL_FILENAME if path.is_dir() else path
    artifact = joblib.load(artifact_path)
    if isinstance(artifact, ModelArtifact):
        return artifact
    if isinstance(artifact, dict) and "model" in artifact:
        return ModelArtifact(
            model=artifact["model"],
            metadata=dict(artifact.get("metadata", {})),
        )
    return ModelArtifact(model=artifact, metadata={})


def save_model(model: Any, model_path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(model_path)
    artifact_path = path / MODEL_FILENAME if path.suffix == "" else path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ModelArtifact(model=model, metadata=metadata or {}), artifact_path)
    return artifact_path


def required_input_columns() -> list[str]:
    return [*FEATURE_COLUMNS, "isFraud"]
