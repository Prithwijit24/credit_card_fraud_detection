from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_detection.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

MODEL_FILENAME = "model.joblib"


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
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
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def build_training_pipeline(seed: int, max_bins: int, max_depth: int, num_trees: int) -> Pipeline:
    del max_bins
    classifier = RandomForestClassifier(
        n_estimators=num_trees,
        max_depth=max_depth,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_cv_pipeline(
    seed: int,
    max_bins: int,
    max_depths: list[int],
    num_trees_list: list[int],
) -> GridSearchCV:
    del max_bins
    pipeline = build_training_pipeline(
        seed=seed,
        max_bins=0,
        max_depth=max_depths[0],
        num_trees=num_trees_list[0],
    )
    param_grid = {
        "classifier__max_depth": max_depths,
        "classifier__n_estimators": num_trees_list,
        "classifier__min_samples_leaf": [5, 10],
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        n_jobs=-1,
        refit=True,
    )


def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)
    artifact_path = path / MODEL_FILENAME if path.is_dir() else path
    return joblib.load(artifact_path)


def save_model(model: Any, model_path: str | Path) -> Path:
    path = Path(model_path)
    artifact_path = path / MODEL_FILENAME if path.suffix == "" else path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    return artifact_path


def required_input_columns() -> list[str]:
    return [*FEATURE_COLUMNS, "is_fraud"]
