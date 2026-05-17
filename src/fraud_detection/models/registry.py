from __future__ import annotations

from typing import Any

from fraud_detection.models.autoencoder import AutoencoderModel
from fraud_detection.models.base import BaseFraudModel
from fraud_detection.models.catboost import CatBoostModel
from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.models.lightgbm import LightGBMModel
from fraud_detection.models.logistic_regression import LogisticRegressionModel
from fraud_detection.models.one_class_svm import OneClassSVMModel
from fraud_detection.models.random_forest import RandomForestModel
from fraud_detection.models.stacking_ensemble import StackingEnsembleModel
from fraud_detection.models.tabtransformer import TabTransformerModel
from fraud_detection.models.xgboost import XGBoostModel

MODEL_REGISTRY: dict[str, type[BaseFraudModel]] = {
    model.name: model
    for model in [
        LogisticRegressionModel,
        RandomForestModel,
        XGBoostModel,
        LightGBMModel,
        CatBoostModel,
        IsolationForestModel,
        OneClassSVMModel,
        AutoencoderModel,
        TabTransformerModel,
        StackingEnsembleModel,
    ]
}


def model_names() -> list[str]:
    return list(MODEL_REGISTRY)


def get_model_builder(name: str, seed: int = 42) -> BaseFraudModel:
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(model_names())
        raise ValueError(f"Unknown model component: {name}. Available: {available}") from exc
    return model_cls(seed=seed)


def build_model(name: str, seed: int = 42, **params: Any) -> Any:
    return get_model_builder(name, seed=seed).build(**params)


def available_components() -> list[dict[str, str]]:
    return [get_model_builder(name).metadata() for name in model_names()]
