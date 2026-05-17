from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from fraud_detection.models import available_components, build_model, model_names


def test_model_registry_exposes_child_models() -> None:
    names = model_names()

    assert "xgboost" in names
    assert "random_forest" in names
    assert "logistic_regression" in names
    assert all("name" in component for component in available_components())


def test_random_forest_child_model_builds_estimator() -> None:
    model = build_model("random_forest", seed=7, max_depth=3, num_trees=5)

    assert isinstance(model, RandomForestClassifier)
    assert model.max_depth == 3
    assert model.n_estimators == 5
