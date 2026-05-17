from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from fraud_detection.models.base import BaseFraudModel


class RandomForestModel(BaseFraudModel):
    name = "random_forest"
    role = "supervised baseline"
    notes = "Robust bagged-tree fallback and challenger."

    def build(self, **params: Any) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=int(params.get("num_trees", params.get("n_estimators", 300))),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else 16,
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
        )
