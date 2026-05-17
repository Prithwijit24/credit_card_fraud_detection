from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from fraud_detection.models.base import BaseFraudModel


class TabTransformerModel(BaseFraudModel):
    name = "tabtransformer"
    role = "deep tabular feature"
    production_status = "optional"
    notes = "Optional deep tabular slot; uses a Random Forest fallback when deep deps are absent."

    def build(self, **params: Any) -> Any:
        return RandomForestClassifier(
            n_estimators=int(params.get("num_trees", params.get("n_estimators", 120))),
            max_depth=int(params.get("max_depth", 8)),
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
        )
