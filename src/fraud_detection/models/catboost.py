from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from fraud_detection.models.base import BaseFraudModel


class CatBoostModel(BaseFraudModel):
    name = "catboost"
    role = "challenger"
    production_status = "optional"
    notes = "Install model extras with `pip install -e .[models]`."

    def build(self, **params: Any) -> Any:
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.seed,
            )
        return CatBoostClassifier(
            iterations=int(params.get("num_trees", params.get("iterations", 800))),
            depth=int(params.get("max_depth", params.get("depth", 8))),
            learning_rate=float(params.get("learning_rate", 0.05)),
            random_seed=self.seed,
            thread_count=1,
            verbose=False,
        )
