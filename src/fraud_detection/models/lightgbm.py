from __future__ import annotations

from typing import Any

from fraud_detection.models.base import BaseFraudModel
from fraud_detection.models.random_forest import RandomForestModel


class LightGBMModel(BaseFraudModel):
    name = "lightgbm"
    role = "challenger"
    production_status = "optional"
    notes = "Install model extras with `pip install -e .[models]`."

    def build(self, **params: Any) -> Any:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return RandomForestModel(seed=self.seed).build(**params)
        return LGBMClassifier(
            n_estimators=int(params.get("num_trees", params.get("n_estimators", 800))),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else -1,
            learning_rate=float(params.get("learning_rate", 0.05)),
            random_state=self.seed,
            n_jobs=-1,
        )
