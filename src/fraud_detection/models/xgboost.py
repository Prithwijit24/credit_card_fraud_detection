from __future__ import annotations

from typing import Any

from fraud_detection.models.base import BaseFraudModel
from fraud_detection.models.random_forest import RandomForestModel


class XGBoostModel(BaseFraudModel):
    name = "xgboost"
    role = "champion candidate"
    notes = "Primary tabular model; falls back to Random Forest if xgboost is unavailable."

    def build(self, **params: Any) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return RandomForestModel(seed=self.seed).build(**params)

        return XGBClassifier(
            n_estimators=int(params.get("num_trees", params.get("n_estimators", 300))),
            max_depth=int(params.get("max_depth", 8)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=int(params.get("min_child_weight", 5)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=self.seed,
            n_jobs=-1,
        )
