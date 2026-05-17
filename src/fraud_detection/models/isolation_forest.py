from __future__ import annotations

from typing import Any

from sklearn.ensemble import IsolationForest

from fraud_detection.models.base import BaseFraudModel


class IsolationForestModel(BaseFraudModel):
    name = "isolation_forest"
    role = "anomaly baseline"
    notes = "Unsupervised anomaly detector for challenger experiments."

    def build(self, **params: Any) -> IsolationForest:
        return IsolationForest(
            n_estimators=int(params.get("num_trees", params.get("n_estimators", 200))),
            contamination=params.get("contamination", "auto"),
            random_state=self.seed,
            n_jobs=-1,
        )
