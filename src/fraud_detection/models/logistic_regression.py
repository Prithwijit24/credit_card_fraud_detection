from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from fraud_detection.models.base import BaseFraudModel


class LogisticRegressionModel(BaseFraudModel):
    name = "logistic_regression"
    role = "calibrated baseline"
    notes = "Fast supervised baseline."

    def build(self, **params: Any) -> LogisticRegression:
        max_iter = int(params.get("max_iter", 1000))
        return LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=self.seed,
        )
