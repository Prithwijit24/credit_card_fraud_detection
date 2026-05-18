from __future__ import annotations

from typing import Any

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection.models.base import BaseFraudModel
from fraud_detection.models.logistic_regression import LogisticRegressionModel
from fraud_detection.models.random_forest import RandomForestModel


class StackingEnsembleModel(BaseFraudModel):
    name = "stacking_ensemble"
    role = "ensemble"
    notes = "Out-of-fold base-model stack for challenger experiments."

    def build(self, **params: Any) -> StackingClassifier:
        return StackingClassifier(
            estimators=[
                ("rf", RandomForestModel(seed=self.seed).build(**params)),
                ("lr", LogisticRegressionModel(seed=self.seed).build()),
            ],
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            n_jobs=1,
        )
