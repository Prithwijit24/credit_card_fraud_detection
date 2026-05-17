from __future__ import annotations

from typing import Any

from sklearn.svm import OneClassSVM

from fraud_detection.models.base import BaseFraudModel


class OneClassSVMModel(BaseFraudModel):
    name = "one_class_svm"
    role = "anomaly baseline"
    notes = "Small-sample unsupervised anomaly detector."

    def build(self, **params: Any) -> OneClassSVM:
        return OneClassSVM(
            kernel=str(params.get("kernel", "rbf")),
            gamma=params.get("gamma", "scale"),
            nu=float(params.get("nu", 0.05)),
        )
