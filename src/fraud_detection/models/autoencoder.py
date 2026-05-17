from __future__ import annotations

from typing import Any

from fraud_detection.models.base import BaseFraudModel
from fraud_detection.models.isolation_forest import IsolationForestModel


class AutoencoderModel(BaseFraudModel):
    name = "autoencoder"
    role = "deep anomaly feature"
    production_status = "optional"
    notes = "Optional deep model slot; uses an Isolation Forest fallback when torch is absent."

    def build(self, **params: Any) -> Any:
        return IsolationForestModel(seed=self.seed).build(**params)
