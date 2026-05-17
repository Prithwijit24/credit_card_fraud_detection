from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)

EXPLAINABILITY_METADATA_KEY = "explainability"
DEFAULT_BACKGROUND_ROWS = 200


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    label: str
    value: float
    method: str
    direction: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": self.value,
            "method": self.method,
            "direction": self.direction,
        }


def _coerce_probability(probability: float | np.floating[Any]) -> float:
    return float(np.clip(float(probability), 0.0, 1.0))


def heuristic_reasons(features: pd.DataFrame | pd.Series, probability: float) -> list[str]:
    row = features.iloc[0] if isinstance(features, pd.DataFrame) else features
    reasons = []
    if row.get("cvv_mismatch", 0) >= 1:
        reasons.append("CVV mismatch")
    if row.get("cross_border", 0) >= 1:
        reasons.append("Cross-border transaction")
    if row.get("is_new_merchant", 0) >= 1:
        reasons.append("New merchant for account")
    if row.get("txn_count_24h", 0) >= 3:
        reasons.append("High 24h transaction velocity")
    if row.get("amount_to_limit_ratio", 0) >= 0.25:
        reasons.append("Large transaction relative to credit limit")
    if probability >= 0.5 and not reasons:
        reasons.append("Model score above elevated-risk threshold")
    return reasons[:3]


def build_explainability_metadata(
    train_df: pd.DataFrame,
    seed: int,
    feature_columns: list[str] | None = None,
    max_background_rows: int = DEFAULT_BACKGROUND_ROWS,
) -> dict[str, Any]:
    columns = feature_columns or FEATURE_COLUMNS
    available_columns = [column for column in columns if column in train_df.columns]
    background = train_df[available_columns].copy()
    if len(background) > max_background_rows:
        background = background.sample(n=max_background_rows, random_state=seed)
    background = background.replace({np.nan: None})
    return {
        "enabled": True,
        "feature_columns": available_columns,
        "background": background.to_dict(orient="records"),
        "background_rows": int(len(background)),
        "methods": ["shap", "lime"],
    }


class ExplainabilityService:
    def __init__(
        self,
        model: Any,
        metadata: dict[str, Any] | None = None,
        max_features: int = 5,
    ) -> None:
        self.model = model
        self.metadata = metadata or {}
        self.max_features = max_features
        explainability = self.metadata.get(EXPLAINABILITY_METADATA_KEY, {})
        self.feature_columns = list(explainability.get("feature_columns") or FEATURE_COLUMNS)
        self.background = self._background_frame(explainability)
        self._shap_explainer: Any | None = None
        self._lime_explainer: Any | None = None
        self._lime_categories = self._build_lime_categories()

    @classmethod
    def from_metadata(
        cls,
        model: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ExplainabilityService:
        return cls(model=model, metadata=metadata)

    def explain(
        self,
        features: pd.DataFrame,
        probability: float,
        max_reasons: int = 3,
    ) -> dict[str, Any]:
        probability = _coerce_probability(probability)
        contributions = [
            *self._shap_contributions(features),
            *self._lime_contributions(features),
        ]
        top_contributions = sorted(contributions, key=lambda item: abs(item.value), reverse=True)[
            : self.max_features
        ]
        top_reasons = self._reason_strings(features, probability, top_contributions, max_reasons)
        return {
            "top_reasons": top_reasons,
            "explanation_methods": sorted({item.method for item in top_contributions})
            or ["heuristic"],
            "feature_contributions": [item.as_dict() for item in top_contributions],
        }

    def explain_frame(
        self,
        scored_df: pd.DataFrame,
        probability_column: str = "fraud_probability",
    ) -> pd.DataFrame:
        explained = scored_df.copy()
        payloads = [
            self.explain(row.to_frame().T, float(row.get(probability_column, 0.0)))
            for _, row in explained.iterrows()
        ]
        explained["top_reasons"] = [payload["top_reasons"] for payload in payloads]
        explained["explanation_methods"] = [payload["explanation_methods"] for payload in payloads]
        explained["feature_contributions"] = [
            payload["feature_contributions"] for payload in payloads
        ]
        return explained

    def _background_frame(self, explainability: dict[str, Any]) -> pd.DataFrame:
        records = explainability.get("background") or []
        if not records:
            return pd.DataFrame(columns=self.feature_columns)
        return pd.DataFrame.from_records(records).reindex(columns=self.feature_columns)

    def _predict_probability(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame, pd.DataFrame):
            model_frame = frame.reindex(columns=self.feature_columns)
        else:
            model_frame = pd.DataFrame(frame, columns=self.feature_columns)
        return np.asarray(self.model.predict_proba(model_frame), dtype=float)[:, 1]

    def _shap_contributions(self, features: pd.DataFrame) -> list[FeatureContribution]:
        if self.background.empty:
            return []
        try:
            import shap

            if self._shap_explainer is None:
                self._shap_explainer = shap.Explainer(self._predict_probability, self.background)
            result = self._shap_explainer(features.reindex(columns=self.feature_columns).iloc[[0]])
            values = np.asarray(result.values)
            if values.ndim == 2:
                values = values[0]
            if values.ndim != 1:
                return []
            return [
                FeatureContribution(
                    feature=feature,
                    label=feature,
                    value=float(value),
                    method="shap",
                    direction="fraud" if value > 0 else "legit",
                )
                for feature, value in zip(self.feature_columns, values, strict=False)
                if np.isfinite(value)
            ]
        except ImportError:
            LOGGER.debug("SHAP is not installed; skipping SHAP explanations.")
        except Exception as exc:
            LOGGER.warning("SHAP explanation failed: %s", exc)
        return []

    def _lime_contributions(self, features: pd.DataFrame) -> list[FeatureContribution]:
        if self.background.empty:
            return []
        try:
            import lime.lime_tabular

            encoded_background = self._encode_lime_frame(self.background)
            if self._lime_explainer is None:
                categorical_indices = [
                    self.feature_columns.index(column)
                    for column in CATEGORICAL_COLUMNS
                    if column in self.feature_columns
                ]
                self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=encoded_background,
                    feature_names=self.feature_columns,
                    class_names=["legit", "fraud"],
                    categorical_features=categorical_indices,
                    mode="classification",
                    discretize_continuous=True,
                    random_state=42,
                )
            encoded_row = self._encode_lime_frame(
                features.reindex(columns=self.feature_columns).iloc[[0]]
            )
            explanation = self._lime_explainer.explain_instance(
                data_row=encoded_row[0],
                predict_fn=self._lime_predict_proba,
                labels=(1,),
                num_features=self.max_features,
                num_samples=1000,
            )
            mapped_features = explanation.as_map().get(1, [])
            labeled_weights = explanation.as_list(label=1)
            contributions = []
            for index, ((feature_index, weight), (label, _)) in enumerate(
                zip(mapped_features, labeled_weights, strict=False)
            ):
                if not np.isfinite(weight):
                    continue
                feature = (
                    self.feature_columns[feature_index]
                    if feature_index < len(self.feature_columns)
                    else f"feature_{index}"
                )
                contributions.append(
                    FeatureContribution(
                        feature=feature,
                        label=str(label),
                        value=float(weight),
                        method="lime",
                        direction="fraud" if weight > 0 else "legit",
                    )
                )
            return contributions
        except ImportError:
            LOGGER.debug("LIME is not installed; skipping LIME explanations.")
        except Exception as exc:
            LOGGER.warning("LIME explanation failed: %s", exc)
        return []

    def _build_lime_categories(self) -> dict[str, list[Any]]:
        categories: dict[str, list[Any]] = {}
        if self.background.empty:
            return categories
        for column in CATEGORICAL_COLUMNS:
            if column in self.background:
                values = self.background[column].fillna("unknown").astype(str)
                categories[column] = sorted(values.unique().tolist()) or ["unknown"]
        return categories

    def _encode_lime_frame(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = frame.reindex(columns=self.feature_columns).copy()
        for column, categories in self._lime_categories.items():
            if column not in encoded:
                continue
            lookup = {value: index for index, value in enumerate(categories)}
            encoded[column] = encoded[column].fillna("unknown").astype(str).map(lookup).fillna(0)
        for column in encoded.columns:
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(0.0)
        return encoded.to_numpy(dtype=float)

    def _decode_lime_frame(self, encoded: np.ndarray) -> pd.DataFrame:
        decoded = pd.DataFrame(encoded, columns=self.feature_columns)
        for column, categories in self._lime_categories.items():
            if column not in decoded:
                continue
            max_index = len(categories) - 1
            codes = np.rint(decoded[column].to_numpy(dtype=float)).clip(0, max_index).astype(int)
            decoded[column] = [categories[code] for code in codes]
        return decoded

    def _lime_predict_proba(self, encoded: np.ndarray) -> np.ndarray:
        decoded = self._decode_lime_frame(encoded)
        fraud_probability = self._predict_probability(decoded)
        return np.column_stack([1.0 - fraud_probability, fraud_probability])

    def _reason_strings(
        self,
        features: pd.DataFrame,
        probability: float,
        contributions: list[FeatureContribution],
        max_reasons: int,
    ) -> list[str]:
        reasons = [
            f"{item.feature} pushed score toward {item.direction}"
            for item in contributions
            if item.direction == "fraud"
        ][:max_reasons]
        if reasons:
            return reasons
        return heuristic_reasons(features, probability)[:max_reasons]
