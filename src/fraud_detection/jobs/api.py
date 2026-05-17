from __future__ import annotations

import logging
import os
from typing import Any

from fraud_detection.config import load_config
from fraud_detection.feature_pipeline import FeaturePipeline
from fraud_detection.feature_store import DuckDBFeatureStore
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.pipeline.explainability import ExplainabilityService
from fraud_detection.pipeline.model_components import available_components
from fraud_detection.pipeline.modeling import load_artifact
from fraud_detection.schemas import frame_from_records

LOGGER = logging.getLogger(__name__)

app_state = {}


def score_payload(
    payload: dict[str, Any],
    model: Any,
    threshold: float,
    schema_version: str = "unknown",
    metadata: dict[str, Any] | None = None,
    explainer: ExplainabilityService | None = None,
) -> dict[str, Any]:
    df = frame_from_records([payload])
    feature_pipeline = FeaturePipeline.from_metadata(metadata)
    features_df = feature_pipeline.transform(df)
    if features_df.empty:
        raise ValueError("Transaction failed data quality validation")

    probability = float(model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1][0])
    risk_band = (
        "critical" if probability >= threshold else ("elevated" if probability >= 0.5 else "normal")
    )
    explanation = (explainer or ExplainabilityService.from_metadata(model, metadata)).explain(
        features_df[FEATURE_COLUMNS],
        probability,
    )
    return {
        "fraud_probability": probability,
        "risk_band": risk_band,
        "threshold": threshold,
        "top_reasons": explanation["top_reasons"],
        "explanation_methods": explanation["explanation_methods"],
        "feature_contributions": explanation["feature_contributions"],
        "schema_version": schema_version,
    }


def get_app() -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Credit Card Fraud Detection API")

    @app.on_event("startup")
    def startup_event() -> None:
        config_path = os.getenv("FRAUD_CONFIG", "base")
        config = load_config(config_path)
        model_path = config.resolve_path(config.data["model_dir"])

        LOGGER.info("Loading model from %s", model_path)
        artifact = load_artifact(model_path)
        app_state["model"] = artifact.model
        app_state["metadata"] = artifact.metadata
        app_state["explainer"] = ExplainabilityService.from_metadata(
            artifact.model,
            artifact.metadata,
        )
        app_state["config"] = config

    @app.post("/score")
    async def score_transaction(request: Request) -> dict[str, Any] | JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        model = app_state["model"]
        config = app_state["config"]
        metadata = app_state.get("metadata", {})
        explainer = app_state.get("explainer")
        threshold = float(config.streaming["fraud_probability_threshold"])

        try:
            return score_payload(
                payload,
                model,
                threshold,
                schema_version=str(metadata.get("schema_version", "unknown")),
                metadata=dict(metadata),
                explainer=explainer,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            LOGGER.exception("Scoring failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/score/batch")
    async def score_batch(request: Request) -> dict[str, list[dict[str, Any]]]:
        payload = await request.json()
        if not isinstance(payload, list):
            return {"scores": []}
        model = app_state["model"]
        config = app_state["config"]
        metadata = dict(app_state.get("metadata", {}))
        explainer = app_state.get("explainer")
        threshold = float(config.streaming["fraud_probability_threshold"])
        scores = [
            score_payload(
                item,
                model,
                threshold,
                schema_version=str(metadata.get("schema_version", "unknown")),
                metadata=metadata,
                explainer=explainer,
            )
            for item in payload
            if isinstance(item, dict)
        ]
        return {"scores": scores}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "fraud-api"}

    @app.get("/metadata")
    async def metadata() -> dict[str, Any]:
        return {
            "artifact": app_state.get("metadata", {}),
            "model_components": available_components(),
        }

    @app.post("/feature-store/append")
    async def append_feature_store(request: Request) -> dict[str, int]:
        payload = await request.json()
        records = payload if isinstance(payload, list) else [payload]
        config = app_state["config"]
        feature_store = config.raw.get("feature_store", {})
        if not feature_store.get("enabled", False):
            return {"rows": 0}
        df = frame_from_records(records)
        features = FeaturePipeline.from_metadata(dict(app_state.get("metadata", {}))).transform(df)
        store = DuckDBFeatureStore(config.resolve_path(feature_store["path"]))
        return {"rows": store.append_features(features)}

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the synchronous Fraud Detection API.")
    parser.add_argument(
        "--config",
        default="base",
        help="Config name from fraud_detection/configs or explicit .yml path.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to.")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    args = parser.parse_args()

    import uvicorn

    os.environ["FRAUD_CONFIG"] = args.config

    uvicorn.run(
        "fraud_detection.jobs.api:get_app",
        host=args.host,
        port=args.port,
        factory=True,
    )


if __name__ == "__main__":
    main()
