from __future__ import annotations

import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fraud_detection.config import load_config
from fraud_detection.features import FEATURE_COLUMNS, add_derived_features, sanitize_input_columns
from fraud_detection.pipeline.modeling import load_model
from fraud_detection.pipeline.validation import validate_transactions
from fraud_detection.schemas import frame_from_records

LOGGER = logging.getLogger(__name__)

app_state = {}


def score_payload(payload: dict[str, Any], model: Any, threshold: float) -> dict[str, float | str]:
    df = frame_from_records([payload])
    valid_df = validate_transactions(sanitize_input_columns(df))
    if valid_df.empty:
        raise ValueError("Transaction failed data quality validation")

    features_df = add_derived_features(valid_df)
    probability = float(model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1][0])
    risk_band = (
        "critical" if probability >= threshold else ("elevated" if probability >= 0.5 else "normal")
    )
    return {
        "fraud_probability": probability,
        "risk_band": risk_band,
        "threshold": threshold,
    }


def get_app() -> FastAPI:
    app = FastAPI(title="Credit Card Fraud Detection API")

    @app.on_event("startup")
    def startup_event() -> None:
        config_path = os.getenv("FRAUD_CONFIG", "configs/base.yaml")
        config = load_config(config_path)
        model_path = config.resolve_path(config.data["model_dir"])

        LOGGER.info("Loading model from %s", model_path)
        app_state["model"] = load_model(model_path)
        app_state["config"] = config

    @app.post("/score")
    async def score_transaction(request: Request) -> dict[str, float | str] | JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        model = app_state["model"]
        config = app_state["config"]
        threshold = float(config.streaming["fraud_probability_threshold"])

        try:
            return score_payload(payload, model, threshold)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            LOGGER.exception("Scoring failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the synchronous Fraud Detection API.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to.")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    args = parser.parse_args()

    os.environ["FRAUD_CONFIG"] = args.config

    uvicorn.run(
        "fraud_detection.jobs.api:get_app",
        host=args.host,
        port=args.port,
        factory=True,
    )


if __name__ == "__main__":
    main()
