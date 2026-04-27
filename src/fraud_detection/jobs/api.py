from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from fraud_detection.config import load_config
from fraud_detection.pipeline.modeling import load_model
from fraud_detection.features import add_derived_features, sanitize_input_columns
from fraud_detection.pipeline.validation import validate_transactions
from fraud_detection.spark import build_spark_session
from fraud_detection.schemas import TRANSACTION_SCHEMA

LOGGER = logging.getLogger(__name__)

app_state = {}


def get_app() -> FastAPI:
    app = FastAPI(title="Credit Card Fraud Detection API")

    @app.on_event("startup")
    def startup_event():
        config_path = os.getenv("FRAUD_CONFIG", "configs/base.yaml")
        config = load_config(config_path)
        spark = build_spark_session(config, "fraud-api")
        model_path = str(config.resolve_path(config.data["model_dir"]))
        
        LOGGER.info("Loading model from %s", model_path)
        model = load_model(model_path)

        app_state["spark"] = spark
        app_state["model"] = model
        app_state["config"] = config

    @app.on_event("shutdown")
    def shutdown_event():
        if "spark" in app_state:
            app_state["spark"].stop()

    @app.post("/score")
    async def score_transaction(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        spark = app_state["spark"]
        model = app_state["model"]
        config = app_state["config"]
        threshold = float(config.streaming["fraud_probability_threshold"])

        try:
            # Create a DataFrame from the single record
            df = spark.createDataFrame([payload], schema=TRANSACTION_SCHEMA)

            # Preprocess and validate
            valid_df = validate_transactions(sanitize_input_columns(df))
            if valid_df.count() == 0:
                return JSONResponse(
                    {"error": "Transaction failed data quality validation"}, status_code=400
                )

            features_df = add_derived_features(valid_df)
            scored_df = model.transform(features_df)

            # Extract probability
            result = scored_df.select("probability").collect()[0]
            prob = float(result["probability"][1])

            risk_band = "critical" if prob >= threshold else ("elevated" if prob >= 0.5 else "normal")

            return {
                "fraud_probability": prob,
                "risk_band": risk_band,
                "threshold": threshold,
            }
        except Exception as e:
            LOGGER.exception("Scoring failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the synchronous Fraud Detection API.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    args = parser.parse_args()

    os.environ["FRAUD_CONFIG"] = args.config

    uvicorn.run(
        "fraud_detection.jobs.api:get_app", 
        host=args.host, 
        port=args.port, 
        factory=True
    )


if __name__ == "__main__":
    main()
