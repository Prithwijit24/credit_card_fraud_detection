from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fraud_detection.config import AppConfig, load_config
from fraud_detection.features import FEATURE_COLUMNS, add_derived_features, sanitize_input_columns
from fraud_detection.logger import configure_logging
from fraud_detection.pipeline.metrics import score_frame
from fraud_detection.pipeline.modeling import load_model
from fraud_detection.pipeline.validation import validate_transactions
from fraud_detection.schemas import coerce_transaction_schema, frame_from_records

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run micro-batch fraud scoring.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def _parse_interval_seconds(value: str) -> float:
    parts = value.strip().split()
    if not parts:
        return 15.0
    number = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else "seconds"
    if unit.startswith("minute"):
        return number * 60
    if unit.startswith("hour"):
        return number * 3600
    return number


def _iter_kafka_batches(config: AppConfig) -> Iterable[pd.DataFrame]:
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:
        message = "Install the stream dependencies with `pip install -e .[stream]`."
        raise SystemExit(message) from exc

    interval_seconds = _parse_interval_seconds(config.streaming["trigger_interval"])
    consumer = KafkaConsumer(
        config.streaming["kafka_topic"],
        bootstrap_servers=config.streaming["kafka_bootstrap_servers"],
        auto_offset_reset=config.streaming["starting_offsets"],
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        consumer_timeout_ms=int(interval_seconds * 1000),
    )
    batch_size = int(config.streaming["max_offsets_per_trigger"])
    while True:
        records = []
        for message in consumer:
            records.append(message.value)
            if len(records) >= batch_size:
                break
        if records:
            yield frame_from_records(records)
        else:
            time.sleep(_parse_interval_seconds(config.streaming["trigger_interval"]))


def _iter_csv_batches(config: AppConfig) -> Iterable[pd.DataFrame]:
    landing_dir = config.resolve_path(config.data["landing_dir"])
    landing_dir.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    interval_seconds = _parse_interval_seconds(config.streaming["trigger_interval"])
    while True:
        batch_files = sorted(landing_dir.glob("*.csv"))
        unseen = [path for path in batch_files if path not in seen]
        if unseen:
            frames = [coerce_transaction_schema(pd.read_csv(path)) for path in unseen]
            seen.update(unseen)
            yield pd.concat(frames, ignore_index=True)
        else:
            time.sleep(interval_seconds)


def build_source_batches(config: AppConfig) -> Iterable[pd.DataFrame]:
    if config.streaming["source"] == "kafka":
        return _iter_kafka_batches(config)
    return _iter_csv_batches(config)


def write_outputs(scored_df: pd.DataFrame, config: AppConfig) -> None:
    threshold = float(config.streaming["fraud_probability_threshold"])
    scored_dir = config.resolve_path(config.data["scored_dir"])
    alerts_dir = config.resolve_path(config.data["alerts_dir"])
    scored_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir.mkdir(parents=True, exist_ok=True)

    enriched = scored_df.copy()
    enriched["risk_band"] = "normal"
    enriched.loc[enriched["fraud_probability"] >= 0.5, "risk_band"] = "elevated"
    enriched.loc[enriched["fraud_probability"] >= threshold, "risk_band"] = "critical"

    batch_id = int(time.time() * 1000)
    enriched.to_json(scored_dir / f"scored-{batch_id}.jsonl", orient="records", lines=True)
    alerts = enriched.loc[enriched["fraud_probability"] >= threshold]
    if not alerts.empty:
        alerts.to_json(alerts_dir / f"alerts-{batch_id}.jsonl", orient="records", lines=True)

    monitor = (
        enriched.groupby(["state", "risk_band"], dropna=False)["fraud_probability"]
        .agg(txn_count="size", avg_fraud_probability="mean")
        .reset_index()
    )
    LOGGER.info("Batch monitoring summary: %s", monitor.to_dict(orient="records"))


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    model_path = config.resolve_path(config.data["model_dir"])
    threshold = float(config.streaming["fraud_probability_threshold"])

    LOGGER.info("Loading model from %s", model_path)
    model = load_model(model_path)
    for raw_batch in build_source_batches(config):
        valid_batch = validate_transactions(sanitize_input_columns(raw_batch))
        if valid_batch.empty:
            LOGGER.warning("Skipping empty or invalid batch.")
            continue
        features = add_derived_features(valid_batch)
        scored = score_frame(model, features[FEATURE_COLUMNS], threshold=threshold)
        overlapping_columns = [column for column in scored.columns if column in features]
        raw_context = features.drop(columns=overlapping_columns)
        scored = pd.concat(
            [raw_context, scored],
            axis=1,
        )
        write_outputs(scored, config)


if __name__ == "__main__":
    main()
