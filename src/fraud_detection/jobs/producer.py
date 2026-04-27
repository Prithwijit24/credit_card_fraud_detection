from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from fraud_detection.config import load_config
from fraud_detection.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish sample transaction events to Kafka.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--records", type=int, default=2500, help="Number of records to publish.")
    parser.add_argument("--delay-seconds", type=float, default=0.05, help="Delay between records.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    try:
        from kafka import KafkaProducer
    except ImportError as exc:
        raise SystemExit(
            "Install the stream dependencies with `pip install -e .[stream]`."
        ) from exc

    topic = config.streaming["kafka_topic"]
    bootstrap_servers = config.streaming["kafka_bootstrap_servers"]
    source_path = Path(config.resolve_path(config.data["test_path"]))
    LOGGER.info("Publishing records from %s to %s", source_path, topic)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= args.records:
                break
            row["row_id"] = int(row.pop("") or index)
            row["amt"] = float(row["amt"])
            row["lat"] = float(row["lat"])
            row["long"] = float(row["long"])
            row["city_pop"] = int(row["city_pop"])
            row["unix_time"] = int(row["unix_time"])
            row["merch_lat"] = float(row["merch_lat"])
            row["merch_long"] = float(row["merch_long"])
            row["is_fraud"] = int(row["is_fraud"])
            producer.send(topic, row)
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()
