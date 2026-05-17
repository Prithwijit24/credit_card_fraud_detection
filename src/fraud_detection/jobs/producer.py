from __future__ import annotations

import argparse
import json
import logging
import time

from fraud_detection.config import load_config
from fraud_detection.logger import configure_logging
from fraud_detection.schemas import read_transactions

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish sample transaction events to Kafka.")
    parser.add_argument(
        "--config",
        default="base",
        help="Config name from fraud_detection/configs or explicit .yml path.",
    )
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
    source_path = config.resolve_path(config.data["test_path"])
    LOGGER.info("Publishing records from %s to %s", source_path, topic)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
    )

    frame = read_transactions(source_path).head(args.records)
    for row in frame.to_dict(orient="records"):
        producer.send(topic, row)
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()
