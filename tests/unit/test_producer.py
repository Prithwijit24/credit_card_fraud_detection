from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from fraud_detection.jobs.producer import main
from tests.fixtures import notebook_transaction


@patch("fraud_detection.jobs.producer.parse_args")
@patch("fraud_detection.jobs.producer.load_config")
@patch("fraud_detection.jobs.producer.read_transactions")
def test_producer_main(
    mock_read_transactions,
    mock_load_config,
    mock_parse_args,
    project_root,
):
    mock_args = MagicMock()
    mock_args.config = "base"
    mock_args.records = 1
    mock_args.delay_seconds = 0.0
    mock_parse_args.return_value = mock_args

    mock_config = MagicMock()
    mock_config.streaming = {
        "kafka_topic": "fraud_events",
        "kafka_bootstrap_servers": "localhost:9092",
    }
    mock_config.data = {"test_path": "dummy.csv"}
    mock_config.resolve_path.return_value = project_root / "dummy.csv"
    mock_load_config.return_value = mock_config
    mock_read_transactions.return_value = pd.DataFrame([notebook_transaction()])

    mock_producer_instance = MagicMock()
    mock_kafka_module = MagicMock()
    mock_kafka_module.KafkaProducer.return_value = mock_producer_instance

    with patch.dict("sys.modules", {"kafka": mock_kafka_module}):
        main()

    assert mock_producer_instance.send.call_count == 1
    args, kwargs = mock_producer_instance.send.call_args
    assert args[0] == "fraud_events"
    payload = args[1]
    assert payload["accountNumber"] == "10001"
    assert payload["transactionAmount"] == 125.5

    mock_producer_instance.flush.assert_called_once()
    mock_producer_instance.close.assert_called_once()
