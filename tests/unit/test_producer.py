from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

from fraud_detection.jobs.producer import main

CSV_SAMPLE = (
    ",trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,city,"
    "state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,"
    "is_fraud\n"
    "0,2020-01-01 00:00:00,123,abc,shopping,100.0,John,Doe,M,123 Main St,Anytown,"
    "CA,12345,34.0,-118.0,1000,Teacher,1980-01-01,txn1,1577836800,34.1,-118.1,0\n"
)


@patch("kafka.KafkaProducer")
@patch("fraud_detection.jobs.producer.parse_args")
@patch("fraud_detection.jobs.producer.load_config")
@patch("fraud_detection.jobs.producer.Path.open", new_callable=mock_open, read_data=CSV_SAMPLE)
def test_producer_main(
    mock_file,
    mock_load_config,
    mock_parse_args,
    mock_kafka_producer,
    project_root,
):
    mock_args = MagicMock()
    mock_args.config = str(project_root / "configs" / "base.yaml")
    mock_args.records = 1
    mock_args.delay_seconds = 0.0  # Speed up test
    mock_parse_args.return_value = mock_args

    mock_config = MagicMock()
    mock_config.streaming = {
        "kafka_topic": "fraud_events",
        "kafka_bootstrap_servers": "localhost:9092",
    }
    mock_config.data = {"test_path": "dummy.csv"}
    mock_config_path = MagicMock()
    mock_config.resolve_path.return_value = mock_config_path
    mock_load_config.return_value = mock_config

    mock_producer_instance = MagicMock()
    mock_kafka_producer.return_value = mock_producer_instance

    main()

    # Verify that the producer sent 1 record
    assert mock_producer_instance.send.call_count == 1
    args, kwargs = mock_producer_instance.send.call_args
    assert args[0] == "fraud_events"  # default topic in base.yaml

    # The payload should have proper types
    payload = args[1]
    assert payload["row_id"] == 0
    assert payload["amt"] == 100.0
    assert payload["city_pop"] == 1000

    mock_producer_instance.flush.assert_called_once()
    mock_producer_instance.close.assert_called_once()
