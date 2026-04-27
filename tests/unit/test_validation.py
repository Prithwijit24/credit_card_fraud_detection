from __future__ import annotations

from pyspark.sql import Row

from fraud_detection.pipeline.validation import validate_transactions


def test_validate_transactions_drops_invalid(spark):
    data = [
        Row(amt=100.0, cc_num="123", merchant="abc", trans_date_trans_time="2020-01-01"),  # valid
        Row(amt=-10.0, cc_num="123", merchant="abc", trans_date_trans_time="2020-01-01"),  # invalid amt
        Row(amt=100.0, cc_num=None, merchant="abc", trans_date_trans_time="2020-01-01"),  # invalid cc_num
        Row(amt=100.0, cc_num="123", merchant=None, trans_date_trans_time="2020-01-01"),  # invalid merchant
        Row(amt=100.0, cc_num="123", merchant="abc", trans_date_trans_time=None),         # invalid date
    ]
    df = spark.createDataFrame(data)
    valid_df = validate_transactions(df)
    assert valid_df.count() == 1
    assert valid_df.collect()[0].amt == 100.0
