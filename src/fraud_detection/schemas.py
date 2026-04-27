from __future__ import annotations

from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

TRANSACTION_SCHEMA = StructType(
    [
        StructField("row_id", LongType(), True),
        StructField("trans_date_trans_time", StringType(), True),
        StructField("cc_num", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("category", StringType(), True),
        StructField("amt", DoubleType(), True),
        StructField("first", StringType(), True),
        StructField("last", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("street", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("zip", StringType(), True),
        StructField("lat", DoubleType(), True),
        StructField("long", DoubleType(), True),
        StructField("city_pop", LongType(), True),
        StructField("job", StringType(), True),
        StructField("dob", StringType(), True),
        StructField("trans_num", StringType(), True),
        StructField("unix_time", LongType(), True),
        StructField("merch_lat", DoubleType(), True),
        StructField("merch_long", DoubleType(), True),
        StructField("is_fraud", LongType(), True),
    ]
)

