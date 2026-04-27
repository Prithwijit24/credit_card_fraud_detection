from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

CATEGORICAL_COLUMNS = ["category", "merchant", "gender", "state", "job"]
NUMERIC_COLUMNS = [
    "amt",
    "city_pop",
    "age_years",
    "distance_km",
    "txn_hour",
    "txn_day_of_week",
    "is_night_txn",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def haversine_distance_km(
    lat1: F.Column,
    lon1: F.Column,
    lat2: F.Column,
    lon2: F.Column,
) -> F.Column:
    earth_radius_km = F.lit(6371.0)
    lat1_rad = F.radians(lat1)
    lat2_rad = F.radians(lat2)
    delta_lat = F.radians(lat2 - lat1)
    delta_lon = F.radians(lon2 - lon1)
    a = (
        F.sin(delta_lat / 2) ** 2
        + F.cos(lat1_rad) * F.cos(lat2_rad) * F.sin(delta_lon / 2) ** 2
    )
    return earth_radius_km * 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))


def add_derived_features(df: DataFrame) -> DataFrame:
    transaction_ts = F.to_timestamp("trans_date_trans_time")
    dob = F.to_date("dob")
    age_years = F.floor(F.months_between(F.to_date(transaction_ts), dob) / 12)
    distance = haversine_distance_km(
        F.col("lat"),
        F.col("long"),
        F.col("merch_lat"),
        F.col("merch_long"),
    )

    return (
        df.withColumn("event_ts", transaction_ts)
        .withColumn("age_years", F.coalesce(age_years, F.lit(0)))
        .withColumn("distance_km", F.round(F.coalesce(distance, F.lit(0.0)), 3))
        .withColumn("txn_hour", F.hour(transaction_ts))
        .withColumn("txn_day_of_week", F.dayofweek(transaction_ts))
        .withColumn(
            "is_night_txn",
            F.when(F.hour(transaction_ts).isin([0, 1, 2, 3, 4, 22, 23]), 1).otherwise(0),
        )
        .fillna(
            {
                "category": "unknown",
                "merchant": "unknown",
                "gender": "U",
                "state": "NA",
                "job": "unknown",
                "amt": 0.0,
                "city_pop": 0,
            }
        )
    )


def sanitize_input_columns(df: DataFrame) -> DataFrame:
    renamed = df
    if "" in df.columns:
        renamed = renamed.withColumnRenamed("", "row_id")
    return renamed
