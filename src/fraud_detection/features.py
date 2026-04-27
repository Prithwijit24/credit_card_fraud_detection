from __future__ import annotations

import numpy as np
import pandas as pd

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
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    earth_radius_km = 6371.0
    lat1_rad = np.radians(lat1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    delta_lat = np.radians(lat2.astype(float) - lat1.astype(float))
    delta_lon = np.radians(lon2.astype(float) - lon1.astype(float))
    a = np.sin(delta_lat / 2) ** 2 + (
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return pd.Series(
        earth_radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)),
        index=lat1.index,
    )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    transaction_ts = pd.to_datetime(transformed["trans_date_trans_time"], errors="coerce")
    dob = pd.to_datetime(transformed["dob"], errors="coerce")

    age_years = ((transaction_ts - dob).dt.days // 365).fillna(0).clip(lower=0)
    distance = haversine_distance_km(
        pd.to_numeric(transformed["lat"], errors="coerce"),
        pd.to_numeric(transformed["long"], errors="coerce"),
        pd.to_numeric(transformed["merch_lat"], errors="coerce"),
        pd.to_numeric(transformed["merch_long"], errors="coerce"),
    )

    transformed["event_ts"] = transaction_ts
    transformed["age_years"] = age_years.astype(int)
    transformed["distance_km"] = distance.fillna(0.0).round(3)
    transformed["txn_hour"] = transaction_ts.dt.hour.fillna(-1).astype(int)
    transformed["txn_day_of_week"] = transaction_ts.dt.dayofweek.add(1).fillna(0).astype(int)
    transformed["is_night_txn"] = transformed["txn_hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype(int)

    categorical_defaults = {
        "category": "unknown",
        "merchant": "unknown",
        "gender": "U",
        "state": "NA",
        "job": "unknown",
    }
    numeric_defaults = {"amt": 0.0, "city_pop": 0}
    transformed = transformed.fillna({**categorical_defaults, **numeric_defaults})
    for column in NUMERIC_COLUMNS:
        transformed[column] = pd.to_numeric(transformed[column], errors="coerce").fillna(0)
    return transformed


def sanitize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {
        column: "row_id"
        for column in renamed.columns
        if str(column).startswith("Unnamed") or column == ""
    }
    return renamed.rename(columns=rename_map)
