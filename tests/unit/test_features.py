from __future__ import annotations

import pandas as pd

from fraud_detection.features import add_derived_features


def test_add_derived_features() -> None:
    rows = [
        {
            "row_id": 1,
            "trans_date_trans_time": "2019-01-01 23:45:00",
            "cc_num": "123",
            "merchant": "merchant_a",
            "category": "grocery_pos",
            "amt": 22.5,
            "first": "Jane",
            "last": "Doe",
            "gender": "F",
            "street": "123 Main",
            "city": "Austin",
            "state": "TX",
            "zip": "78701",
            "lat": 30.2672,
            "long": -97.7431,
            "city_pop": 100000,
            "job": "Engineer",
            "dob": "1990-05-20",
            "trans_num": "tx-1",
            "unix_time": 1546386300,
            "merch_lat": 30.2682,
            "merch_long": -97.7501,
            "is_fraud": 0,
        }
    ]
    transformed = add_derived_features(pd.DataFrame(rows)).iloc[0]

    assert int(transformed["txn_hour"]) == 23
    assert int(transformed["is_night_txn"]) == 1
    assert transformed["distance_km"] > 0
    assert transformed["age_years"] >= 28
