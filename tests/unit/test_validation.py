from __future__ import annotations

import pandas as pd

from fraud_detection.pipeline.validation import validate_transactions


def test_validate_transactions_drops_invalid():
    data = [
        {"amt": 100.0, "cc_num": "123", "merchant": "abc", "trans_date_trans_time": "2020-01-01"},
        {"amt": -10.0, "cc_num": "123", "merchant": "abc", "trans_date_trans_time": "2020-01-01"},
        {"amt": 100.0, "cc_num": None, "merchant": "abc", "trans_date_trans_time": "2020-01-01"},
        {"amt": 100.0, "cc_num": "123", "merchant": None, "trans_date_trans_time": "2020-01-01"},
        {"amt": 100.0, "cc_num": "123", "merchant": "abc", "trans_date_trans_time": None},
    ]
    valid_df = validate_transactions(pd.DataFrame(data))
    assert len(valid_df) == 1
    assert valid_df.iloc[0].amt == 100.0
