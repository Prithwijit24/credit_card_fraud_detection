from __future__ import annotations

import pandas as pd

from fraud_detection.pipeline.validation import validate_transactions
from tests.fixtures import notebook_transaction


def test_validate_transactions_drops_invalid():
    data = [
        notebook_transaction(transactionAmount=100.0),
        notebook_transaction(transactionAmount=-10.0),
        notebook_transaction(accountNumber=None),
        notebook_transaction(merchantName=None),
        notebook_transaction(transactionDateTime=None),
    ]
    valid_df = validate_transactions(pd.DataFrame(data))
    assert len(valid_df) == 1
    assert valid_df.iloc[0].transactionAmount == 100.0
