from __future__ import annotations

CATEGORICAL_COLUMNS = [
    "merchantName",
    "merchantCategoryCode",
    "acqCountry",
    "merchantCountryCode",
    "posEntryMode",
    "posConditionCode",
    "transactionType",
    "merchantState",
]

ENTITY_ENCODING_SOURCE_COLUMNS = [
    "merchantName",
    "merchantCategoryCode",
    "transactionType",
    "posEntryMode",
]

ENTITY_ENCODING_COLUMNS = [
    f"{column}_fraud_rate" for column in ENTITY_ENCODING_SOURCE_COLUMNS
]

NUMERIC_COLUMNS = [
    "transactionAmount",
    "creditLimit",
    "availableMoney",
    "currentBalance",
    "txn_hour",
    "txn_day_of_week",
    "txn_month",
    "is_weekend",
    "is_night_txn",
    "account_age_days",
    "days_since_address_change",
    "months_to_expiry",
    "utilization",
    "amount_to_limit_ratio",
    "cvv_mismatch",
    "expiry_key_mismatch",
    "card_present_flag",
    "cross_border",
    "pos_risk_tier",
    "txn_count_24h",
    "txn_count_48h",
    "txn_count_7d",
    "txn_count_30d",
    "txn_amount_sum_24h",
    "txn_amount_sum_7d",
    "txn_amount_mean_7d",
    "seconds_since_last_txn",
    "is_new_merchant",
    *ENTITY_ENCODING_COLUMNS,
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
