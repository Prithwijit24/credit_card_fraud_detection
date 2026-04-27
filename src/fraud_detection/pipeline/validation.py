from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

LOGGER = logging.getLogger(__name__)


def validate_transactions(df: DataFrame) -> DataFrame:
    """
    Applies data quality checks to the transaction DataFrame.
    Drops records that violate core assumptions.
    """
    initial_count = df.count()
    
    # 1. Transaction amount must be positive
    # 2. Critical fields must not be null
    valid_df = df.filter(
        (F.col("amt") > 0)
        & F.col("cc_num").isNotNull()
        & F.col("merchant").isNotNull()
        & F.col("trans_date_trans_time").isNotNull()
    )
    
    final_count = valid_df.count()
    dropped_count = initial_count - final_count
    
    if dropped_count > 0:
        LOGGER.warning("Data validation dropped %s malformed records.", dropped_count)
    else:
        LOGGER.info("Data validation passed. No records dropped.")
        
    return valid_df
