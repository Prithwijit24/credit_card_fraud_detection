from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeature(ABC):
    name: str
    output_columns: tuple[str, ...] = ()

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
