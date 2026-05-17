from __future__ import annotations

from typing import Any, cast

from fraud_detection.models import available_components as _available_components
from fraud_detection.models import build_model as _build_model


def available_components() -> list[dict[str, str]]:
    return cast(list[dict[str, str]], _available_components())


def build_optional_model(name: str, seed: int = 42) -> Any:
    return _build_model(name, seed=seed)
