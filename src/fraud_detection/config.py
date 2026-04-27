from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    root_dir: Path

    @property
    def app(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw["app"])

    @property
    def data(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw["data"])

    @property
    def runtime(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw.get("runtime", {}))

    @property
    def streaming(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw["streaming"])

    @property
    def model(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw["model"])

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root_dir / path).resolve()


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig(raw=raw, root_dir=path.parent.parent)
