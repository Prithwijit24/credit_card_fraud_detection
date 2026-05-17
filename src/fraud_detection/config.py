from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_CONFIG_NAME = "base"


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


def _project_root_from(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def _packaged_config_path(name: str) -> Path:
    config_name = name if name.endswith(".yml") else f"{name}.yml"
    return Path(str(files("fraud_detection.configs").joinpath(config_name)))


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return _packaged_config_path(DEFAULT_CONFIG_NAME)

    raw_path = Path(config_path)
    if raw_path.exists():
        return raw_path.resolve()

    if raw_path.suffix == "":
        return _packaged_config_path(raw_path.name)

    return _packaged_config_path(raw_path.name)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig(raw=raw, root_dir=_project_root_from(path.parent))
