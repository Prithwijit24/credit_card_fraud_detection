from __future__ import annotations

from fraud_detection.config import load_config


def test_load_config(project_root) -> None:
    config = load_config("base")
    assert config.app["name"] == "credit-card-fraud-detection"
    assert config.resolve_path(config.data["model_dir"]).name == "latest"
