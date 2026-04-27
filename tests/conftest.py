from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    java_path = shutil.which("java")
    if java_path is None:
        pytest.skip("Java is required for Spark tests.")

    os.environ.setdefault("JAVA_HOME", str(Path(java_path).resolve().parents[1]))
    session = (
        SparkSession.builder.master("local[2]")
        .appName("fraud-detection-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
