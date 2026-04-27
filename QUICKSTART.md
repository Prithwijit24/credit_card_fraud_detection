# Quickstart Guide

This project now runs on Pandas and scikit-learn. No distributed compute services are
required.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,stream,api]"
```

## 2. Train

Training reads `configs/base.yaml`, loads the historical CSV, builds Pandas features, tunes the
sklearn pipeline on PR-AUC, and saves only if the quality gate passes.

```bash
python scripts/train_model.py --config configs/base.yaml
```

Expected outputs:

- `models/trained/latest/model.joblib`
- `data/metrics/latest_metrics.json`

## 3. Score Streaming Events

Start Kafka:

```bash
docker compose up -d zookeeper kafka
```

Start the Pandas micro-batch scorer:

```bash
python scripts/run_streaming_job.py --config configs/base.yaml
```

In another terminal, publish sample transactions:

```bash
python scripts/produce_events.py --config configs/base.yaml --records 1000 --delay-seconds 0.02
```

Inspect:

- `data/scored/`
- `data/alerts/`

## 4. Score With the API

```bash
python -m fraud_detection.jobs.api --config configs/base.yaml --host 127.0.0.1 --port 8000
```

Then post one transaction to `http://127.0.0.1:8000/score`.

## 5. Run Checks

```bash
pytest
ruff check .
mypy src
```

## 6. Docker Demo

```bash
docker compose up --build
```
