# Credit Card Fraud Detection Pipeline

[![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Kafka](https://img.shields.io/badge/Kafka-3.6+-000000?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

An end-to-end fraud detection system built around Pandas feature engineering, a persisted
scikit-learn model pipeline, Kafka micro-batch scoring, and a low-latency FastAPI scoring path.
The design emphasizes the parts that matter in a senior data science interview: training-serving
parity, rare-event evaluation, explicit quality gates, reproducible artifacts, and operationally
simple deployment.

## System Architecture

```mermaid
graph TD
    subgraph "Offline Training"
        A[(Historical Transactions CSV)] --> B[Pandas Schema Coercion]
        B --> C[Validation + Feature Engineering]
        C --> D[sklearn Preprocessing + Random Forest]
        D --> E[PR-AUC Cross-Validation]
        E --> F{Promotion Gate}
        F -->|Pass| G[models/trained/latest/model.joblib]
        F -->|Fail| H[Abort Promotion]
    end

    subgraph "Online Scoring"
        I((Kafka Topic)) --> J[Pandas Micro-Batch Scorer]
        K[FastAPI /score] --> L[Single-Transaction Scorer]
        G -.-> J
        G -.-> L
    end

    subgraph "Outputs"
        J --> M[data/scored/*.jsonl]
        J --> N[data/alerts/*.jsonl]
        J --> O[Batch Monitoring Logs]
        L --> P[fraud_probability + risk_band]
    end
```

## What Makes This Production-Oriented

| Capability | Implementation |
| :--- | :--- |
| Training-serving parity | `features.py` and `schemas.py` are reused by training, API scoring, and stream scoring. |
| Rare-event evaluation | Model selection optimizes average precision / PR-AUC instead of accuracy. |
| Promotion control | Training refuses to save weak models when PR-AUC or recall misses the quality gate. |
| Imbalance handling | Random Forest uses class weighting plus sample weights from observed fraud skew. |
| Unknown category safety | The sklearn pipeline uses imputation and one-hot encoding with unknown handling. |
| Simple serving artifact | The full preprocessing + model pipeline is saved as `model.joblib`. |
| Operational paths | Supports batch training, Kafka micro-batch scoring, and synchronous REST scoring. |

## Project Structure

```text
configs/                  Runtime paths, thresholds, and model hyperparameters
docker/                   Lightweight Python images for trainer and scorer
docs/architecture.md      Operational architecture and extension notes
scripts/                  Thin CLI wrappers
src/fraud_detection/
  features.py             Shared Pandas feature engineering
  schemas.py              Canonical transaction schema coercion
  pipeline/
    validation.py         Data quality checks
    modeling.py           sklearn pipeline and model persistence
    metrics.py            Fraud-focused evaluation and scoring helpers
  jobs/
    train.py              Offline trainer with CV and promotion gate
    streaming.py          Kafka/CSV micro-batch scorer
    api.py                FastAPI scoring endpoint
tests/                    Unit and integration coverage
```

## Getting Started

Install the local development environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,stream,api]"
```

Train and promote a model:

```bash
python scripts/train_model.py --config configs/base.yaml
```

Run the scorer and produce sample traffic:

```bash
docker compose up -d zookeeper kafka
python scripts/run_streaming_job.py --config configs/base.yaml
python scripts/produce_events.py --config configs/base.yaml --records 1000 --delay-seconds 0.02
```

Or run the containerized demo:

```bash
docker compose up --build
```

## Outputs

- `models/trained/latest/model.joblib`: promoted sklearn preprocessing + model artifact
- `data/metrics/latest_metrics.json`: ROC-AUC, PR-AUC, F1, precision, recall, and confusion counts
- `data/scored/*.jsonl`: scored transactions with probabilities and risk bands
- `data/alerts/*.jsonl`: transactions above the fraud probability threshold

## Quality Checks

```bash
pytest
ruff check .
mypy src
```

## Documentation

- [Quickstart Guide](./QUICKSTART.md)
- [Codebase Roadmap](./CODEBASE_GUIDE.md)
- [Architecture Deep-Dive](./docs/architecture.md)
