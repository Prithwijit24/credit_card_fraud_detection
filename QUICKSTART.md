# 🚀 Quickstart Guide

This project runs on Pandas, scikit-learn-compatible pipelines, DuckDB feature storage, Kafka,
FastAPI, Streamlit, packaged YAML configs, and the notebook transaction schema from
`notebooks/credit_card_fraud_complete.ipynb`.

## 1. 🛠️ Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,stream,api,ui]"
```

Optional model libraries are grouped separately:

```bash
pip install -e ".[models,deep]"
```

Put the historical transaction file at `data/transactions.txt`, or change
`src/fraud_detection/configs/base.yml`.

## 2. 🏋️ Train

Training reads `src/fraud_detection/configs/base.yml`, loads the historical JSONL/CSV transaction
file, builds leakage-safe Pandas features, persists a DuckDB feature snapshot, tunes the model on
PR-AUC, and saves only if the PR-AUC and recall gates pass.

```bash
fraud train --config base
```

Train a specific registered model:

```bash
fraud train --config base --mode individual --model random_forest
```

Stacked mode trains anomaly feature models first, then booster models, then the meta learner:

```bash
fraud train --config base --mode stacked
```

Expected outputs:

- 📦 `models/trained/latest/model.joblib`
- 📈 `data/metrics/latest_metrics.json`

## 3. 🌊 Score Streaming Events

Start Kafka:

```bash
docker compose up -d zookeeper kafka
```

Start the Pandas micro-batch scorer:

```bash
fraud stream --config base
```

In another terminal, publish sample transactions:

```bash
fraud produce --config base --records 1000 --delay-seconds 0.02
```

Inspect:

- 📁 `data/scored/`
- 🚨 `data/alerts/`

## 4. 🌐 Score With the API

```bash
fraud api --config base --host 127.0.0.1 --port 8000
```

Then post one notebook-schema transaction to `http://127.0.0.1:8000/score`.

Useful endpoints:

- 🩺 `GET /health`
- ℹ️ `GET /metadata`
- 📝 `POST /score`
- 📑 `POST /score/batch`
- ➕ `POST /feature-store/append`

## 5. 🖥️ Use the Streamlit UI

```bash
FRAUD_API_URL=http://127.0.0.1:8000 streamlit run scripts/run_ui.py
```

Or use the console entrypoint:

```bash
FRAUD_API_URL=http://127.0.0.1:8000 fraud ui
```

## 6. ✅ Run Checks

```bash
pytest
ruff check .
mypy src
```

## 7. 🐳 Docker Demo

```bash
docker compose up --build
```

Compose builds and runs Kafka, the trainer, streaming scorer, FastAPI service, Streamlit UI, and
sample producer. The Docker config expects training data mounted at `/data/transactions.txt` from
the host `../data` directory.

## 8. ☸️ Kubernetes Manifests

Replace `ghcr.io/OWNER/REPO` in `k8s/*.yaml`, then apply:

```bash
kubectl apply -f k8s/
```
