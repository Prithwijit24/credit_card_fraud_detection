# ⚡ Quickstart Guide

Get the pipeline up and running in minutes. Follow these **3 main phases**.

---

## 🏗️ Phase 1: Environment Setup

### 1. Python Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,stream]"
```

> [!NOTE]
> **Host Requirements**: If you run Spark directly on your machine, ensure **Java 17+** is installed.

---

## 🧠 Phase 2: Model Training

### 2. Run the Trainer
```bash
python scripts/train_model.py --config configs/base.yaml
```

**✅ Expected Outputs:**
- `models/trained/latest/` (The serialized ML pipeline)
- `data/metrics/latest_metrics.json` (Performance report)

---

## 🌊 Phase 3: Live Streaming

### 3. Start Infrastructure
```bash
docker compose up -d zookeeper kafka namenode datanode spark-master spark-worker
```

### 4. Start the Scorer
```bash
python scripts/run_streaming_job.py --config configs/base.yaml
```

### 5. Simulate Traffic
In a separate terminal, push sample events to Kafka:
```bash
python scripts/produce_events.py --config configs/base.yaml --records 1000 --delay-seconds 0.02
```

---

## 🕵️ Phase 4: Verification

### Inspect the Results
- 🏁 **Scored Events**: `data/scored/`
- 🚨 **Fraud Alerts**: `data/alerts/`

### Run Quality Checks
```bash
pytest          # Logic tests
ruff check .    # Linting
mypy src        # Type checking
```

---

> [!TIP]
> **One-Command Wonder**: Want to run everything in a containerized sandbox? 
> Just run `docker compose up --build`.
