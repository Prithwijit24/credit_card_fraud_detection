# 🛡️ Credit Card Fraud Detection Pipeline

[![Spark](https://img.shields.io/badge/Spark-3.5+-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Kafka-3.6+-000000?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

A **production-grade** real-time fraud detection engine. This isn't just a model; it's a complete ecosystem designed to train, score, and alert at scale using the modern Big Data stack.

---

## 📽️ System Architecture

Understanding the flow is easy. We move from **Historical Insights** to **Real-Time Actions**:

```mermaid
graph TD
    subgraph "1. Training Phase (Offline)"
        A[(Historical CSV)] -->|Clean & Engineer| B[PySpark Trainer]
        B -->|Hyperparameter Tuning| C{Quality Gate}
        C -->|Pass| D[Serialized Model Artifact]
    end

    subgraph "2. Scoring Phase (Online)"
        E((Kafka Stream)) -->|Live Events| F[Structured Streaming Job]
        D -.->|Load Model| F
        F -->|Predict| G[Fraud Probability]
    end

    subgraph "3. Output & Monitoring"
        G --> H[(Scored Parquet Sink)]
        G -->|Risk > Threshold| I[🔥 Fraud Alert JSON]
        G --> J[📊 Console Metrics]
    end

    style I fill:#f96,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Unified Logic** | Exactly the same feature engineering used for both Training and Streaming. No skew! |
| **Quality Gates** | Automatic AUC-ROC validation (>= 0.85) before any model is promoted. |
| **Dual-Path Scoring** | Choose between **High-Throughput Streaming** or **Low-Latency REST API**. |
| **Fault Tolerant** | Built-in Spark checkpointing and Kafka replay support. |

---

## 📂 Project Structure

> [!TIP]
> Explore the codebase like a pro by following this layout:

- ⚙️ **`configs/`**: The brain of the operation. Define paths and hyperparams here.
- 🏗️ **`src/fraud_detection/`**: The core engine.
  - 🧪 **`pipeline/`**: Logic for validation and modeling.
  - ⚡ **`jobs/`**: The actual workers (Trainer, Streamer, API).
- 🧪 **`tests/`**: Battle-testing every component.
- 🐳 **`docker-compose.yml`**: Launch the entire universe with one command.

---

## 🛠️ Getting Started

### 1. The Fast Path (Docker)
Launch the full stack including Kafka, HDFS, and Spark:
```bash
docker compose up --build
```

### 2. The Developer Path (Local)
```bash
make install-dev
make train      # Train the model
make stream     # Start scoring live events
```

> [!IMPORTANT]
> **Prerequisites**: Ensure you have Java 17+ installed if running outside Docker.

---

## 📈 Operational Monitoring

Once running, check these directories for live results:
- 🎯 **Scored Events**: `data/scored/`
- 🚨 **High-Risk Alerts**: `data/alerts/`
- 📊 **Model Metrics**: `data/metrics/`

---

## 🔗 Documentation
- [📘 Quickstart Guide](./QUICKSTART.md) - Get running in 2 minutes.
- [🗺️ Codebase Roadmap](./CODEBASE_GUIDE.md) - Where to look first.
- [🏛️ Architecture Deep-Dive](./docs/architecture.md) - How it all fits together.

---
*Built with ❤️ for High-Performance Engineering.*
