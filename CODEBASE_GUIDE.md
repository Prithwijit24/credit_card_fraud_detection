# 🗺️ Codebase Reading Roadmap

Follow this curated path to understand the system architecture and implementation logic.

---

## 🛤️ Learning Path

| Step | File / Directory | Why read this? |
| :--- | :--- | :--- |
| 1 | `README.md` | Start here for the high-level system overview. |
| 2 | `configs/base.yaml` | Understand the environment: data paths, Spark configs, and model hyperparams. |
| 3 | `src/.../validation.py` | See how we ensure data quality before processing. |
| 4 | `src/.../features.py` | **Most Important!** This logic is shared between training and inference. |
| 5 | `src/.../modeling.py` | Learn how we transform features and tune the RandomForest. |
| 6 | `src/jobs/` | Explore the 3 paths: `train.py`, `streaming.py`, and `api.py`. |

---

## 📂 Engineering Anatomy

### ⚙️ Infrastructure & Config
- **`configs/`**: Centralized configuration management.
- **`docker/`**: Production-ready environment recipes.
- **`docker-compose.yml`**: The local runtime topology.

### 🧠 Core Engine (`src/fraud_detection/`)
- 🛡️ **`pipeline/`**: The "Science" - Validation, Metrics, and Model Definition.
- ⚡ **`jobs/`**: The "Execution" - Entrypoints for batch and stream processing.
- 🛠️ **`spark.py`**: A robust factory for Spark Sessions.

### 🧪 Quality Assurance
- **`tests/`**: Unit tests for features and integration tests for the full pipeline.
- **`.github/workflows/`**: CI/CD automation for testing and deployment.

---

## 💡 Key Design Decisions

> [!IMPORTANT]
> **No Training-Serving Skew**: We import the exact same Python functions from `features.py` into both the Spark Trainer and the FastAPI Scorer.

> [!TIP]
> **Quality Gate**: The `train.py` job will **fail** and refuse to save the model if the AUC-ROC is below 0.85. We only ship excellence.

---

## 🚫 What to Skip (For Now)
- `notebooks/`: Contains legacy exploratory code. It’s useful for history, but not part of the production pipeline.

