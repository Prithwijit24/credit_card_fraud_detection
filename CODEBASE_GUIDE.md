# 🗺️ Codebase Reading Roadmap

Use this path to explore the system efficiently and understand the key components for development, debugging, or interviews.

---

## 🚀 Learning Path

| Step | Area | 📂 Location | 💡 Why read this? |
| :--- | :--- | :--- | :--- |
| **1** | **Overview** | `README.md` | 🌟 System overview & operating modes. |
| **2** | **Config** | `src/fraud_detection/configs/base.yml` | ⚙️ Data paths, stream & model params. |
| **3** | **Config Logic** | `src/fraud_detection/config.py` | 🔍 Resolver & path handling. |
| **4** | **Schema** | `src/fraud_detection/schemas.py` | 📝 Input normalization contract. |
| **5** | **Validation** | `src/fraud_detection/pipeline/validation.py` | ✅ Quality assurance before modeling. |
| **6** | **Feature Contract**| `src/fraud_detection/feature_pipeline.py` | 🧬 Unified training/serving contract. |
| **7** | **Transforms** | `src/fraud_detection/feature_transforms/` | 🧩 Modular feature engineering. |
| **8** | **Model Registry** | `src/fraud_detection/models/registry.py` | 🧠 Model lookup & dependencies. |
| **9** | **Pipeline** | `src/fraud_detection/pipeline/modeling.py` | 🏗️ Sklearn pipelines & persistence. |
| **10**| **Metrics** | `src/fraud_detection/pipeline/metrics.py` | 📊 Fraud-focused evaluation. |
| **11**| **Entrypoints** | `src/fraud_detection/jobs/` | 🚀 Production execution paths. |

---

## 🔬 Engineering Anatomy

### 📊 Core Data Science Layer
*   📝 **Schema & Validation**: `schemas.py` & `validation.py` ensure data consistency.
*   🧬 **Feature Engine**: `feature_pipeline.py` orchestrates `feature_transforms/` (history, time, account, amount, risk, encodings).
*   🏗️ **Modeling**: `modeling.py` creates robust sklearn-compatible (individual or stacked) pipelines.
*   📊 **Evaluation**: `metrics.py` computes fraud-specific performance.
*   🗄️ **Storage**: `feature_store.py` manages the DuckDB feature store.

### 🧠 Model Layer
*   🌟 **Interface**: `models/base.py` defines the standard contract.
*   🗃️ **Registry**: `models/registry.py` is the single source of truth for model lookup.
*   🏆 **Champions/Baselines**: `xgboost.py`, `random_forest.py`, `logistic_regression.py`.
*   📈 **Anomaly/Deep**: `isolation_forest.py`, `one_class_svm.py`, `autoencoder.py`, `tabtransformer.py`.
*   ✨ **Challengers**: `lightgbm.py`, `catboost.py`, `stacking_ensemble.py`.

### ⚙️ Execution & Runtime
*   🏋️ **Training**: `jobs/train.py` (CV, promotion gate).
*   🌊 **Streaming**: `jobs/streaming.py` (Kafka/CSV scoring).
*   🌐 **API**: `jobs/api.py` (Synchronous scoring).
*   🖥️ **UI**: `jobs/ui.py` (Streamlit console).
*   🐳 **Containerization**: `docker/` (Trainer, Streaming, API, UI images).
*   ☸️ **Orchestration**: `k8s/` (Deployment manifests).

---

## 💡 Key Design Philosophies

| Philosophy | 🛡️ Impact |
| :--- | :--- |
| **Single Artifact** | No drift between training & serving. |
| **Modular Transforms** | Testable components, unified contract. |
| **Metadata-in-Model** | Encoders travel with the artifact. |
| **PR-AUC Objective** | Optimizes for highly imbalanced data. |
| **DuckDB Store** | Embedded, audit-ready replay store. |

---

## 🚫 Avoid
- 📓 `notebooks/`: Purely exploratory.
- 🗑️ `data/` or `models/` (generated artifacts).
