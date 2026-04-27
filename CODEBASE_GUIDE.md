# Codebase Reading Roadmap

Use this path to understand the system quickly and explain it clearly in an interview.

## Learning Path

| Step | File / Directory | Why read this? |
| :--- | :--- | :--- |
| 1 | `README.md` | System overview, architecture, and operating modes. |
| 2 | `configs/base.yaml` | Data locations, stream settings, fraud threshold, and model hyperparameters. |
| 3 | `src/fraud_detection/schemas.py` | Canonical Pandas schema coercion across CSV, API, and stream inputs. |
| 4 | `src/fraud_detection/pipeline/validation.py` | Core data quality assumptions before modeling. |
| 5 | `src/fraud_detection/features.py` | Shared feature engineering used by every scoring path. |
| 6 | `src/fraud_detection/pipeline/modeling.py` | sklearn preprocessing, Random Forest, CV, and artifact persistence. |
| 7 | `src/fraud_detection/pipeline/metrics.py` | Fraud-specific metrics and thresholded scoring. |
| 8 | `src/fraud_detection/jobs/` | Production entrypoints: training, micro-batch scoring, and API scoring. |

## Engineering Anatomy

### Core Data Science Layer

- `schemas.py`: normalizes raw records into the canonical transaction shape.
- `validation.py`: drops malformed records, non-positive amounts, missing IDs, and invalid dates.
- `features.py`: derives age, distance, transaction hour, day of week, and night-transaction flags.
- `modeling.py`: creates a single sklearn pipeline that owns preprocessing and classification.
- `metrics.py`: reports ROC-AUC, PR-AUC, F1, precision, recall, and confusion counts.

### Execution Layer

- `jobs/train.py`: offline trainer with stratified split, CV, sample weighting, and promotion gate.
- `jobs/streaming.py`: Kafka or CSV micro-batch scorer writing scored events and alerts.
- `jobs/api.py`: synchronous single-transaction scoring with the same persisted pipeline.
- `jobs/producer.py`: sample Kafka producer for replaying transaction events.

### Runtime Layer

- `configs/`: runtime settings, data paths, thresholds, and hyperparameters.
- `docker/`: Python images for trainer and scorer.
- `docker-compose.yml`: local Kafka plus trainer/scorer/producer topology.
- `tests/`: focused unit and integration coverage for the shared logic and model pipeline.

## Design Decisions Worth Calling Out

| Decision | Why it matters |
| :--- | :--- |
| Pandas feature layer | Simple local development and transparent feature logic for tabular fraud data. |
| One persisted sklearn pipeline | Prevents preprocessing drift between training and serving. |
| PR-AUC model selection | Better objective for highly imbalanced fraud detection than accuracy. |
| Recall quality gate | Controls false negatives, which are usually expensive in fraud systems. |
| Thresholded risk bands | Separates probability estimation from operational alert policy. |
| Micro-batch scoring | Keeps the streaming path understandable while preserving Kafka-based ingestion. |

## What to Skip First

- `notebooks/`: exploratory history, not part of the production runtime.
- Generated local data under `data/`, `models/`, and `coverage.xml`.
