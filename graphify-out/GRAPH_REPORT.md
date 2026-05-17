# Graph Report - .  (2026-05-17)

## Corpus Check
- Corpus is ~13,087 words - fits in a single context window. You may not need a graph.

## Current Documentation Update
- The Markdown docs now describe the packaged config layout under
  `src/fraud_detection/configs/*.yml`, replacing the old root-level `configs/*.yaml` references.
- Runtime commands now use the unified `fraud` CLI for training, streaming, producing events,
  serving the API, and launching the UI.
- The documented runtime surface now includes separate trainer, streaming, API, and UI Docker
  images plus Kubernetes manifests for the same services.
- Feature documentation now reflects `FeaturePipeline`, modular `feature_transforms/`, entity
  encoders persisted in model metadata, and the DuckDB feature-store adapter.
- Model documentation now reflects the registry-backed component set, individual and stacked
  training modes, optional challenger libraries, and sklearn-compatible fallbacks.

## Summary
- 247 nodes · 368 edges · 26 communities (18 shown, 8 thin omitted)
- Extraction: 79% EXTRACTED · 20% INFERRED · 1% AMBIGUOUS · INFERRED: 73 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_API Scoring Core|API Scoring Core]]
- [[_COMMUNITY_Architecture Docs|Architecture Docs]]
- [[_COMMUNITY_Feature API Flow|Feature API Flow]]
- [[_COMMUNITY_Test Contracts|Test Contracts]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_Transaction Ingestion|Transaction Ingestion]]
- [[_COMMUNITY_Runtime Config|Runtime Config]]
- [[_COMMUNITY_Packaged Runtime Config|Packaged Runtime Config]]
- [[_COMMUNITY_CI Kubernetes|CI Kubernetes]]
- [[_COMMUNITY_Feature Store|Feature Store]]
- [[_COMMUNITY_Local Streaming Demo|Local Streaming Demo]]
- [[_COMMUNITY_Model Components|Model Components]]
- [[_COMMUNITY_Package Compatibility|Package Compatibility]]
- [[_COMMUNITY_Package Root|Package Root]]
- [[_COMMUNITY_Pipeline Package|Pipeline Package]]
- [[_COMMUNITY_Test Package|Test Package]]
- [[_COMMUNITY_VS Code Settings|VS Code Settings]]
- [[_COMMUNITY_Project Root Fixture|Project Root Fixture]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 13 edges
2. `main()` - 12 edges
3. `Single Transaction Scoring Payload Pipeline` - 12 edges
4. `add_derived_features()` - 11 edges
5. `Micro Batch Fraud Scoring Job` - 11 edges
6. `Model Training Job` - 11 edges
7. `DuckDBFeatureStore` - 10 edges
8. `_build_train_test_features()` - 10 edges
9. `score_payload()` - 9 edges
10. `Derived Transaction Feature Engineering` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Production Serving Blocks` --semantically_similar_to--> `Serving Surfaces`  [INFERRED] [semantically similar]
  production_system_design.svg → docs/architecture.md
- `Deployment and Monitoring Plan` --semantically_similar_to--> `Operational Notes`  [INFERRED] [semantically similar]
  fraud_detection_project_plan.svg → docs/architecture.md
- `Configured Model Components` --semantically_similar_to--> `Modeling Scope Plan`  [INFERRED] [semantically similar]
  src/fraud_detection/configs/base.yml → fraud_detection_project_plan.svg
- `Trainer Streaming API UI Producer Services` --references--> `Streamlit UI Entrypoint`  [AMBIGUOUS]
  docker-compose.yml → scripts/run_ui.py
- `Legacy Credit Card Fraud EDA Notebook Export` --conceptually_related_to--> `Credit Card Fraud Detection Platform`  [AMBIGUOUS]
  notebooks/credit_card_fraud_detection.py → README.md

## Hyperedges (group relationships)
- **Shared Training and Serving Contract** — architecture_canonical_schema, readme_training_serving_parity, readme_model_artifact [EXTRACTED 1.00]
- **Local Docker Demo Runtime** — docker_compose_kafka_topology, docker_compose_app_services, docker_config_runtime_contract [EXTRACTED 1.00]
- **Kubernetes Release Surface** — k8s_api_deployment, k8s_streaming_deployment, k8s_ui_deployment [EXTRACTED 1.00]
- **Training And Scoring Share Feature Pipeline** — train_feature_building, api_score_payload, streaming_microbatch_scoring [INFERRED 0.90]
- **Model Artifact Lifecycle** — train_training_job, modeling_artifact_persistence, api_fastapi_app [INFERRED 0.85]
- **Transaction Ingestion Contract** — schemas_transaction_schema, producer_kafka_transaction_producer, streaming_source_batches [INFERRED 0.82]
- **Fixture Transaction Schema Supports Tests** — fixtures_notebook_transaction, fixtures_notebook_transaction_schema, test_training_pipeline_fit_and_score, test_api_score_transaction_success, test_features_add_derived_features, test_producer_main, test_validation_drops_invalid [INFERRED 0.85]
- **Training Feature Pipeline Contract** — test_training_pipeline_fit_and_score, features_add_derived_features, metrics_compute_class_weights, modeling_build_training_pipeline, features_feature_columns [EXTRACTED 1.00]
- **Scoring Validation Contract** — test_api_score_transaction_validation_failure, api_score_payload, validation_validate_transactions, test_validation_drops_invalid [INFERRED 0.80]

## Communities (26 total, 8 thin omitted)

### Community 0 - "API Scoring Core"
Cohesion: 0.08
Nodes (43): Fraud Detection FastAPI App, API Feature Store Append Endpoint, Single Transaction Scoring Payload Pipeline, API Top Risk Reasons, Base YAML Config, AppConfig Configuration Object, YAML Config Loader, Config Relative Path Resolver (+35 more)

### Community 1 - "Architecture Docs"
Cohesion: 0.08
Nodes (37): Canonical Transaction Schema, Evaluation Strategy, Event Lifecycle, Operational Notes, Serving Surfaces, Training Lifecycle, Configured Model Components, Base Runtime Configuration Contract (+29 more)

### Community 2 - "Feature API Flow"
Cohesion: 0.1
Nodes (22): add_derived_features(), _add_history_features(), apply_entity_encoders(), _bool_as_int(), fit_entity_encoders(), _safe_datetime(), _safe_numeric(), sanitize_input_columns() (+14 more)

### Community 3 - "Test Contracts"
Cohesion: 0.14
Nodes (20): Feature Columns, Account History Feature Engineering, Notebook Transaction Fixture, Notebook Transaction Schema Fields, Compute Class Weights, Build Training Pipeline, Kafka Producer Dependency, Producer Main (+12 more)

### Community 4 - "Training Pipeline"
Cohesion: 0.16
Nodes (15): _chronological_split(), main(), _metadata(), parse_args(), collect_metrics(), score_frame(), write_metrics(), _build_classifier() (+7 more)

### Community 5 - "Transaction Ingestion"
Cohesion: 0.24
Nodes (13): _coerce_bool(), coerce_transaction_schema(), frame_from_records(), read_transactions(), read_transactions_csv(), read_transactions_jsonl(), build_source_batches(), _iter_csv_batches() (+5 more)

### Community 6 - "Runtime Config"
Cohesion: 0.14
Nodes (6): AppConfig, load_config(), configure_logging(), main(), parse_args(), test_load_config()

### Community 7 - "Packaged Runtime Config"
Cohesion: 0.24
Nodes (4): AppConfig, load_config(), resolve_config_path(), packaged YAML config lookup

### Community 8 - "CI Kubernetes"
Cohesion: 0.36
Nodes (9): CD Image Publishing, CD Manifest Smoke Rendering, CI Container Builds, CI Quality Pipeline, fraud-api Deployment and Service, fraud-config ConfigMap, fraud-detection Namespace, fraud-streaming Deployment (+1 more)

### Community 10 - "Local Streaming Demo"
Cohesion: 0.32
Nodes (8): Trainer Streaming API UI Producer Services, Kafka and Zookeeper Topology, Docker Compose Local Stack, fraud produce command, fraud_detection.jobs.producer.main, Streaming Demo Workflow, fraud stream command, fraud_detection.jobs.streaming.main

## Ambiguous Edges - Review These
- `Credit Card Fraud Detection Platform` → `Legacy Credit Card Fraud EDA Notebook Export`  [AMBIGUOUS]
  notebooks/credit_card_fraud_detection.py · relation: conceptually_related_to
- `Trainer Streaming API UI Producer Services` → `Streamlit UI Entrypoint`  [AMBIGUOUS]
  docker-compose.yml · relation: references
- `XGBoost Random Forest Classifier Fallback` → `Optional Model Dependency Fallbacks`  [AMBIGUOUS]
  src/fraud_detection/models/xgboost.py · relation: conceptually_related_to

## Knowledge Gaps
- **23 isolated node(s):** `python-envs.defaultEnvManager`, `ModelComponent`, `Core Data Science Layer`, `CI Container Builds`, `fraud_detection.jobs.producer.main` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Credit Card Fraud Detection Platform` and `Legacy Credit Card Fraud EDA Notebook Export`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Trainer Streaming API UI Producer Services` and `Streamlit UI Entrypoint`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `XGBoost Random Forest Classifier Fallback` and `Optional Model Dependency Fallbacks`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `main()` connect `Transaction Ingestion` to `Feature Store`, `Feature API Flow`, `Training Pipeline`, `Runtime Config`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `add_derived_features()` connect `Feature API Flow` to `Transaction Ingestion`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `load_config()` connect `Runtime Config` to `Training Pipeline`, `Transaction Ingestion`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `main()` (e.g. with `configure_logging()` and `load_config()`) actually correct?**
  _`main()` has 9 INFERRED edges - model-reasoned connections that need verification._
