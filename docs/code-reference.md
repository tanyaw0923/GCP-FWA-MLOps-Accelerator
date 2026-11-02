# Code reference and testing

> Detailed documentation extracted from the original project README. Run commands from the repository root unless otherwise stated.

## Repository Structure

```text
GCP-FWA-MLOps-Accelerator/
├── .gitignore
├── README.md
├── cloudbuild-random-forest.yaml
├── cloudbuild-random-forest-serving.yaml
├── cloudbuild-xgboost.yaml
├── cloudbuild-xgboost-serving.yaml
├── cloudbuild-performance-monitoring.yaml
├── model_training_pipeline.yaml
│
├── data_exports/
│   ├── claims.csv
│   ├── provider_eye_features.csv
│   ├── investigation_results.csv
│   ├── experiment_runs.csv
│   ├── experiment_metrics.csv
│   ├── provider_predictions.csv
│   ├── model_data_quality.csv
│   ├── model_feature_drift.csv
│   ├── model_prediction_drift.csv
│   ├── model_performance_degradation.csv
│   └── model_training_state.csv
│
├── docker/
│   ├── random_forest/                 # Random Forest training image
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── random-forest-serving/         # Random Forest serving image
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── xgboost/                       # XGBoost training image
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── xgboost-serving/               # XGBoost serving image
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   └── performance-monitoring/
│       ├── Dockerfile
│       └── requirements.txt
│
├── sql/
│   ├── evaluation/
│   ├── features/
│   ├── feedback/
│   ├── monitoring/
│   │   ├── model_monitoring.template.sql
│   │   ├── model_monitoring.sql
│   │   └── model_performance_degradation.sql
│   ├── scoring/
│   └── validation/
│
├── src/
│   ├── config/
│   │   ├── load_config.py
│   │   └── pipeline_config.yaml
│   ├── data/
│   │   └── generate_synthetic_claims.py
│   ├── feedback/
│   │   └── simulate_investigation_feedback.py
│   ├── feature_store/
│   │   ├── refresh_provider_feature_history.py
│   │   └── refresh_short_term_provider_features.py
│   ├── models/
│   │   ├── isolation_forest.py
│   │   ├── train_random_forest.py
│   │   ├── run_random_forest_training.py
│   │   ├── train_xgboost.py
│   │   ├── run_xgboost_training.py
│   │   └── promote_model_candidate.py
│   ├── monitoring/
│   │   ├── generate_monitoring_sql.py
│   │   ├── run_model_performance_monitoring.py
│   │   └── run_random_forest_monitoring.py
│   ├── pipeline/
│   │   ├── model_training_pipeline.py
│   │   ├── run_model_training_pipeline.py
│   │   └── components/
│   │       ├── calculate_rolling_window_component.py
│   │       ├── check_retraining_decision_component.py
│   │       ├── model_promotion_gate_component.py
│   │       ├── refresh_provider_features_component.py
│   │       ├── register_model_component.py
│   │       ├── train_random_forest_component.py
│   │       └── train_xgboost_component.py
│   ├── scoring/
│   │   ├── run_batch_prediction.py
│   │   └── postprocess_predictions.py
│   ├── utils/
│   │   └── observability.py
│   └── validate_repo.sh
│
└── tests/
    ├── README.md
    ├── unit/
    │   ├── test_config_contracts.py
    │   ├── test_monitoring_sql_generation.py
    │   ├── test_promotion_gate.py
    │   ├── test_retraining_decision.py
    │   └── test_rolling_window.py
    └── integration/
        └── test_container_dependency_reproducibility.sh
```

## Code Orchestration — How the Files Work Together

For a first-time reader, follow the execution path from configuration to pipeline orchestration, model training, experiment tracking, registration, and approval.

The main runtime entry point is:

```text
src/pipeline/run_model_training_pipeline.py
```

The file-level orchestration is:

```text
src/config/pipeline_config.yaml
        |
        | model type
        | data month
        | rolling-window settings
        | feature definitions
        | hyperparameters
        | BigQuery tables
        | artifact paths
        | experiment settings
        v
src/config/load_config.py
        |
        v
src/pipeline/run_model_training_pipeline.py
        |
        | - loads configuration
        | - selects Random Forest or XGBoost
        | - creates feature signature
        | - creates model signature
        | - creates experiment_id and run_id
        | - captures Git commit
        | - resolves artifact locations
        | - submits the compiled pipeline
        v
model_training_pipeline.yaml
        ^
        |
        | compiled from
        |
src/pipeline/model_training_pipeline.py
        |
        | reusable Vertex AI DAG
        v
Check Retraining Decision
        |
        | check_retraining_decision_component.py
        |
        | evaluates:
        | - new data month
        | - feature signature change
        | - model signature change
        | - data quality alert
        | - feature drift alert
        | - prediction drift alert
        | - performance degradation alert
        v
Calculate Rolling Window
        |
        | calculate_rolling_window_component.py
        v
Refresh Provider Features
        |
        | refresh_provider_features_component.py
        |
        | rebuilds provider-level TRAIN / TEST
        | features in BigQuery
        v
     model_type?
      /       \
     /         \
Random Forest  XGBoost
     |             |
     v             v
train_random_   train_xgboost_
forest_         component.py
component.py
     |             |
     | launches    | launches
     | training    | training
     | container   | container
     v             v
run_random_     run_xgboost_
forest_         training.py
training.py
     |             |
     v             v
train_random_   train_xgboost.py
forest.py
     |             |
     +------+------+
            |
            | - load provider TRAIN / TEST data
            | - fit candidate model
            | - evaluate metrics
            | - save model artifact
            | - upload artifact to GCS
            | - write experiment metadata
            | - write experiment metrics
            v
BigQuery Experiment Tracking
            |
            +-- experiment_runs
            +-- experiment_metrics
            |
            v
model_promotion_gate_component.py
            |
            | compare challenger metrics with
            | the approved production run
            |
            | promotion eligible?
            |      /       \
            |     no       yes
            |     |         |
            |     v         v
            |   stop    register_model_component.py
            |               |
            |               | register candidate in
            |               | Vertex AI Model Registry
            |               v
            |       mark_candidate_pending_component.py
            |               |
            |               v
            |        PENDING_APPROVAL
            |
            | separate manual action
            v
src/models/promote_model_candidate.py
            |
            | - locate pending candidate
            | - resolve model artifact type
            | - copy candidate artifact to production
            | - update approved signatures/model type
            | - mark candidate APPROVED
            v
Production Model
```

The production monitoring path is separate from the training DAG:

```text
Cloud Scheduler
        |
        v
Cloud Run Job
        |
        v
src/monitoring/run_model_performance_monitoring.py
        |
        +-- provider_predictions
        +-- investigation_results
        +-- experiment_metrics
        |
        v
model_performance_degradation
        |
        +-- email summary
        |
        v
check_retraining_decision_component.py
        |
        v
daily training pipeline
```

## Where to Start Reading the Code

For a first pass through the repository, read the files in this order:

```text
1. src/config/pipeline_config.yaml
   Understand what a data scientist can configure.

2. src/pipeline/run_model_training_pipeline.py
   Understand how configuration becomes one pipeline execution.

3. src/pipeline/model_training_pipeline.py
   Understand the overall Vertex AI DAG.

4. src/pipeline/components/check_retraining_decision_component.py
   Understand why retraining is triggered.

5. src/pipeline/components/calculate_rolling_window_component.py
   Understand how rolling TRAIN / TEST windows are calculated.

6. src/pipeline/components/refresh_provider_features_component.py
   Understand the shared feature-engineering layer used by the training pipeline.

7. src/feature_store/refresh_provider_feature_history.py
   Understand the monthly 6-month provider profile refresh and historical feature snapshots.

8. src/feature_store/refresh_short_term_provider_features.py
   Understand the daily 7-day behavioral refresh and how the latest monthly profile is carried forward.

9. src/models/train_random_forest.py
   src/models/train_xgboost.py
   Understand the actual model-training and evaluation logic.

10. src/pipeline/components/model_promotion_gate_component.py
   Understand how a new challenger is compared with the approved production
   run before it becomes eligible for registration.

11. src/pipeline/components/register_model_component.py
   Understand candidate registration.

12. src/pipeline/components/mark_candidate_pending_component.py
    Understand the manual-approval boundary.

13. src/models/promote_model_candidate.py
    Understand how an approved candidate becomes production and how its
    run_id becomes the approved baseline for future model comparisons.

14. src/monitoring/run_model_performance_monitoring.py
    Understand how investigator feedback is used to evaluate production
    precision / recall / F1, write performance-degradation status, and
    send the scheduled monitoring email.
```

Most of the framework is model-independent:

```text
Shared
────────────────────────────────────────
pipeline_config.yaml
run_model_training_pipeline.py
model_training_pipeline.py
retraining decision
rolling window
feature refresh
experiment tracking
model registration
candidate state
manual promotion
scheduled performance monitoring
monitoring-driven retraining

Model-specific
────────────────────────────────────────
train_random_forest_component.py
run_random_forest_training.py
train_random_forest.py

or

train_xgboost_component.py
run_xgboost_training.py
train_xgboost.py
```

This separation lets a data scientist introduce a new model idea without rebuilding the rest of the governed MLOps workflow.

## Testing

The repository separates lightweight unit tests from environment-dependent integration tests.

- **Unit tests** validate retraining triggers, promotion-gate behavior, rolling train/test windows, model-specific configuration contracts, and configuration-driven monitoring SQL generation without requiring access to GCP resources.
- **Integration tests** validate runtime assumptions that depend on external infrastructure. The container reproducibility test compares each model image's installed direct dependencies against the pinned versions declared in its `requirements.txt`.

Run the unit tests with:

```bash
python -m pytest tests/unit -v
```

Run the container dependency reproducibility check with:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
bash tests/integration/test_container_dependency_reproducibility.sh
```

The integration test requires Docker access to the project's Artifact Registry images, while the unit-test suite is designed to run locally or in CI without GCP credentials.
