# GCP FWA MLOps Accelerator

A reusable Google Cloud MLOps framework for provider-level healthcare fraud, waste, and abuse (FWA) modeling.

This project is designed as a shared ML accelerator for a team of data scientists. Data scientists can experiment with new data, features, model configurations, and algorithms while reusing the same governed workflow for retraining decisions, feature refresh, training, evaluation, experiment tracking, candidate registration, approval, and production promotion.

## Business Goal

The goal is to avoid having every data scientist build a separate production workflow.

The shared MLOps framework manages:

- rolling train/test windows
- provider-level feature generation
- retraining trigger logic
- reusable Vertex AI orchestration
- experiment tracking
- standardized model evaluation
- candidate model registration
- manual approval before production promotion
- model state tracking
- monitoring inputs
- scheduled production performance monitoring
- investigator-feedback performance evaluation
- email monitoring summaries
- containerized and reproducible model training

Data scientists customize:

- feature definitions
- algorithm selection
- hyperparameters
- preprocessing
- model-specific training logic
- new modeling ideas and experiments

The key design principle is:

> Data, features, and modeling techniques can evolve independently while the same governed MLOps workflow is reused.

## Demo Scenario

The repository demonstrates three retraining use cases.

<img width="1672" height="941" alt="MLOps accelerator architecture" src="https://github.com/user-attachments/assets/31a80327-2da5-4865-9902-bf6b5e9d701d" />

---

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/dc5b3eb5-b340-47d3-b4d5-bc6ea1e8d178" />


<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/032ce02e-cb6e-4df9-ae1a-5e8028ed688e" />

---

<img width="1644" height="957" alt="image" src="https://github.com/user-attachments/assets/768177d2-792f-4d41-a461-282520bc47f3" />

#### Example Email Attached Below:
[FWA Model Performance Monitoring - ALERT email.pdf](https://github.com/user-attachments/files/33799524/FWA.Model.Performance.Monitoring.-.ALERT.email.pdf)

---

### Use Case 1 — New Data

A new monthly claims dataset becomes available.

The retraining decision compares the current configured data month with the approved production training state. If the month is newer, the pipeline refreshes the rolling train/test windows and trains a new candidate.

Example:

```text
September 2026
TRAIN: Jan-Jun
TEST:  Jul-Sep

October 2026
TRAIN: Feb-Jul
TEST:  Aug-Oct
```

### Use Case 2 — New Feature

The data month stays the same, but the feature configuration changes.

The launcher generates a feature signature from:

- `feature_sql`
- `derived_feature_sql`
- the configured model feature list

If the current feature signature differs from the approved feature signature, retraining is triggered.

The feature introduced in this demo is:

```text
avg_member_provider_distance
```

It is created from the claim-level field:

```text
member_provider_distance
        ↓
AVG(member_provider_distance)
        ↓
avg_member_provider_distance
```

### Use Case 3 — New Model Idea

The data and features remain unchanged, but a data scientist wants to test a different modeling approach.

The demo switches:

```text
Random Forest
     ↓
XGBoost
```

The launcher creates a model signature from the model type, model name, and hyperparameters. If the model signature differs from the approved production signature, the same pipeline trains a new candidate without rebuilding the rest of the MLOps workflow.

## Continuous Training Flow

```text
                  Retraining Decision
                          |
                          v
                  Rolling Data Window
                          |
                          v
                Provider Feature Refresh
                          |
                +---------+---------+
                |                   |
                v                   v
         Random Forest          XGBoost
                |                   |
                v                   v
              Train               Train
                |                   |
                v                   v
             Evaluate            Evaluate
                |                   |
                +---------+---------+
                          |
                          v
                 Experiment Tracking
                          |
                          v
                 Register Candidate
                          |
                          v
                  Vertex Model Registry
                          |
                          v
                   PENDING_APPROVAL
                          |
                          v
                   Manual Approval
                          |
                          v
             Promote Candidate Artifact
                          |
                          v
                    Production Model
```

Production remains unchanged until the candidate is explicitly approved.

## Retraining Decision

Retraining can be triggered by several independent signals:

```text
New monthly data
       OR
Feature configuration changed
       OR
Model configuration changed
       OR
Data quality alert
       OR
Feature drift alert
       OR
Prediction drift alert
       OR
Performance degradation alert
       ↓
Should Retrain = True
```

The training-state record represents the ML product rather than a specific algorithm:

```text
provider_fraud_model
        |
        +-- approved_model_type
        +-- approved_model_signature
        +-- approved_feature_signature
        |
        +-- candidate_model_type
        +-- candidate_model_signature
        +-- candidate_feature_signature
        +-- candidate_status
```

This allows, for example, an approved Random Forest model and an XGBoost candidate to coexist in the same governed workflow.

## Scheduled Performance Monitoring

Production performance monitoring is intentionally separated from the Vertex AI training pipeline.

The monitoring workflow runs twice per month:

```text
Cloud Scheduler
1st and 15th at 7:00 AM
America/Edmonton
        |
        v
Cloud Run Job
fwa-performance-monitoring
        |
        v
run_model_performance_monitoring.py
        |
        +--> provider_predictions
        |
        +--> investigation_results
        |
        +--> experiment_metrics
        |
        v
Calculate production precision / recall / F1
        |
        v
Compare with training baseline
        |
        v
fraud_monitoring.model_performance_degradation
        |
        +--> PASS / ALERT / INSUFFICIENT_DATA
        |
        +--> investigator feedback summary
        |
        v
Email monitoring report
```

The Cloud Scheduler job is:

```text
fwa-performance-monitoring-schedule
```

with cron:

```text
0 7 1,15 * *
```

and timezone:

```text
America/Edmonton
```

The Cloud Run Job provides the execution environment for the custom Python monitoring code. It reads scored providers and completed investigation outcomes from BigQuery, calculates a confusion matrix and production precision / recall / F1, compares those metrics with the training baseline for the same run, writes the result to:

```text
fraud_monitoring.model_performance_degradation
```

and emails a summary containing:

- providers reviewed
- confirmed fraud / confirmed normal
- fraud yield
- confirmed fraud amount
- TP / FP / TN / FN
- current precision / recall / F1
- training baseline metrics
- absolute percentage-point changes
- relative performance changes
- monitoring status and alert reason

The current degradation thresholds are relative to the training baseline:

```text
Precision: 10%
Recall:    15%
F1:        10%
```

If the result is `ALERT`, the monitoring job does not retrain the model directly. The daily continuous-training pipeline reads the latest performance status as another retraining signal and decides whether a new candidate should be created.

This keeps monitoring and training decoupled:

```text
Scheduled monitoring
        |
        v
Write model health status
        |
        v
Daily training pipeline
        |
        v
Check Retraining Decision
        |
        v
ALERT -> train new candidate
        |
        v
PENDING_APPROVAL
```

Email credentials are not stored in Git. The Cloud Run Job receives the sender and recipient as environment variables and reads the Gmail App Password from Secret Manager.

## Experiment Tracking

Each successful training execution records reproducibility metadata and model metrics in BigQuery.

The launcher creates:

```text
experiment_id
    stable ID derived from experiment_name

run_id
    unique ID for one pipeline execution

feature_version
    current feature signature

dataset_version
    current_data_month

git_commit
    current Git commit SHA
```

The two primary tables are:

```text
fraud_experiments.experiment_runs
fraud_experiments.experiment_metrics
```

A successful run records one experiment-run row plus metric rows such as:

```text
pr_auc
roc_auc
precision
recall
```

## Demo Data Exports

The `data_exports/` folder contains CSV snapshots of the key BigQuery tables used throughout the demo so reviewers can inspect the data flow without needing access to the GCP project.

The exports cover the main stages of the pipeline:

```text
Raw claims
    ↓
claims.csv

Provider-level feature engineering
    ↓
provider_eye_features.csv

Investigation feedback
    ↓
investigation_results.csv

Experiment tracking
    ↓
experiment_runs.csv
experiment_metrics.csv

Model scoring
    ↓
provider_predictions.csv

Monitoring
    ↓
model_data_quality.csv
model_feature_drift.csv
model_prediction_drift.csv
model_performance_degradation.csv

Model governance / approval state
    ↓
model_training_state.csv
```

These files are demo snapshots of the corresponding BigQuery tables:

| Export | BigQuery table | Purpose |
| --- | --- | --- |
| `claims.csv` | `fwa_claims.claims` | Synthetic claim-level source data |
| `provider_eye_features.csv` | `fraud_features.provider_eye_features` | Provider-level TRAIN / TEST feature snapshot |
| `investigation_results.csv` | `fraud_feedback.investigation_results` | Investigator feedback and confirmed labels |
| `experiment_runs.csv` | `fraud_experiments.experiment_runs` | Experiment lineage and run metadata |
| `experiment_metrics.csv` | `fraud_experiments.experiment_metrics` | PR-AUC, ROC-AUC, precision, recall, and other run metrics |
| `provider_predictions.csv` | `fraud_experiments.provider_predictions` | Provider-level prediction output |
| `model_data_quality.csv` | `fraud_monitoring.model_data_quality` | Latest data-quality monitoring results |
| `model_feature_drift.csv` | `fraud_monitoring.model_feature_drift` | Feature-drift monitoring results |
| `model_prediction_drift.csv` | `fraud_monitoring.model_prediction_drift` | Prediction-drift monitoring results |
| `model_performance_degradation.csv` | `fraud_monitoring.model_performance_degradation` | Investigator-feedback performance monitoring, baseline comparison, and PASS / ALERT status |
| `model_training_state.csv` | `fraud_monitoring.model_training_state` | Approved and candidate model state, signatures, and approval status |

The CSV files are for demonstration and repository review only. BigQuery remains the runtime source of truth for pipeline execution.

## Google Cloud Services

The implementation uses:

- **BigQuery** — claims data, provider features, experiment tracking, monitoring outputs, and training state
- **Vertex AI Pipelines** — reusable training orchestration
- **Vertex AI Model Registry** — candidate model registration
- **Cloud Storage** — candidate and production model artifacts
- **Artifact Registry** — custom Random Forest and XGBoost training images
- **Cloud Build** — reproducible training and monitoring image builds
- **Cloud Run Jobs** — serverless execution environment for scheduled performance monitoring
- **Cloud Scheduler** — invokes performance monitoring on the 1st and 15th of each month
- **Secret Manager** — stores the Gmail App Password used by the monitoring job
- **Kubeflow Pipelines SDK** — pipeline and component definitions

## Model Artifacts

Random Forest and XGBoost use different artifact formats:

```text
Random Forest
models/random_forest/candidate/model.joblib
models/random_forest/production/model.joblib

XGBoost
models/xgboost/candidate/model.bst
models/xgboost/production/model.bst
```

The custom Docker images in this repository are training images. They are separate from the Vertex AI serving containers referenced when registering models.

## Repository Structure

```text
GCP-FWA-MLOps-Accelerator/
├── .gitignore
├── README.md
├── cloudbuild-random-forest.yaml
├── cloudbuild-xgboost.yaml
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
│   ├── random_forest/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── xgboost/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── performance-monitoring/
│       ├── Dockerfile
│       └── requirements.txt
│
├── sql/
│   ├── features/
│   ├── feedback/
│   ├── monitoring/
│   ├── scoring/
│   └── validation/
│
└── src/
    ├── config/
    │   ├── load_config.py
    │   └── pipeline_config.yaml
    │
    ├── data/
    │   └── generate_synthetic_claims.py
    │
    ├── feedback/
    │   └── simulate_investigation_feedback.py
    │
    ├── models/
    │   ├── isolation_forest.py
    │   ├── train_random_forest.py
    │   ├── run_random_forest_training.py
    │   ├── train_xgboost.py
    │   ├── run_xgboost_training.py
    │   └── promote_model_candidate.py
    │
    ├── monitoring/
    │   ├── run_random_forest_monitoring.py
    │   └── run_model_performance_monitoring.py
    │
    ├── pipeline/
    │   ├── model_training_pipeline.py
    │   ├── run_model_training_pipeline.py
    │   └── components/
    │
    └── scoring/
        ├── run_batch_prediction.py
        └── postprocess_predictions.py
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
register_model_component.py
            |
            | register candidate in
            | Vertex AI Model Registry
            v
mark_candidate_pending_component.py
            |
            v
PENDING_APPROVAL
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

## Build-Time vs Run-Time Flow

Training-image creation and pipeline execution are separate concerns.

### Build time

```text
cloudbuild-random-forest.yaml
        ↓
docker/random_forest/Dockerfile
        ↓
Random Forest training image
        ↓
Artifact Registry


cloudbuild-xgboost.yaml
        ↓
docker/xgboost/Dockerfile
        ↓
XGBoost training image
        ↓
Artifact Registry
```

### Run time

```text
pipeline_config.yaml
        ↓
run_model_training_pipeline.py
        ↓
Vertex AI Pipeline
        ↓
train_*_component.py
        ↓
previously built training image
        ↓
run_*_training.py
        ↓
train_*.py
```

In short:

```text
Cloud Build
    → builds the model-training environment

Vertex AI Pipelines
    → orchestrates when and how training runs

train_random_forest.py / train_xgboost.py
    → contain the model-specific ML logic
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
   Understand the shared feature-engineering layer.

7. src/models/train_random_forest.py
   src/models/train_xgboost.py
   Understand the actual model-training and evaluation logic.

8. src/pipeline/components/register_model_component.py
   Understand candidate registration.

9. src/pipeline/components/mark_candidate_pending_component.py
   Understand the manual-approval boundary.

10. src/models/promote_model_candidate.py
    Understand how an approved candidate becomes production.

11. src/monitoring/run_model_performance_monitoring.py
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

## Configuration-Driven Experimentation

The primary configuration lives in:

```text
src/config/pipeline_config.yaml
```

The currently configured dataset version is controlled by:

```yaml
data:
  current_data_month: "2026-10-01"
```

The active model is controlled by:

```yaml
model:
  model_type: random_forest
```

For Use Case 3, changing only:

```yaml
model:
  model_type: xgboost
```

causes the launcher to select the corresponding model name, hyperparameters, artifact locations, serving container, and model signature while reusing the rest of the pipeline.

The shared provider feature set currently includes:

```text
high_acuity_pct
pct_99214
pct_99215
referral_rate
eye_procedure_pct
avg_paid_per_claim
high_acuity_oe_ratio
avg_member_provider_distance
```

## Synthetic Demo Data

The repository includes a synthetic claim generator:

```text
src/data/generate_synthetic_claims.py
```

The synthetic data intentionally includes overlap between normal and suspicious provider behavior so model performance is meaningful rather than trivially perfect.

Generate the local CSV with:

```bash
python -m src.data.generate_synthetic_claims
```

The generated file can then be loaded into the configured BigQuery claims table.

## Build Training Images

Random Forest:

```bash
gcloud builds submit \
  --config cloudbuild-random-forest.yaml \
  .
```

XGBoost:

```bash
gcloud builds submit \
  --config cloudbuild-xgboost.yaml \
  .
```

Both builds use the repository root as the Docker build context.

## Compile the Vertex AI Pipeline

From the repository root:

```bash
python -m src.pipeline.model_training_pipeline
```

This generates:

```text
model_training_pipeline.yaml
```

## Run the Candidate Training Pipeline

Run:

```bash
python -m src.pipeline.run_model_training_pipeline
```

The launcher:

1. loads `pipeline_config.yaml`
2. selects the active model configuration
3. generates feature and model signatures
4. creates experiment and run identifiers
5. captures the Git commit
6. resolves artifact locations and runtime parameters
7. submits the compiled pipeline to Vertex AI
8. waits for the pipeline result
9. records successful training metadata and metrics
10. leaves the resulting candidate in `PENDING_APPROVAL`

## Manual Candidate Promotion

After reviewing a candidate:

```bash
python -m src.models.promote_model_candidate
```

The promotion workflow:

1. finds the pending candidate
2. determines its model type
3. selects the appropriate artifact format
4. copies the candidate artifact to the production path
5. updates approved feature/model signatures and model type
6. marks the candidate as approved

Training and production promotion are intentionally separate. A newly trained candidate does not automatically replace the approved production model.

## Original Fraud Modeling Foundation

The project began with an Isolation Forest workflow for identifying anomalous provider behavior without requiring fraud labels.

Investigation feedback was then used to support supervised fraud modeling with Random Forest and XGBoost.

The current architecture preserves that experimentation history while focusing on a reusable, governed continuous-training workflow.

## Current Scope

This repository is a technical MLOps demonstration rather than a complete enterprise production platform.

The current workflow focuses on:

- batch-oriented fraud modeling
- configurable feature generation
- retraining decisions
- model training and evaluation
- experiment tracking
- candidate registration
- manual approval
- artifact promotion
- scheduled investigator-feedback performance monitoring
- monitoring email notifications
- monitoring-driven retraining signals

A larger production implementation could add automated test gates, stronger IAM controls, CI/CD deployment automation, richer experiment comparison, approval workflows, additional model families, and online serving where appropriate.

## Common Code Issue

If Cloud Shell Python ADC fails with:

```text
Unexpected response from metadata server:
service account info is missing 'email' field
```

a local Application Default Credentials login can be used:

```bash
gcloud auth application-default login
```

## Key Takeaway

> Data, features, and modeling techniques can evolve independently while the same governed MLOps workflow is reused.

The accelerator lets data scientists spend more time on fraud strategy, feature design, experimentation, and model evaluation while shared infrastructure handles the repeatable MLOps workflow.
