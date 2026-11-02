# Architecture and ML lifecycle

> Detailed documentation extracted from the original project README. Run commands from the repository root unless otherwise stated.

## Architecture Overview

<img width="1998" height="1125" alt="image" src="https://github.com/user-attachments/assets/f3647476-e04e-44e6-b4d4-c84a5b64d422" />

---

The repository demonstrates three retraining use cases.

<img width="1672" height="941" alt="MLOps accelerator architecture" src="https://github.com/user-attachments/assets/31a80327-2da5-4865-9902-bf6b5e9d701d" />

---

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/dc5b3eb5-b340-47d3-b4d5-bc6ea1e8d178" />

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

## Google Cloud Services

The implementation uses:

- **BigQuery** — claims data, provider features, experiment tracking, monitoring outputs, and training state
- **Vertex AI Pipelines** — reusable training orchestration
- **Vertex AI Model Registry** — candidate model registration
- **Cloud Storage** — candidate and production model artifacts
- **Artifact Registry** — custom Random Forest and XGBoost training images
- **Cloud Build** — reproducible training and monitoring image builds
- **Cloud Run Jobs** — serverless execution environment for scheduled feature refreshes and performance monitoring
- **Cloud Scheduler** — invokes the monthly long-term feature refresh, daily short-term feature refresh, and scheduled performance monitoring
- **Secret Manager** — stores the Gmail App Password used by the monitoring job
- **Kubeflow Pipelines SDK** — pipeline and component definitions

## Original Fraud Modeling Foundation

The project began with an Isolation Forest workflow for identifying anomalous provider behavior without requiring fraud labels.

Investigation feedback was then used to support supervised fraud modeling with Random Forest and XGBoost.

The current architecture preserves that experimentation history while focusing on a reusable, governed continuous-training workflow.

## Continuous Training Flow

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/55074adc-edf4-4ca6-a4e8-d7c9d2b639a8" />


Every training run is still recorded in the BigQuery experiment tables before the promotion decision is made. A challenger that does not satisfy the promotion criteria remains available for analysis, but it is not registered as a candidate.

Production remains unchanged until a promotion-eligible candidate is explicitly approved.

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

## Vertex AI Feature Store and Two-Speed Features

The repository contains two related feature layers with different responsibilities.

The **active supervised model path** uses `fraud_features.provider_eye_features`. The Vertex training pipeline rebuilds this table from raw claims for the configured rolling TRAIN / TEST windows, and batch scoring uses the TEST feature contract from the same table. This keeps the current Random Forest and XGBoost training and scoring paths consistent.

Separately, `fraud_features.provider_feature_history` is a timestamped historical feature layer used to demonstrate reusable provider history, two-speed feature refreshes, point-in-time feature storage, and a future Vertex AI Feature Store / online-serving path. It is **not currently the source table used by the supervised training pipeline**. Keeping that distinction explicit avoids claiming a training-serving integration that the current demo does not yet implement.

For the interview demo, the feature layer uses two refresh speeds:

```text
Long-term provider profile
  trailing 6 months
  refreshed monthly

Short-term provider behavior
  trailing 7 days
  refreshed daily
```

Both horizons are written into the same timestamped BigQuery history so the latest provider row contains a stable monthly profile plus fresh daily behavior.

```text
Monthly job (1st of month, 6:00 AM)
        |
        v
refresh_provider_feature_history.py
        |
        +--> trailing 6-month provider profile
        |
        v
fraud_features.provider_feature_history

Daily job (6:30 AM)
        |
        v
refresh_short_term_provider_features.py
        |
        +--> trailing 7-day behavior
        +--> carries forward latest monthly profile
        |
        v
fraud_features.provider_feature_history
        |
        +--> historical feature analysis / backtesting
        |
        +--> Vertex AI Feature Group
        |      eye_provider_fraud_features
        |
        +--> Feature View / Bigtable online store
               future real-time scoring
```

### Monthly long-term refresh

The long-term provider profile is refreshed on the **1st of every month at 6:00 AM America/Edmonton**.

Cloud Scheduler cron:

```text
0 6 1 * *
```

The job uses the previous completed month-end as the snapshot date.

```text
November 1  -> October 31 snapshot
March 1     -> February 28 / 29 snapshot
January 1   -> December 31 snapshot
```

Run manually with:

```bash
python -m src.feature_store.refresh_provider_feature_history
```

or backfill a specific month-end:

```bash
python -m src.feature_store.refresh_provider_feature_history \
  --snapshot-date 2026-09-30
```

### Daily short-term refresh

The short-term behavioral layer is refreshed **daily at 6:30 AM America/Edmonton**.

Cloud Scheduler cron:

```text
30 6 * * *
```

The daily job defaults to **yesterday** so only completed claim data is used.

For example, when the job runs on October 23:

```text
snapshot date = October 22
7-day window  = October 16-22
```

Run manually with:

```bash
python -m src.feature_store.refresh_short_term_provider_features
```

or for a specific date:

```bash
python -m src.feature_store.refresh_short_term_provider_features \
  --snapshot-date 2026-10-22
```

On the 1st of the month, the monthly job runs first at 6:00 AM and the daily job runs at 6:30 AM. This lets the daily snapshot use the newly refreshed long-term provider profile.

### Long-term provider profile

The long-term feature layer uses the trailing **6 months** of provider claims to capture relatively stable behavior.

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

### Short-term behavioral features

The short-term feature layer uses the trailing **7 days** to capture recent changes in provider behavior.

```text
claim_count_7d
high_acuity_pct_7d
referral_rate_7d
avg_paid_per_claim_7d
avg_member_provider_distance_7d
high_acuity_oe_ratio_7d
```

These short-term features are stored for experimentation and future scoring designs; they are not currently included in the active supervised model feature list.

```text
referral_rate       -> trailing 6-month baseline
referral_rate_7d    -> trailing 7-day behavior
```

A future model or monitoring rule could compare the two to identify providers whose recent behavior has shifted from their longer-term pattern.

### Canonical feature history

Both jobs write to:

```text
fraud_features.provider_feature_history
```

with:

```text
entity_id = provider_id
feature_timestamp = snapshot timestamp
```

The monthly job refreshes the long-term provider profile. The daily job calculates fresh 7-day behavior and carries forward the latest available long-term values into the daily row.

This produces a point-in-time history such as:

```text
Provider A

2026-09-30
  referral_rate      = monthly long-term value
  referral_rate_7d   = month-end 7-day value

2026-10-01
  referral_rate      = carried forward from Sep 30
  referral_rate_7d   = recalculated for Sep 25-Oct 1

2026-10-02
  referral_rate      = carried forward from Sep 30
  referral_rate_7d   = recalculated for Sep 26-Oct 2
```

### Why Feature Store is used

The current demo uses the Feature Store layer to show how provider features can be stored and reused beyond a single training run:

- maintain timestamped provider feature history
- support point-in-time feature inspection and backtesting
- keep long-term and short-term provider behavior in one reusable history
- provide a path to a registered Vertex AI Feature Group and future online serving

For the **active batch model**, BigQuery `provider_eye_features` remains the execution contract for TRAIN / TEST and batch scoring. The Feature Store history is a parallel reusable layer rather than a second hidden training path.

This design choice is intentional for the interview demo: the production workflow is runnable today, while the Feature Store layer demonstrates how the architecture can evolve toward centralized point-in-time training and online feature serving without pretending that integration is already complete.

### Current Feature Store resources

```text
Feature Group:
eye_provider_fraud_features

Online Store:
provider_fraud_online_store

Feature View:
eye_provider_fraud_feature_view
```

The resulting architecture stays simple enough for an interview demo while clearly separating the runnable batch-model contract from the reusable historical / future-online Feature Store layer.

## Experiment Tracking

Each successful training execution records reproducibility metadata and model metrics in BigQuery, regardless of whether the trained model later passes the Model Promotion Gate.

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
f1_score
```

Related runs intentionally share the same `experiment_id` when they belong to the same experiment family. For example, multiple XGBoost retraining attempts use the same XGBoost experiment identifier, while each execution receives a unique `run_id`.

```text
provider_eye_fraud_xgboost
        |
        +-- experiment_id: same experiment family
        |
        +-- run_id: good XGBoost retraining run
        |
        +-- run_id: weaker XGBoost challenger run
```

This makes it easy to compare multiple model attempts side by side in BigQuery while preserving a unique identifier for every training execution.

## Model Promotion Gate

The Model Promotion Gate separates **training success** from **promotion eligibility**.

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/032ce02e-cb6e-4df9-ae1a-5e8028ed688e" />

Every challenger is trained and evaluated first, and its run metadata and evaluation metrics are written to BigQuery. The gate then compares the new run with the currently approved production run using the configured promotion criteria.

Current default guardrails:

```text
PR-AUC change     >=  0.00
Recall change     >=  0.00
F1 change         >=  0.00
Precision change  >= -0.05
```

The precision rule intentionally allows a controlled tradeoff: for an imbalanced fraud-detection problem, improved recall can justify a small precision decrease, but the gate prevents precision from degrading without limit.

A failed challenger is **still preserved** in the experiment tables for analysis and reproducibility. It simply does not advance into Model Registry or the manual-approval workflow.

The training-state table tracks:

```text
candidate_run_id
approved_run_id
```

When a candidate is manually promoted, its `candidate_run_id` becomes the new `approved_run_id`. Future retraining attempts are compared against that approved run.

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

The repository uses versioned custom Docker images for Random Forest and XGBoost training and serving, plus a dedicated performance-monitoring image. The model-specific serving containers enforce the configured feature contract at prediction time.

## Batch Inference and Provider Lineage

`pipeline_config.yaml` is the single source of truth for the model feature contract.

For the selected model type, `run_batch_prediction.py` reads:

```text
model.<model_type>.features
```

and dynamically materializes one shared scoring table:

```text
fraud_features.model_batch_input
```

from the provider-level feature table. The schema is rebuilt immediately before each batch job as:

```text
provider_id + exactly the configured model features
```

This means a feature added to Random Forest or XGBoost in `pipeline_config.yaml` automatically flows into training, serving registration, and batch inference without editing model-specific scoring SQL.

`provider_id` is lineage metadata, not a model feature. Vertex batch prediction uses object instances with:

```text
excluded_fields = ["provider_id"]
```

so the serving container receives only configured model features while `provider_id` is attached back to the BigQuery prediction output.

```text
pipeline_config.yaml
        |
        v
model.<model_type>.features
        |
        +-----------------------+
        |                       |
        v                       v
training                 dynamic batch input
                                |
                                v
provider_id + configured features
                                |
                                v
Vertex AI Batch Prediction
provider_id excluded from model instance
                                |
                                v
model-specific serving app
RF predict_proba / XGBoost predict
                                |
                                v
BigQuery current-job output
provider_id + prediction
                                |
                                v
fraud_experiments.provider_predictions
```

The scoring script validates that the registered Vertex model carries the same feature contract and serving image as `pipeline_config.yaml`, fails fast on row-level prediction errors, and never reuses an older `predictions_*` table.

Run batch inference with:

```bash
python -m src.scoring.run_batch_prediction \
  --model-id <VERTEX_MODEL_ID> \
  --run-id <APPROVED_RUN_ID>
```

The `--run-id` argument is optional, but using the approved run explicitly gives the cleanest model-governance lineage for the demo.

## Investigation Capacity and Business Value

Fraud operations are capacity constrained, so the model is evaluated as a ranking system in addition to standard PR-AUC / precision / recall / F1 metrics.

The investigation queue is model-independent:

```text
Latest curated provider_predictions run
        |
        v
Rank providers by fraud_score
        |
        v
Top 50 investigation queue
        |
        +-- P1: ranks 1-10
        +-- P2: ranks 11-25
        +-- P3: ranks 26-50
```

This works the same way whether the current scored model is Random Forest or XGBoost.

Operational evaluation lives in:

```text
sql/evaluation/model_business_value.sql
```

and reports model value at investigator capacities:

```text
K = 10
K = 20
K = 50
```

with:

- **precision@K** — fraction of selected providers that are fraud
- **recall@K** — fraction of the known fraud population captured within investigator capacity
- **lift@K** — precision@K divided by the underlying fraud prevalence
- **confirmed fraud dollars captured@K** — value already confirmed through investigation feedback

For this synthetic demo, recall@K and lift@K use the synthetic provider label from the claim generator. In production, the same framework would use matured SIU, audit, recovery, or adjudication outcomes and would account for label delay and investigation-selection bias.

## Scheduled Performance Monitoring

Production performance monitoring is intentionally separated from the Vertex AI training pipeline.

<img width="1644" height="957" alt="image" src="https://github.com/user-attachments/assets/768177d2-792f-4d41-a461-282520bc47f3" />

#### Example Email Attached Below:
[FWA Model Performance Monitoring - ALERT email.pdf](https://github.com/user-attachments/files/33799524/FWA.Model.Performance.Monitoring.-.ALERT.email.pdf)

---
The Cloud Run Job provides the execution environment for the custom Python monitoring code. It reads scored providers and completed investigation outcomes from BigQuery, calculates a confusion matrix and production precision / recall / F1, compares those metrics with the training baseline for the same run, and writes the result to:

```text
fraud_monitoring.model_performance_degradation
```

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

## Monitoring Email Configuration

Production performance monitoring can send a summary email when these environment variables are set:

```bash
export EMAIL_ADDRESS="sender@gmail.com"
export EMAIL_PASSWORD="GOOGLE_APP_PASSWORD"
export MONITORING_EMAIL="recipient@gmail.com"
```

For Gmail, `EMAIL_PASSWORD` must be a Google App Password, not the normal Gmail sign-in password. Email delivery uses SMTP over implicit TLS on port 465. A notification failure is logged but does not invalidate a monitoring result that was already written to BigQuery.

## Operational Observability with Cloud Logging

The pipeline uses structured Python logging for operational observability across the main training, batch-scoring, and production-monitoring paths.

The reusable helper is:

```text
src/utils/observability.py
```

It sends structured operational events directly to Cloud Logging with `log_struct()` and falls back to standard Python logging when Cloud Logging is unavailable. This keeps the terminal output clean for interactive demos while avoiding the background-thread shutdown behavior of `CloudLoggingHandler`. This keeps local / Cloud Shell execution usable without making observability a hard runtime dependency.

Operational events include:

```text
Training orchestration
  - training_pipeline_submission_started
  - training_pipeline_submitted
  - training_pipeline_state_changed
  - training_pipeline_completed
  - training_pipeline_failed

Batch scoring
  - batch_scoring_started
  - batch_model_resolved
  - batch_input_validated
  - batch_lineage_resolved
  - batch_prediction_completed
  - batch_row_errors
  - batch_output_validated
  - batch_scoring_completed

Performance monitoring
  - performance_monitoring_started
  - performance_monitoring_completed
  - performance_monitoring_write_failed
  - monitoring_email_sent
  - monitoring_email_skipped
  - monitoring_email_failed
```

Structured fields include values such as:

```text
run_id
experiment_id
model_id
model_version
pipeline_job
batch_job
input_rows
prediction_rows
error_rows
monitoring_status
alert_reason
```

Claim-level payloads, member identifiers, and other sensitive healthcare data should not be written to application logs.

### Enable Cloud Logging

Enable the API:

```bash
gcloud services enable logging.googleapis.com
```

For Cloud Shell or another interactive Python environment, install the logging client if it is not already available:

```bash
pip install google-cloud-logging
```

Managed workload service accounts should have permission to write log entries, typically through `roles/logging.logWriter` or an equivalent custom role containing `logging.logEntries.create`.

The managed performance-monitoring container includes `google-cloud-logging` in its requirements. Rebuild that image after pulling this change:

```bash
gcloud builds submit \
  --config cloudbuild-performance-monitoring.yaml \
  .
```

Cloud Logging complements, rather than replaces, model monitoring:

```text
Operational observability
Cloud Logging
  -> pipeline execution
  -> serving / scoring failures
  -> job state changes
  -> operational alerts

Model observability
BigQuery monitoring tables
  -> data quality
  -> feature drift
  -> prediction drift
  -> production performance
```
