# GCP FWA MLOps Accelerator

A reusable Google Cloud MLOps framework for provider-level healthcare fraud, waste, and abuse (FWA) modeling.

This project is designed as a shared ML accelerator for a team of data scientists. Data scientists can experiment with new data, features, model configurations, and algorithms while reusing the same governed workflow for retraining decisions, feature refresh, training, evaluation, experiment tracking, candidate registration, approval, and production promotion.

#### ▶▶▶Click the image below to watch the video on YouTube◀◀◀
[![Watch the demo](https://img.youtube.com/vi/9gN_HCLjT6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=9gN_HCLjT6k)

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

## Architecture Overview

<img width="1998" height="1125" alt="image" src="https://github.com/user-attachments/assets/f3647476-e04e-44e6-b4d4-c84a5b64d422" />

---

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
                 Model Promotion Gate
                          |
                    Promotion eligible?
                     /            \
                   No              Yes
                   |                |
                   v                v
          Keep production     Register Candidate
          model unchanged            |
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

## Model Promotion Gate

The Model Promotion Gate separates **training success** from **promotion eligibility**.

Every challenger is trained and evaluated first, and its run metadata and evaluation metrics are written to BigQuery. The gate then compares the new run with the currently approved production run using the configured promotion criteria.

Current default guardrails:

```text
PR-AUC change     >=  0.00
Recall change     >=  0.00
F1 change         >=  0.00
Precision change  >= -0.05
```

The precision rule intentionally allows a controlled tradeoff: for an imbalanced fraud-detection problem, improved recall can justify a small precision decrease, but the gate prevents precision from degrading without limit.

The decision flow is:

```text
Train challenger
       |
       v
Evaluate challenger
       |
       v
Write experiment_runs / experiment_metrics
       |
       v
Compare with approved_run_id
       |
       v
Model Promotion Gate
      / \
     /   \
  FAIL   PASS
   |       |
   v       v
Keep    Register
current candidate
model      |
           v
      PENDING_APPROVAL
```

A failed challenger is still preserved in the experiment tables for analysis and reproducibility. It simply does not advance into Model Registry or the manual-approval workflow.

The training-state table tracks:

```text
candidate_run_id
approved_run_id
```

When a candidate is manually promoted, its `candidate_run_id` becomes the new `approved_run_id`. Future retraining attempts are compared against that approved run.


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

The monitoring workflow runs twice per month:

#### Example Email Output Attached Below:
[FWA Model Performance Monitoring - ALERT email.pdf](https://github.com/user-attachments/files/33799524/FWA.Model.Performance.Monitoring.-.ALERT.email.pdf)

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

## Monitoring Email Configuration

Production performance monitoring can send a summary email when these environment variables are set:

```bash
export EMAIL_ADDRESS="sender@gmail.com"
export EMAIL_PASSWORD="GOOGLE_APP_PASSWORD"
export MONITORING_EMAIL="recipient@gmail.com"
```

For Gmail, `EMAIL_PASSWORD` must be a Google App Password, not the normal Gmail sign-in password. Email delivery uses SMTP over implicit TLS on port 465. A notification failure is logged but does not invalidate a monitoring result that was already written to BigQuery.


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
- **Cloud Run Jobs** — serverless execution environment for scheduled feature refreshes and performance monitoring
- **Cloud Scheduler** — invokes the monthly long-term feature refresh, daily short-term feature refresh, and scheduled performance monitoring
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

The repository uses versioned custom Docker images for Random Forest and XGBoost training and serving, plus a dedicated performance-monitoring image. The model-specific serving containers enforce the configured feature contract at prediction time.

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

## Configuration-Driven Experimentation

The primary configuration lives in:

```text
src/config/pipeline_config.yaml
```

The currently configured dataset version is controlled by:

```yaml
data:
  current_data_month: "2026-09-01"
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

## Reuse This MLOps Infrastructure for New Experiments

### Minimum changes for a new experiment using an existing model family

For example, if a second fraud experiment still uses Random Forest but targets a different provider pattern, the intended change path is:

```text
1. pipeline_config.yaml
   - experiment_name
   - active features
   - feature SQL / derived feature SQL
   - validation rules
   - table names if the experiment uses separate datasets
   - hyperparameters / artifact paths if needed

2. Source data
   - load the new claims / events into the configured source table

3. Regenerate monitoring SQL
   python -m src.monitoring.generate_monitoring_sql

4. Recompile the KFP pipeline only if pipeline/component code changed
   python -m src.pipeline.model_training_pipeline

5. Run tests
   python -m pytest tests/unit -v

6. Launch the experiment
   python -m src.pipeline.run_model_training_pipeline
```

The design goal is that a normal feature/model experiment should **not** require edits to the orchestration code, serving code, or monitoring generator.

### Adding a completely new model type

Adding a model family beyond `random_forest` and `xgboost` requires more work because model training and serving are algorithm-specific. In addition to adding a new `model.<model_type>` block in `pipeline_config.yaml`, add or update:

```text
src/models/train_<new_model>.py
src/pipeline/components/train_<new_model>_component.py
src/pipeline/model_training_pipeline.py
docker/<new-model>/
docker/<new-model>-serving/
cloudbuild-<new-model>.yaml
cloudbuild-<new-model>-serving.yaml
tests/
```

The new branch in `model_training_pipeline.py` should still reuse the shared components for:

```text
retraining decision
rolling train/test window
feature refresh
experiment tracking
promotion gate
Vertex Model Registry
candidate state
manual approval
monitoring
```

After changing pipeline topology or component definitions, recompile:

```bash
python -m src.pipeline.model_training_pipeline
```

and commit the updated `model_training_pipeline.yaml`.

### Feature changes

When adding a new model feature, update the experiment in this order:

```text
pipeline_config.yaml
  -> feature_sql or derived_feature_sql
  -> model.<model_type>.features
  -> monitoring.feature_validation_rules
```

Then regenerate monitoring SQL:

```bash
python -m src.monitoring.generate_monitoring_sql
```

The generator expands the data-quality null/invalid checks and feature-drift `UNPIVOT` list from the active model feature contract. Batch scoring and the custom serving container also consume that same configured feature list, reducing duplicated hard-coded feature definitions.

### Moving the accelerator to another GCP project

At minimum, update:

```text
src/config/pipeline_config.yaml
cloudbuild-random-forest.yaml
cloudbuild-random-forest-serving.yaml
cloudbuild-xgboost.yaml
cloudbuild-xgboost-serving.yaml
cloudbuild-performance-monitoring.yaml
```

Replace project IDs, bucket names, BigQuery table paths, Artifact Registry image URIs, and Feature Store resource IDs. Then rebuild the required images, regenerate monitoring SQL, recompile the pipeline if needed, and rerun the unit/integration checks before launching the experiment.

## Synthetic Demo Data

The demo uses 100,000 synthetic eye-disease claims from primary care providers in Texas during 2026.

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

## Container Image Versioning

Custom project images use explicit semantic-style versions rather than floating tags.

```text
Random Forest training
random-forest-training:v1.0.0

Random Forest serving
random-forest-serving:v1.0.1

XGBoost training
xgboost-training:v1.0.0

XGBoost serving
xgboost-serving:v1.0.1

Performance monitoring
performance-monitoring:v1.0.0
```

The version is declared as the Cloud Build substitution `_IMAGE_VERSION`, so a new container release can be built without editing the build steps.

For example:

```bash
gcloud builds submit \
  --config cloudbuild-xgboost.yaml \
  --substitutions=_IMAGE_VERSION=v1.1.0 \
  .
```

For model-training images, the image URI is also part of the model signature. Changing the selected training image therefore changes the candidate signature and triggers the governed retraining workflow.

The KFP container component references a pinned image version at compile time, so when a training-image version changes, update the corresponding image version in `pipeline_config.yaml` and the KFP component, then recompile `model_training_pipeline.yaml`.

## End-to-End Demo Runbook

The following commands reflect the current demo workflow from candidate training through manual approval, model lookup, batch inference, investigation, and monitoring.

### 1. Pull the Latest Main Branch

```bash
git checkout main
git pull origin main
```

### 2. Modify `pipeline_config.yaml`

Before running the pipeline, review:

```text
src/config/pipeline_config.yaml
```

Update the configuration for the experiment you want to run. Common changes include:

- `model.model_type` — choose `random_forest` or `xgboost`
- `data.current_data_month` — set the dataset month used by the rolling train/test windows
- `model.<model_type>.features` — select the feature contract for training and batch inference
- `model.<model_type>.parameters` — adjust model hyperparameters when testing a new modeling idea
- training or serving image versions when a new container version is released

Because `pipeline_config.yaml` is the shared source of truth, these settings drive retraining signatures, model training, serving registration, and batch-scoring feature selection.

### 3. Build the Required Container Images

Build the training image when the model-training code or training environment changes.

Random Forest training:

```bash
gcloud builds submit \
  --config cloudbuild-random-forest.yaml \
  .
```

XGBoost training:

```bash
gcloud builds submit \
  --config cloudbuild-xgboost.yaml \
  .
```

Build the serving image when prediction logic or the serving environment changes.

Random Forest serving:

```bash
gcloud builds submit \
  --config cloudbuild-random-forest-serving.yaml \
  .
```

XGBoost serving:

```bash
gcloud builds submit \
  --config cloudbuild-xgboost-serving.yaml \
  .
```

You do not need to rebuild every image for every run. Rebuild only the image affected by your code or environment change.

### 4. Compile and Run the Candidate Training Pipeline

Compile the Vertex AI pipeline after any serving-image or component change:

```bash
python -m src.pipeline.model_training_pipeline
```

Run the candidate training pipeline:

```bash
python -m src.pipeline.run_model_training_pipeline
```

### 5. Manually Promote the Candidate

After reviewing the candidate metrics and confirming the model is ready for production:

```bash
python -m src.models.promote_model_candidate
```

### 6. List Registered Vertex Model IDs

List all models in the project:

```bash
gcloud ai models list \
  --project=fwa-mlops-accelerator-demo \
  --region=us-central1
```

Inspect a specific model:

```bash
gcloud ai models describe MODEL_ID \
  --project=fwa-mlops-accelerator-demo \
  --region=us-central1 \
  --format="yaml(displayName,artifactUri,containerSpec.imageUri,containerSpec.predictRoute,containerSpec.healthRoute)"
```

### 7. Batch-Scoring Input Is Generated Automatically

No separate preparation command is required; `run_batch_prediction.py` materializes the scoring input from the configured feature contract.

### 8. Run Batch Inference

Run scoring with the selected Vertex model:

```bash
python -m src.scoring.run_batch_prediction \
  --model-id MODEL_ID
```

For the cleanest lineage, explicitly pass the approved training run:

```bash
python -m src.scoring.run_batch_prediction \
  --model-id MODEL_ID \
  --run-id APPROVED_RUN_ID
```

The approved run can be retrieved from the training-state table:

```bash
bq query --use_legacy_sql=false '
SELECT
  approved_model_type,
  approved_run_id,
  last_training_timestamp AS approved_at
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_training_state`
WHERE
  model_name = "provider_fraud_model"
'
```

### 9. Create the Investigation Queue

```bash
bq query \
  --use_legacy_sql=false \
  < sql/feedback/create_investigation_queue.sql
```

The queue is model-independent and prioritizes the top-scoring providers according to investigator capacity.

### 10. Simulate Investigator Feedback

Run the normal simulation using synthetic ground truth:

```bash
python -m src.feedback.simulate_investigation_feedback
```

To intentionally demonstrate production-performance degradation, use a degradation rate. This flips a reproducible share of true-positive, model-predicted-positive providers to `CONFIRMED_NORMAL`, creating additional false positives and lowering monitored precision:

```bash
python -m src.feedback.simulate_investigation_feedback \
  --degradation-rate 0.30
```

Use `--seed` to make the simulated degradation reproducible. The default seed is `42`.

```bash
python -m src.feedback.simulate_investigation_feedback \
  --degradation-rate 0.30 \
  --seed 42
```

This degradation mode is for demo/testing only; real production monitoring should use actual investigator, SIU, or audit outcomes.

### 11. Evaluate Business Value

```bash
bq query \
  --use_legacy_sql=false \
  < sql/evaluation/model_business_value.sql
```

### 12. Run Data / Feature / Prediction Monitoring

```bash
bq query \
  --quiet \
  --format=none \
  --use_legacy_sql=false \
  < sql/monitoring/model_monitoring.sql
```

### 13. Run Investigator-Feedback Performance Monitoring

```bash
python -m src.monitoring.run_model_performance_monitoring
```

## Original Fraud Modeling Foundation

The project began with an Isolation Forest workflow for identifying anomalous provider behavior without requiring fraud labels.

Investigation feedback was then used to support supervised fraud modeling with Random Forest and XGBoost.

The current architecture preserves that experimentation history while focusing on a reusable, governed continuous-training workflow.

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

## What's Missing from Production

This repository demonstrates an end-to-end batch MLOps lifecycle, but it is intentionally not positioned as a complete enterprise production platform. A larger production implementation would add the following controls.

### 1. Automated Rollback Process

The current workflow supports candidate registration, manual approval, and promotion of a model artifact to production. A production system should also support an explicit rollback path.

A stronger rollback design would include:

- retain immutable versions of previously approved production artifacts
- store the exact model version, feature version, dataset version, container image version, and Git commit for each production release
- allow operators to mark a prior approved model as the active production version
- restore both the model artifact and its associated serving / scoring configuration
- verify rollback health with post-deployment validation before traffic or batch scoring resumes
- preserve the rollback event in the model-governance audit trail

Conceptually:

```text
Production model v3
        |
        | monitoring / operational issue
        v
Rollback decision
        |
        v
Previously approved model v2
        |
        v
Restore artifact + config + lineage
        |
        v
Validation
        |
        v
Resume production scoring
```

### 2. Automated CI/CD Test Gates

The repository now includes a dedicated `tests/` hierarchy with unit coverage for rolling-window logic, retraining decisions, promotion-gate behavior, configuration contracts, monitoring SQL generation, and an Artifact Registry container-dependency reproducibility check. The remaining production gap is automated CI orchestration: these checks are not yet enforced by a GitHub Actions or equivalent pull-request workflow.

A production CI/CD gate should automatically run and enforce:

- the existing unit-test suite
- configuration and monitoring-SQL contract validation
- KFP pipeline compilation checks
- model-training and batch-scoring smoke tests
- Docker image build validation
- dependency and security scanning
- deployment to a non-production environment before production promotion

A failed test gate should prevent the model, image, or pipeline change from advancing.

```text
Code / config change
        |
        v
CI tests
        |
        +-- unit tests
        +-- schema tests
        +-- pipeline compile
        +-- image build
        +-- scoring contract tests
        |
        v
All gates pass?
     /       \
   No         Yes
   |           |
   v           v
 Stop      Deploy candidate
```

### 3. Large-Scale Feature Governance

The current Feature Store layer demonstrates timestamped provider history, reusable feature definitions, and long-term / short-term feature refreshes. At enterprise scale, feature governance would need stronger controls.

A larger implementation would typically add:

- centralized ownership and stewardship for each feature
- feature descriptions, business definitions, and approved use cases
- versioned feature definitions
- lineage from raw source to derived feature
- freshness and quality SLAs
- deprecation policies
- access controls by team / domain
- feature discovery and reuse across multiple models
- automated validation for training-serving consistency
- point-in-time correctness checks
- audit history for changes to feature logic

This prevents multiple teams from independently creating slightly different versions of the same business concept and improves reproducibility across models.

### 4. IAM and Role-Based Access Control

The demo focuses on the ML lifecycle rather than enterprise security administration. A production deployment should use least-privilege IAM and clearly separate responsibilities between data scientists, platform engineers, investigators, and model approvers.

Example responsibilities could be separated as follows:

```text
Data Scientist
  - read approved training data
  - create experiments
  - submit candidate training jobs
  - cannot directly promote to production

ML Platform / Service Account
  - execute pipelines
  - read / write required BigQuery tables
  - access model artifacts
  - register candidate models

Model Approver
  - review candidate metrics
  - approve / reject production promotion

Fraud Investigator
  - access investigation queue
  - write investigation outcomes
  - no model-training permissions

Production Operator
  - manage deployment / rollback
  - restricted access to production resources
```

Additional production controls would include:

- dedicated service accounts per workload
- least-privilege BigQuery dataset permissions
- restricted Artifact Registry and Cloud Storage access
- Secret Manager access only for required services
- separation of development and production projects
- audit logging for model registration, approval, promotion, and rollback

These capabilities would strengthen the platform for enterprise deployment without changing the core design principle of the accelerator: model-specific data science can evolve while the governed MLOps workflow remains reusable.

## Current Scope

This repository is a technical MLOps demonstration rather than a complete enterprise production platform.

The current workflow focuses on:

- batch-oriented fraud modeling
- two-speed provider feature refreshes: monthly long-term profile plus daily 7-day behavior
- Vertex AI Feature Store registration and future online-serving path
- configurable feature generation
- retraining decisions
- model training and evaluation
- experiment tracking
- automated Model Promotion Gate
- candidate registration
- manual approval
- artifact promotion
- model-independent, capacity-based investigation queue
- top-K operational evaluation and confirmed fraud-dollar capture
- scheduled investigator-feedback performance monitoring
- continuous fraud-score distribution drift monitoring
- monitoring email notifications
- structured Cloud Logging operational observability
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
