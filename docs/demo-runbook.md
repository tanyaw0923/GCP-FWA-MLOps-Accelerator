# Demo runbook and data

> Detailed documentation extracted from the original project README. Run commands from the repository root unless otherwise stated.

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
