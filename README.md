# GCP FWA MLOps Accelerator

A reusable Google Cloud MLOps framework for provider-level healthcare fraud, waste, and abuse (FWA) detection.

The project demonstrates how data scientists can experiment with new data, features, and modeling approaches while reusing the same governed training workflow for feature refresh, model training, evaluation, registration, approval, and retraining decisions.

## Business Goal

The goal is to provide a shared ML accelerator for a team of data scientists rather than having every model owner build a separate production workflow.

The framework separates two responsibilities:

**Shared MLOps framework**
- rolling train/test window management
- reusable provider-level feature generation
- retraining trigger logic
- standardized training orchestration
- candidate model registration
- manual approval before production promotion
- model state tracking
- monitoring and feedback integration
- containerized training and reproducible builds

**Data scientist customization**
- feature definitions
- algorithm selection
- hyperparameters
- preprocessing
- model-specific training logic
- new modeling ideas and experiments

This design allows data, features, and modeling techniques to evolve independently while the same governed MLOps workflow is reused.

## Demo Scenario

The repository supports three retraining use cases.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/31a80327-2da5-4865-9902-bf6b5e9d701d" />

---

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/591a8d49-8f89-43c9-a104-74847c37b540" />

### Use Case 1 — New Data

A new monthly claims dataset becomes available.

The pipeline detects that the latest data month is newer than the approved production training state and retrains the candidate model using refreshed rolling windows.

Example:

```text
September 2026
TRAIN: Jan-Jun
TEST:  Jul-Sep

October 2026
TRAIN: Feb-Jul
TEST:  Aug-Oct
```

### Use Case 2 — New Features

The data month remains the same, but the configured feature set changes.

The launcher generates a feature signature from the configured feature definitions. When the current feature signature differs from the approved production signature, retraining is triggered.

Example feature added in this demo:

```text
avg_member_provider_distance
```

### Use Case 3 — New Model Idea

The data and features remain unchanged, but a data scientist wants to test a new modeling approach.

The demo switches the configured model type from:

```text
Random Forest
     ↓
XGBoost
```

A model signature is generated from the model type, model name, and hyperparameters. When the signature differs from the approved production model signature, the same training pipeline is reused to train and register a new candidate.

This is the primary interview demo scenario because it shows that model experimentation can occur without rebuilding the rest of the MLOps workflow.

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
          Evaluate Model      Evaluate Model
                |                   |
                +---------+---------+
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

The pipeline can trigger retraining from several independent signals:

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

This allows an approved Random Forest model and an XGBoost candidate to coexist within the same governed workflow.

## Google Cloud Services

The implementation uses:

- **BigQuery** — claims data, provider-level features, monitoring outputs, and training state
- **Vertex AI Pipelines** — reusable model training orchestration
- **Vertex AI Model Registry** — candidate model registration
- **Cloud Storage** — candidate and production model artifacts
- **Artifact Registry** — custom training container images
- **Cloud Build** — reproducible Random Forest and XGBoost training image builds
- **Kubeflow Pipelines SDK** — component and pipeline definitions

## Model Artifacts

Random Forest and XGBoost use separate artifact formats:

```text
Random Forest
models/random_forest/candidate/model.joblib
models/random_forest/production/model.joblib

XGBoost
models/xgboost/candidate/model.bst
models/xgboost/production/model.bst
```

The model registry stores both:

```text
artifact_uri
    → location of the trained model artifact in Cloud Storage

serving_container_image_uri
    → Vertex AI prediction container capable of loading the model
```

The custom Docker images in this repository are training images. They are separate from the Vertex AI serving containers used when registering models.

## Repository Structure

```text
GCP-FWA-MLOps-Accelerator/
├── README.md
├── model_training_pipeline.yaml
├── cloudbuild-random-forest.yaml
├── cloudbuild-xgboost.yaml
│
├── docker/
│   ├── random_forest/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── xgboost/
│       ├── Dockerfile
│       └── requirements.txt
│
├── sql/
│   ├── features/
│   ├── feedback/
│   ├── monitoring/
│   ├── scoring/
│   ├── validation/
│   └── supporting SQL scripts
│
└── src/
    ├── config/
    │   ├── load_config.py
    │   └── pipeline_config.yaml
    │
    ├── data/
    │   ├── generate_synthetic_claims.py
    │   ├── exploratory_analysis.py
    │   └── isolation_forest_sanity_check.py
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
    │   └── run_random_forest_monitoring.py
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

## Configuration-Driven Experimentation

The primary model configuration lives in:

```text
src/config/pipeline_config.yaml
```

For Use Case 3, the core experiment can be expressed by changing:

```yaml
model:
  model_type: xgboost
```

The launcher then selects the corresponding:
- model name
- hyperparameters
- candidate artifact location
- Vertex serving container
- model signature

The rest of the pipeline remains reusable.

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

```bash
python -m src.pipeline.run_model_training_pipeline
```

The launcher:
1. loads the YAML configuration
2. generates feature and model signatures
3. resolves model-specific artifact paths
4. submits the compiled pipeline to Vertex AI
5. waits for the pipeline result
6. leaves the resulting candidate in `PENDING_APPROVAL`

## Manual Candidate Promotion

After reviewing the candidate model:

```bash
python -m src.models.promote_model_candidate
```

The promotion workflow:
1. finds the pending candidate
2. determines its model type
3. selects the correct artifact format
4. copies the candidate artifact to the production path
5. updates the approved feature/model signatures and model type
6. marks the candidate as approved

This keeps model promotion separate from training and prevents a newly trained candidate from automatically replacing the approved production model.

## Original Fraud Modeling Foundation

The project began with an Isolation Forest workflow for identifying anomalous provider behavior without requiring fraud labels.

Investigation feedback was then used to create supervised labels and expand the framework to Random Forest and XGBoost models.

The current architecture preserves that experimentation history while focusing on a reusable continuous-training framework for supervised and unsupervised fraud modeling.

## Current Scope

This repository is a technical demonstration of an MLOps accelerator rather than a complete enterprise production platform.

The implemented workflow focuses on batch-oriented fraud modeling, candidate registration, artifact promotion, monitoring, and retraining governance. A production implementation could further add automated testing gates, richer experiment tracking, IAM hardening, CI/CD deployment triggers, online endpoints where appropriate, and additional model families.

## Key Takeaway

> Data, features, and modeling techniques can evolve independently while the same governed MLOps workflow is reused.

The accelerator lets data scientists spend more time on fraud strategy, feature design, experimentation, and model evaluation while shared infrastructure handles the repeatable MLOps workflow.
