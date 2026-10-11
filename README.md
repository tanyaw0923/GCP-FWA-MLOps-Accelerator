# GCP-FWA-MLOps-Accelerator

## Overview

This project demonstrates a reusable MLOps architecture for detecting potentially fraudulent healthcare providers/claims using Google Cloud Platform.

### Business goal: 
A reusable ML product that data scientists can plug in a new dataset, target, features, model techniques, and the same pipeline handles validation, training, evaluation, registration, deployment, monitoring, and retraining. Eventually support rapid experimentation without allowing each data scientist to build a separate production workflow.

### Users: 
The pipeline is designed for a team of talented and innovative data scientists who may experiment with different:

### Key Design Principles

The Shared MLOps Framework manages:
- data ingestion
- data validation
- shared feature generation
- pipeline orchestration CI/CD, ex. retrain using more recent data
- experiment tracking
- standardized model evaluation
- model registration
- production approval with manual approval
- deployment
- model monitoring & governance

Then Data scientists customize:
- model-specific feature engineering
- algorithm selection
- hyperparameters
- preprocessing
<<<<<<< HEAD
- training logic
- feature engineering strategies
- modeling techniques
=======
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

## Code Orchestration — How the Files Work Together

For a first-time reader, the easiest way to understand this repository is to follow the execution path from configuration → pipeline orchestration → model training → experiment tracking → model registration → manual promotion.

The main runtime entry point is:

\`\`\`text
src/pipeline/run_model_training_pipeline.py
\`\`\`

The file-level orchestration is:

\`\`\`text
src/config/pipeline_config.yaml
        |
        | model type, data month, features,
        | hyperparameters, BigQuery tables,
        | artifact paths, experiment settings
        v
src/config/load_config.py
        |
        v
src/pipeline/run_model_training_pipeline.py
        |
        | - loads configuration
        | - selects RF or XGBoost
        | - creates feature/model signatures
        | - creates experiment_id + run_id
        | - captures Git commit
        | - resolves artifact locations
        | - submits the compiled Vertex pipeline
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
        | src/pipeline/components/
        | check_retraining_decision_component.py
        |
        | checks:
        | - new data month
        | - feature signature change
        | - model configuration change
        | - data quality alert
        | - feature drift alert
        | - prediction drift alert
        v
Calculate Rolling Window
        |
        | calculate_rolling_window_component.py
        v
Refresh Provider Features
        |
        | refresh_provider_features_component.py
        | rebuilds provider-level TRAIN/TEST data in BigQuery
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
     +------+
            |
            | - load TRAIN/TEST features
            | - fit model
            | - evaluate PR-AUC / ROC-AUC /
            |   precision / recall
            | - save candidate artifact
            | - record experiment metadata
            | - record experiment metrics
            v
BigQuery Experiment Tracking
     |
     +-- fraud_experiments.experiment_runs
     +-- fraud_experiments.experiment_metrics
            |
            v
register_model_component.py
            |
            | register candidate in Vertex AI Model Registry
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
            | - find pending candidate
            | - copy candidate artifact to production
            | - update approved signatures/model type
            | - mark candidate APPROVED
            v
Production Model
\`\`\`

### Build-Time vs Run-Time Flow

Training images and pipeline execution are separate concerns:

\`\`\`text
BUILD TIME

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
\`\`\`

At run time:

\`\`\`text
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
\`\`\`

In short:

\`\`\`text
Cloud Build
    → builds the training environment

Vertex AI Pipelines
    → orchestrates when and how training runs

train_random_forest.py / train_xgboost.py
    → contain the actual model-training and evaluation logic
\`\`\`

### Where to Start Reading the Code

For a first pass through the repository, read the files in this order:

\`\`\`text
1. src/config/pipeline_config.yaml
   Understand what a data scientist can configure.

2. src/pipeline/run_model_training_pipeline.py
   Understand how configuration becomes one pipeline execution.

3. src/pipeline/model_training_pipeline.py
   Understand the overall Vertex AI orchestration DAG.

4. src/pipeline/components/check_retraining_decision_component.py
   Understand why retraining is triggered.

5. src/pipeline/components/calculate_rolling_window_component.py
   Understand how rolling TRAIN / TEST windows are determined.

6. src/pipeline/components/refresh_provider_features_component.py
   Understand the shared provider-level feature engineering layer.

7. src/models/train_random_forest.py
   src/models/train_xgboost.py
   Understand the actual ML training and evaluation logic.

8. src/pipeline/components/register_model_component.py
   Understand candidate registration in Vertex AI Model Registry.

9. src/pipeline/components/mark_candidate_pending_component.py
   Understand the manual-approval governance boundary.

10. src/models/promote_model_candidate.py
    Understand how an approved candidate becomes production.
\`\`\`

Most of the framework is model-independent:

\`\`\`text
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

Model-specific
────────────────────────────────────────
train_random_forest_component.py
run_random_forest_training.py
train_random_forest.py

or

train_xgboost_component.py
run_xgboost_training.py
train_xgboost.py
\`\`\`

This separation lets data scientists introduce a new model idea without rebuilding the rest of the governed MLOps workflow.

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
>>>>>>> 07a8a25 (Document code orchestration and reading order)
- hyperparameters

---

## Architecture (CD)
```text
                         Git Repository
                              |
                    push / merge to branch
                              |
                              v
                         Cloud Build
                    test + build + package
                              |
                              v
                      Artifact Registry
                              |
                              v
                    Vertex AI Pipeline
                              |
           +------------------+------------------+
           |                  |                  |
           v                  v                  v
    Data Validation     Shared Features    Custom Features
                                                |
                                                v
                                             Train
                                                |
                                                v
                                            Evaluate
                                                |
                                                v
                                      Candidate Model Comparison
                                                |
                                                v
                                        Register Candidate
                                                |
                                                v
                                       Vertex Model Registry
                                                |
                                                v
                                        Manual Approval
                                                |
                                                v
                                     Production Deployment
                                                |
                                                v
                                            Monitoring

```
---
## When new dataset becomes available(CI):

```text 

New File in Cloud Storage
          |
          v
       Eventarc
          |
          v
 Cloud Run Function
          |
    filename validation
          |
          v
 Vertex AI Pipeline

```

---

## Project Structure
```text 
fwa-provider-ml/
|
├── common/
│   ├── data_validation/
│   ├── evaluation/
│   ├── monitoring/
│   ├── feature_library/
│   └── utils/
|
├── models/
│   ├── provider_xgboost/
│   │   ├── features.py
│   │   ├── train.py
│   │   └── config.yaml
│   |
│   ├── provider_autoencoder/
│   │   ├── features.py
│   │   ├── train.py
│   │   └── config.yaml
│   |
│   └── provider_graph_model/
│       ├── features.py
│       ├── train.py
│       └── config.yaml
|
├── pipelines/
│   └── training_pipeline.py
|
├── triggers/
│   └── storage_trigger/
|
├── deployment/
│   └── deploy_model.py
|
├── tests/
|
├── cloudbuild.yaml
├── Dockerfile
└── README.md

```
