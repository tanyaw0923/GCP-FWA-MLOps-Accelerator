# GCP-Binary-MLOps-Accelerator

## Overview

This project demonstrates a reusable MLOps architecture for detecting potentially fraudulent healthcare providers/claims using Google Cloud Platform.

Business goal: a reusable ML product that data scientists can plug in a new dataset, target, features, model techniques, and the same pipeline handles validation, training, evaluation, registration, deployment, monitoring, and retraining. Eventually support rapid experimentation without allowing each data scientist to build a separate production workflow.

Users: The pipeline is designed for a team of talented and innovative data scientists who may experiment with different:

- feature engineering strategies
- modeling techniques
- hyperparameters

while sharing a common framework for:

- data validation
- model evaluation
- CI/CD, ex. retrain using more recent data
- model registration
- manual approval
- deployment
- model monitoring
- governance

---

## Architecture

```text (CD)
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
                                    Champion Comparison
                                                |
                                                v
                                      Register Candidate
                                                |
                                                v
                                      Vertex Model Registry
                                                |
                                        Manual Approval
                                                |
                                                v
                                       Production Deployment
                                                |
                                                v
                                            Monitoring

When new dataset becomes available, to trigger CI:

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


Key Design Principles
1. Shared MLOps Framework

The central/shared platform manages:
- data ingestion
- data validation
- shared feature generation
- pipeline orchestration
- experiment tracking
- standardized evaluation
- model registration
- production approval
- deployment
- monitoring

Then Data scientists customize:
- model-specific feature engineering
- algorithm selection
- hyperparameters
- preprocessing
- training logic

Repo Structure:
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
