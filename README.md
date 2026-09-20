# GCP-FWA-MLOps-Accelerator

## Overview

This project demonstrates a reusable MLOps architecture for detecting potentially fraudulent healthcare providers/claims using Google Cloud Platform.

### Business goal: 
a reusable ML product that data scientists can plug in a new dataset, target, features, model techniques, and the same pipeline handles validation, training, evaluation, registration, deployment, monitoring, and retraining. Eventually support rapid experimentation without allowing each data scientist to build a separate production workflow.

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
- training logic
- feature engineering strategies
- modeling techniques
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
