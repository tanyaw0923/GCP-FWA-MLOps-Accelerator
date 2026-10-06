## Overview

This project demonstrates a reusable MLOps architecture for detecting potentially fraudulent healthcare providers/claims using Google Cloud Platform.

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/44fc006d-9fc8-453d-bd13-1d4abe37e41e" />

### Business goal: 
A reusable ML product that data scientists can plug in a new dataset, target, features, model techniques, and the same pipeline handles validation, training, evaluation, registration, deployment, monitoring, and retraining. Eventually support rapid experimentation without allowing each data scientist to build a separate production workflow.

### Users: 
The pipeline is designed for a team of talented and innovative data scientists who can utilize this tool then focus on exciting new things to try

---
#### Use Case 1
<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/50f09ab9-9262-433c-a01a-e1489529a467" />

Data Updated from 2026-09 to 2026-10
→ retraining triggered
→ TRAIN window [6 MONTH] = 2026-02-01 to 2026-07-31
→ TEST window [3 MONTH] = 2026-08-01 to 2026-10-31
→ provider features rebuilt
→ Random Forest candidate trained
→ candidate evaluated
→ candidate registered
→ candidate_status = PENDING_APPROVAL
→ production remains unchanged


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

## Project Structure
```text 
fwa-fraud-mlops-accelerator/
├── .gitignore
├── README.md
├── random_forest_dockerfile
├── requirements-training.txt
├── sql/
│   ├── add_exploration_feedback.sql
│   ├── check_random_forest_retraining_trigger.sql
│   ├── create_batch_scoring_input.sql
│   ├── create_investigation_queue.sql
│   ├── create_open_investigation_queue.sql
│   ├── create_random_forest_retraining_dataset.sql
│   ├── data_validation.sql
│   ├── ...
│   └── xgboost_validation.sql
└── src/
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   ├── load_config.py
    │   └── pipeline_config.yaml
    ├── data/
    │   ├── exploratory_analysis.py
    │   ├── generate_synthetic_claims.py
    │   └── isolation_forest_sanity_check.py
    ├── feedback/
    │   └── simulate_investigation_feedback.py
    ├── models/
    │   ├── __init__.py
    │   ├── isolation_forest.py
    │   ├── run_random_forest_training.py
    │   └── train_random_forest.py
    ├── monitoring/
    │   ├── __init__.py
    │   └── run_random_forest_monitoring.py
    ├── pipeline/
    │   ├── __init__.py
    │   ├── random_forest_training_pipeline.py
    │   ├── random_forest_downstream_qa_pipeline.py
    │   ├── run_random_forest_pipeline.py
    │   ├── run_random_forest_downstream_qa_pipeline.py
    │   └── components/
    │       ├── __init__.py
    │       ├── train_random_forest_component.py
    │       ├── register_random_forest_model_component.py
    │       ├── run_random_forest_batch_prediction_component.py
    │       ├── postprocess_random_forest_predictions_component.py
    │       ├── run_random_forest_monitoring_component.py
    │       ├── check_retraining_decision_component.py
    │       ├── refresh_provider_features_component.py
    │       └── update_training_state_component.py
    └── scoring/
        └── run_batch_prediction.py
```
