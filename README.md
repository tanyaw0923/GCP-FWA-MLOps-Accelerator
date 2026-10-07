
### Use Case 1 Business Context
---
Data Updated from 2026-09 to 2026-10 

→ retraining triggered

→ TRAIN window [6 months] = 2026-02-01 to 2026-07-31

→ TEST window [3 months] = 2026-08-01 to 2026-10-31

→ provider-level features rebuilt with new month of data

→ Random Forest candidate retrained using the updated dataset

→ candidate evaluated on the same TEST window

→ candidate registered in Vertex AI Model Registry

→ candidate_status = PENDING_APPROVAL

→ current production model remains unchanged

→ new candidate is promoted only after manual approval

### Use Case 1 Architect
---
<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/50f09ab9-9262-433c-a01a-e1489529a467" />


## Repo Structure
```text 
GCP-FWA-MLOps-Accelerator/
│
├── README.md
├── .gitignore
├── random_forest_dockerfile
├── requirements-training.txt
├── random_forest_training_pipeline.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── load_config.py
│   │   └── pipeline_config.yaml
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── generate_synthetic_claims.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_random_forest.py
│   │   ├── run_random_forest_training.py
│   │   ├── promote_random_forest_candidate.py
│   │   └── isolation_forest.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── random_forest_training_pipeline.py
│   │   ├── run_random_forest_pipeline.py
│   │   │
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── calculate_rolling_window_component.py
│   │       ├── check_retraining_decision_component.py
│   │       ├── refresh_provider_features_component.py
│   │       ├── train_random_forest_component.py
│   │       ├── register_random_forest_model_component.py
│   │       └── mark_candidate_pending_component.py
│   │
│   └── sql/
│       ├── provider_feature_engineering.sql
│       ├── create_random_forest_retraining_dataset.sql
│       ├── check_random_forest_retraining_trigger.sql
│       ├── random_forest_data_quality.sql
│       ├── random_forest_feature_drift.sql
│       └── random_forest_prediction_drift.sql
│
├── model_artifacts/
│   ├── isolation_forest/
│   │   └── model.joblib
│   └── random_forest/
│       └── model.joblib
│
└── docs/
    ├── architecture/
    └── demo/
```
