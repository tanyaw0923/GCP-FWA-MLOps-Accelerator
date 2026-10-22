## Metadata Tracking in the Training Pipeline

This branch adds explicit experiment metadata publishing to the XGBoost training workflow.

<img width="726" height="901" alt="image" src="https://github.com/user-attachments/assets/66b9097f-2693-42c5-8880-910e8c989519" />

After the candidate model is trained, the pipeline publishes the run metadata and evaluation context before model registration. This makes each model candidate easier to trace back to the exact configuration, code version, data window, and feature set used to create it.

The metadata step captures information such as:

- experiment name and run ID
- model parameters
- evaluation metrics
- feature signature
- model signature
- training data window
- Git commit version
- model artifacts and metric outputs

These details are also stored in the BigQuery experiment-tracking tables:

```text
fraud_experiments.experiment_runs
fraud_experiments.experiment_metrics
```

This gives the project two complementary views of the same experiment history:

```text
Vertex AI Metadata
        ↓
Run lineage and artifact relationships

BigQuery Experiment Tables
        ↓
Tabular analysis, comparison, reporting, and monitoring
```

The BigQuery tables make it easy to analyze experiment history across runs—for example, comparing model versions, feature configurations, dataset windows, hyperparameters, and evaluation metrics using SQL.

The overall flow is:

```text
Retraining Decision
        ↓
Calculate Rolling Window
        ↓
Refresh Provider Features
        ↓
Train XGBoost Candidate
        ↓
Publish XGBoost Metadata
        ↓
Write Experiment Runs / Metrics to BigQuery
        ↓
Register XGBoost Candidate
        ↓
Mark Candidate PENDING_APPROVAL
```

The purpose of the metadata layer is not to change the model-training logic. It improves reproducibility, traceability, governance, and experiment analysis by making it easier to answer questions such as:

```text
Which data window trained this model?
Which feature set was used?
Which hyperparameters were applied?
What metrics did the candidate achieve?
Which Git version produced the run?
How did this run compare with previous experiments?
```

This is especially useful when multiple data scientists are experimenting with different data, features, and model configurations while sharing the same reusable MLOps workflow.
