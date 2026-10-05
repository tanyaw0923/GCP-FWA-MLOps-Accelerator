#!/usr/bin/env python
# coding: utf-8

# In[53]:


from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
import joblib
import pandas as pd

from google.cloud import bigquery, storage

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from src.config.load_config import load_config


# In[55]:


#configuration
config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]

MODEL_TYPE = config["model"]["model_type"]
MODEL_NAME = config["model"]["model_name"]
EXPERIMENT_NAME = config["model"]["experiment_name"]
MODEL_PARAMS = config["model"]["parameters"]

FEATURE_COLUMNS = config["features"]

FEATURE_TABLE = config["data"]["feature_table"]
FEEDBACK_TABLE = config["data"]["feedback_table"]

EXPERIMENT_RUNS_TABLE = config["data"]["experiment_runs_table"]
EXPERIMENT_METRICS_TABLE = config["data"]["experiment_metrics_table"]

ARTIFACT_DIR = Path("model_artifacts") / MODEL_TYPE
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

DATASET_VERSION = "investigator_feedback_v3"

print("\nPipeline configuration loaded")
print("Project:", PROJECT_ID)
print("Region:", REGION)
print("Model:", MODEL_TYPE)
print("Experiment:", EXPERIMENT_NAME)
print("Features:", FEATURE_COLUMNS)



# In[30]:


#initialize connection
bq = bigquery.Client(project=PROJECT_ID)

experiment_id = f"{MODEL_TYPE}_{EXPERIMENT_NAME}"

run_id = (
    f"{MODEL_TYPE}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
    f"{uuid.uuid4().hex[:6]}"
)

created_at = datetime.now(timezone.utc)

print("\nExperiment ID:", experiment_id)
print("Run ID:", run_id)


# In[31]:


#load confirmed investigation label as target variable
training_query = f"""
WITH latest_feedback AS (
  SELECT *
  FROM (
    SELECT
      provider_id,
      confirmed_fraud,
      investigation_result,
      investigator,
      investigation_date,
      created_at,
      ROW_NUMBER() OVER (
        PARTITION BY provider_id
        ORDER BY investigation_date DESC, created_at DESC
      ) AS rn
    FROM `{FEEDBACK_TABLE}`
    WHERE confirmed_fraud IS NOT NULL
  )
  WHERE rn = 1
)

SELECT
  f.provider_id,
  f.high_acuity_pct,
  f.pct_99214,
  f.pct_99215,
  f.referral_rate,
  f.eye_procedure_pct,
  f.avg_paid_per_claim,
  f.high_acuity_oe_ratio,

  CAST(feedback.confirmed_fraud AS INT64) AS fraud_label,
  feedback.investigation_result,
  feedback.investigator

FROM `{FEATURE_TABLE}` f
JOIN latest_feedback feedback
USING(provider_id)

WHERE f.dataset_split = 'TEST'
"""

training_df = bq.query(training_query).to_dataframe()

if training_df.empty:
    raise ValueError("No labeled providers found for Random Forest training.")

print("\nTraining dataset loaded")
print("Providers:", len(training_df))
print(
    training_df["fraud_label"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Normal", 1: "Fraud"})
)


# In[32]:


#prepare training and target variables
X = training_df[FEATURE_COLUMNS].astype(float)
y = training_df["fraud_label"].astype(int)

fraud_count = int((y == 1).sum())
normal_count = int((y == 0).sum())

if y.nunique() < 2:
    raise ValueError(
        "Training dataset must contain both fraud and normal labels."
    )

print("\nLabel summary")
print("Fraud:", fraud_count)
print("Normal:", normal_count)


# In[33]:


#define random forest model

model = RandomForestClassifier(
    n_estimators=MODEL_PARAMS["n_estimators"],
    max_depth=MODEL_PARAMS["max_depth"],
    min_samples_leaf=MODEL_PARAMS["min_samples_leaf"],
    class_weight=MODEL_PARAMS["class_weight"],
    random_state=MODEL_PARAMS["random_state"],
    n_jobs=MODEL_PARAMS["n_jobs"],
)

print("\nRandom Forest parameters")
print(json.dumps(MODEL_PARAMS, indent=2))


# In[34]:


#conduct cross validation
minimum_class_count = int(y.value_counts().min())

if minimum_class_count < 2:
    raise ValueError(
        "Not enough observations in the minority class for "
        "stratified cross-validation."
    )

n_splits = min(5, minimum_class_count)

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=MODEL_PARAMS["random_state"],
)

cv_probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=MODEL_PARAMS["n_jobs"],
)[:, 1]

cv_predictions = (cv_probabilities >= 0.5).astype(int)


# In[44]:


#model evaluation
pr_auc = average_precision_score(y, cv_probabilities)
roc_auc = roc_auc_score(y, cv_probabilities)
precision = precision_score(y, cv_predictions, zero_division=0)
recall = recall_score(y, cv_predictions, zero_division=0)

print("\n==============================================")
print("Random Forest Cross-Validation Metrics")
print("==============================================")

print(f"\nPR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")


# In[45]:


#metrics results look good therefore train the model
model.fit(X, y)

print("\nFinal Random Forest model trained.")
print("Number of features:", model.n_features_in_)


# In[46]:


import os
import joblib

ARTIFACT_DIR.mkdir(parents=True,exist_ok=True,)

joblib.dump(model,MODEL_PATH,)

print("\nModel artifact saved:")
print(MODEL_PATH)


# In[47]:


#check feature importance
importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n==============================================")
print("Feature Importance")
print("==============================================\n")

print(importance_df.to_string(index=False))


# In[48]:


#save model metadata
metadata = {
    "project_id": PROJECT_ID,
    "region": REGION,
    "experiment_id": experiment_id,
    "run_id": run_id,
    "experiment_name": EXPERIMENT_NAME,
    "model_type": MODEL_TYPE,
    "model_name": MODEL_NAME,
    "dataset_version": DATASET_VERSION,
    "created_at": created_at.isoformat(),
    "training_provider_count": len(training_df),
    "fraud_count": fraud_count,
    "normal_count": normal_count,
    "features": FEATURE_COLUMNS,
    "model_parameters": MODEL_PARAMS,
    "metrics": metrics,
}

with open(
    METADATA_PATH,
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

print("\nModel metadata saved:")
print(METADATA_PATH)


# In[49]:


#log experiment
run_row = pd.DataFrame(
    [
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "experiment_name": EXPERIMENT_NAME,
            "model_type": MODEL_TYPE,
            "dataset_version": DATASET_VERSION,
            "created_at": created_at,
        }
    ]
)

run_job = bq.load_table_from_dataframe(
    run_row,
    EXPERIMENT_RUNS_TABLE,
)

run_job.result()

print("\nExperiment run logged:")
print(EXPERIMENT_RUNS_TABLE)


# In[50]:


#create metrics
metrics = {
    "pr_auc": float(
        average_precision_score(y, cv_probabilities)
    ),
    "roc_auc": float(
        roc_auc_score(y, cv_probabilities)
    ),
    "precision": float(
        precision_score(
            y,
            cv_predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y,
            cv_predictions,
            zero_division=0,
        )
    ),
}

print(metrics)


# In[51]:


#log experiment metrics
metric_rows = []

for metric_name, metric_value in metrics.items():
    metric_rows.append(
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "created_at": created_at,
        }
    )

metrics_df = pd.DataFrame(metric_rows)

metric_job = bq.load_table_from_dataframe(
    metrics_df,
    EXPERIMENT_METRICS_TABLE,
)

metric_job.result()

print("\nExperiment metrics logged:")
print(EXPERIMENT_METRICS_TABLE)


# In[52]:


#generate summary
print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 60)

print("Experiment ID:", experiment_id)
print("Run ID:", run_id)
print("Experiment:", EXPERIMENT_NAME)
print("Dataset:", DATASET_VERSION)

print("\nTraining population")
print("Providers:", len(training_df))
print("Fraud:", fraud_count)
print("Normal:", normal_count)

print("\nMetrics")
for metric_name, metric_value in metrics.items():
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )

print("\nArtifact:")
print(MODEL_PATH)


# In[56]:


#final version
#change train_random_forest.py into a callable function
#use feature_table instead of defined features
def train_random_forest(
    project_id,
    feature_table,
    feedback_table,
    feature_columns,
    model_output_uri,
):
    print("=" * 60)
    print("Random Forest Training")
    print("=" * 60)

    print(f"Project: {project_id}")
    print(f"Feature table: {feature_table}")
    print(f"Feedback table: {feedback_table}")
    print(f"Features: {feature_columns}")

    bq = bigquery.Client(project=project_id)

    feature_select = ",\n      ".join(
        [f"p.{feature}" for feature in feature_columns]
    )

    query = f"""
    WITH latest_feedback AS (
      SELECT *
      FROM `{feedback_table}`
      WHERE confirmed_fraud IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY provider_id
        ORDER BY investigation_date DESC, created_at DESC
      ) = 1
    )

    SELECT
      f.provider_id,
      {feature_select},
      CAST(f.confirmed_fraud AS INT64) AS fraud_label
    FROM latest_feedback f
    JOIN `{feature_table}` p
      ON f.provider_id = p.provider_id
    WHERE p.dataset_split = 'TEST'
    """

    df = bq.query(query).to_dataframe()

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df["fraud_label"].nunique() < 2:
        raise ValueError(
            "Training data must contain both fraud classes."
        )

    X = df[feature_columns]
    y = df["fraud_label"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    min_class_count = int(
        y.value_counts().min()
    )

    n_splits = min(
        5,
        min_class_count,
    )

    if n_splits < 2:
        raise ValueError(
            "Not enough samples for stratified CV."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    cv_probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    cv_predictions = (
        cv_probabilities >= 0.5
    ).astype(int)

    metrics = {
        "pr_auc": float(
            average_precision_score(
                y,
                cv_probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                cv_probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y,
                cv_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                cv_predictions,
                zero_division=0,
            )
        ),
    }

    print("\nCross-validation metrics:")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    model.fit(X, y)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\nFeature Importance:")
    print(
        feature_importance.to_string(
            index=False
        )
    )

    local_model_path = "/tmp/model.joblib"

    joblib.dump(
        model,
        local_model_path,
    )

    if not model_output_uri.startswith("gs://"):
        raise ValueError(
            "model_output_uri must start with gs://"
        )

    uri_without_prefix = (
        model_output_uri
        .replace("gs://", "", 1)
    )

    bucket_name, blob_path = (
        uri_without_prefix.split("/", 1)
    )

    storage_client = storage.Client(
        project=project_id
    )

    bucket = storage_client.bucket(
        bucket_name
    )

    blob = bucket.blob(
        blob_path
    )

    blob.upload_from_filename(
        local_model_path
    )

    print(
        f"\nModel uploaded to: "
        f"{model_output_uri}"
    )

    return {
        "metrics": metrics,
        "model_output_uri": model_output_uri,
        "training_rows": len(df),
        "features": feature_columns,
    }


if __name__ == "__main__":
    config = load_config()

    model_output_uri = (
        "gs://"
        f"{config['bucket']}/"
        "models/random_forest/model.joblib"
    )

    train_random_forest(
        project_id=config["project_id"],
        feature_table=config["data"]["feature_table"],
        feedback_table=config["data"]["feedback_table"],
        feature_columns=config["features"],
        model_output_uri=model_output_uri,
    )


# In[ ]:




