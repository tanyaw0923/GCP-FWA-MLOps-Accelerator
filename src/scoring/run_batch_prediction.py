from google.cloud import aiplatform


# ============================================================
# Configuration
# ============================================================

PROJECT_ID = "fwa-mlops-accelerator-demo"
REGION = "us-central1"

MODEL_ID = "4093535116523995136"

INPUT_URI = (
    "bq://fwa-mlops-accelerator-demo."
    "fraud_features.xgboost_batch_input"
)

OUTPUT_URI = (
    "bq://fwa-mlops-accelerator-demo."
    "fraud_experiments"
)


# ============================================================
# Initialize Vertex AI
# ============================================================

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)


# ============================================================
# Load registered model
# ============================================================

model = aiplatform.Model(
    model_name=MODEL_ID
)

print("Model:")
print(model.resource_name)


# ============================================================
# Run batch prediction
# ============================================================

batch_job = model.batch_predict(
    job_display_name="provider-eye-fraud-xgboost-batch-v2",
    bigquery_source=INPUT_URI,
    bigquery_destination_prefix=OUTPUT_URI,
    instances_format="bigquery",
    predictions_format="bigquery",
    machine_type="n1-standard-2",
    sync=True,
)


# ============================================================
# Results
# ============================================================

print("\nBatch prediction complete.")
print(f"Job resource: {batch_job.resource_name}")
print(f"Job state: {batch_job.state}")
