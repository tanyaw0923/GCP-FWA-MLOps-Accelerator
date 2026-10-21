from google.cloud import aiplatform

PROJECT_ID = "fwa-mlops-accelerator-demo"
REGION = "us-east1"

ARTIFACT_URI = (
    "gs://fwa-mlops-accelerator-demo-folder/"
    "models/xgboost/candidate/"
)

SERVING_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai/"
    "prediction/xgboost-cpu.2-1:latest"
)

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)

model = aiplatform.Model.upload(
    display_name="fwa-xgboost-candidate-us-east1",
    artifact_uri=ARTIFACT_URI,
    serving_container_image_uri=SERVING_CONTAINER,
    sync=True,
)

print("New model:")
print(model.resource_name)
