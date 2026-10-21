from google.cloud import aiplatform


PROJECT_ID = "fwa-mlops-accelerator-demo"
REGION = "us-central1"

CONTAINER_URI = (
    "us-central1-docker.pkg.dev/"
    "fwa-mlops-accelerator-demo/"
    "fwa-mlops/"
    "xgboost-serving:v2"
)


aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)


model = aiplatform.Model.upload(
    display_name="fwa-xgboost-custom-serving",

    serving_container_image_uri=CONTAINER_URI,

    serving_container_predict_route="/predict",

    serving_container_health_route="/health",

    serving_container_ports=[
        8080
    ],

    sync=True,
)


print("\nCustom XGBoost model registered:")
print(model.resource_name)
