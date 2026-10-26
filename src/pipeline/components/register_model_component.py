from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
)
def register_model_component(
    project_id: str,
    region: str,
    model_display_name: str,
    artifact_uri: str,
    serving_container_image_uri: str,
    feature_columns_json: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=region,
    )

    print("=" * 60)
    print("Registering Candidate Model")
    print("=" * 60)

    print(
        f"Project: "
        f"{project_id}"
    )

    print(
        f"Region: "
        f"{region}"
    )

    print(
        f"Model display name: "
        f"{model_display_name}"
    )

    print(
        f"Artifact URI: "
        f"{artifact_uri}"
    )

    print(
        f"Serving container: "
        f"{serving_container_image_uri}"
    )

    print(
        f"Serving feature contract: "
        f"{feature_columns_json}"
    )

    upload_kwargs = {
        "display_name": model_display_name,
        "artifact_uri": artifact_uri,
        "serving_container_image_uri": (
            serving_container_image_uri
        ),
        "serving_container_environment_variables": {
            "FEATURE_COLUMNS_JSON": feature_columns_json,
        },
        "sync": True,
    }

    if (
        "xgboost-serving" in serving_container_image_uri
        or "random-forest-serving"
        in serving_container_image_uri
    ):
        upload_kwargs.update(
            {
                "serving_container_predict_route": (
                    "/predict"
                ),
                "serving_container_health_route": (
                    "/health"
                ),
                "serving_container_ports": [
                    8080
                ],
            }
        )

    model = aiplatform.Model.upload(
        **upload_kwargs
    )

    print("\nModel registered successfully.")

    print(
        f"Model resource name: "
        f"{model.resource_name}"
    )

    return model.resource_name
