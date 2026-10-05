from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
)
def register_random_forest_model_component(
    project_id: str,
    region: str,
    model_display_name: str,
    artifact_uri: str,
    serving_container_image_uri: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=region,
    )

    print("=" * 60)
    print("Registering Random Forest Model")
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

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        sync=True,
    )

    print("\nModel registered successfully.")

    print(
        f"Model resource name: "
        f"{model.resource_name}"
    )

    return model.resource_name
