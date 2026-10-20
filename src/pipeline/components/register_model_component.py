from kfp import dsl
from kfp.dsl import Input, Model


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
    candidate_model: Input[Model],
    artifact_uri: str,
    serving_container_image_uri: str,
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
        f"KFP Model artifact URI: "
        f"{candidate_model.uri}"
    )

    print(
        f"KFP Model metadata: "
        f"{candidate_model.metadata}"
    )

    print(
        f"Serving container: "
        f"{serving_container_image_uri}"
    )

    registered_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        sync=True,
    )

    print("\nModel registered successfully.")

    print(
        f"Model resource name: "
        f"{registered_model.resource_name}"
    )

    return registered_model.resource_name
