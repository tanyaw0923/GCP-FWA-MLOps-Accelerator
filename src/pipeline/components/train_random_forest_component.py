from kfp import dsl


@dsl.container_component
def train_random_forest_component(
    project_id: str,
    feature_table: str,
    feedback_table: str,
    feature_columns_json: str,
    model_output_uri: str,
):
    return dsl.ContainerSpec(
        image=(
            "us-central1-docker.pkg.dev/"
            "fwa-mlops-accelerator-demo/"
            "fwa-mlops/"
            "random-forest-training:v1"
        ),
        args=[
            "--project-id",
            project_id,
            "--feature-table",
            feature_table,
            "--feedback-table",
            feedback_table,
            "--feature-columns",
            feature_columns_json,
            "--model-output-uri",
            model_output_uri,
        ],
    )
