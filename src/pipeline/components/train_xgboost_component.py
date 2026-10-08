from kfp import dsl


# ============================================================
# XGBOOST TRAINING COMPONENT
#
# The container image must be known at pipeline compile time.
# KFP does not allow the image field itself to be supplied as
# a runtime pipeline parameter.
# ============================================================

@dsl.container_component
def train_xgboost_component(
    project_id: str,
    feature_table: str,
    feature_columns_json: str,
    model_parameters_json: str,
    model_output_uri: str,
):
    return dsl.ContainerSpec(
        image=(
            "us-central1-docker.pkg.dev/"
            "fwa-mlops-accelerator-demo/"
            "fwa-mlops/"
            "xgboost-training:v1"
        ),
        args=[
            "--project-id",
            project_id,

            "--feature-table",
            feature_table,

            "--feature-columns",
            feature_columns_json,

            "--model-parameters",
            model_parameters_json,

            "--model-output-uri",
            model_output_uri,
        ],
    )
