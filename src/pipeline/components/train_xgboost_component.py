from kfp import dsl


# ============================================================
# XGBOOST TRAINING COMPONENT
#
# This component launches the XGBoost training container.
#
# It receives:
#   - model training inputs
#   - experiment tracking metadata
#   - rolling training window
#
# The interface matches the Random Forest component so the
# surrounding MLOps workflow remains reusable.
#
# Note:
# The container image must be known at pipeline compile time.
# KFP does not allow the image itself to be supplied as a
# runtime pipeline parameter.
# ============================================================


@dsl.container_component
def train_xgboost_component(
    # --------------------------------------------------------
    # Core training inputs
    # --------------------------------------------------------
    project_id: str,
    feature_table: str,
    feature_columns_json: str,
    model_parameters_json: str,
    model_output_uri: str,

    # --------------------------------------------------------
    # Experiment tracking
    # --------------------------------------------------------
    experiment_runs_table: str,
    experiment_metrics_table: str,
    experiment_id: str,
    run_id: str,
    experiment_name: str,
    experiment_owner: str,
    model_type: str,
    feature_version: str,
    dataset_version: str,
    git_commit: str,

    # --------------------------------------------------------
    # Training window
    # --------------------------------------------------------
    training_start: str,
    training_end: str,
):

    return dsl.ContainerSpec(
        image=(
            "us-central1-docker.pkg.dev/"
            "fwa-mlops-accelerator-demo/"
            "fwa-mlops/"
            "xgboost-training:v1.0.0"
        ),

        args=[
            # ------------------------------------------------
            # Core training inputs
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Experiment tracking
            # ------------------------------------------------

            "--experiment-runs-table",
            experiment_runs_table,

            "--experiment-metrics-table",
            experiment_metrics_table,

            "--experiment-id",
            experiment_id,

            "--run-id",
            run_id,

            "--experiment-name",
            experiment_name,

            "--experiment-owner",
            experiment_owner,

            "--model-type",
            model_type,

            "--feature-version",
            feature_version,

            "--dataset-version",
            dataset_version,

            "--git-commit",
            git_commit,

            # ------------------------------------------------
            # Training window
            # ------------------------------------------------

            "--training-start",
            training_start,

            "--training-end",
            training_end,
        ],
    )
