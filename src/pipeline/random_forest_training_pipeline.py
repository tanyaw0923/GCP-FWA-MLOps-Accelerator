import json

from kfp import compiler, dsl

from src.pipeline.components.calculate_rolling_window_component import (
    calculate_rolling_window_component,
)
from src.pipeline.components.check_retraining_decision_component import (
    check_retraining_decision_component,
)
from src.pipeline.components.mark_candidate_pending_component import (
    mark_candidate_pending_component,
)
from src.pipeline.components.refresh_provider_features_component import (
    refresh_provider_features_component,
)
from src.pipeline.components.register_random_forest_model_component import (
    register_random_forest_model_component,
)
from src.pipeline.components.train_random_forest_component import (
    train_random_forest_component,
)


# ============================================================
# RANDOM FOREST CONTINUOUS TRAINING PIPELINE
#
# Use Case 1:
#   New monthly data triggers retraining.
#
# Use Case 2:
#   A new feature configuration triggers retraining.
#
# Candidate models are registered and marked
# PENDING_APPROVAL. Production is not updated here.
# ============================================================

@dsl.pipeline(
    name="random-forest-fraud-training",
    description=(
        "Continuous training pipeline for provider-level "
        "fraud detection using rolling data windows and "
        "config-driven feature engineering."
    ),
)
def random_forest_training_pipeline(
    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------
    project_id: str,
    region: str,

    # --------------------------------------------------------
    # Data / continuous training
    # --------------------------------------------------------
    claims_table: str,
    current_data_month: str,
    current_feature_signature: str,
    training_months: int,
    testing_months: int,

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------
    feature_table: str,
    feature_sql_json: str,
    derived_feature_sql_json: str,
    feature_columns_json: str,

    # --------------------------------------------------------
    # Model artifacts
    # --------------------------------------------------------
    model_output_uri: str,
    model_artifact_uri: str,

    # --------------------------------------------------------
    # Vertex Model Registry
    # --------------------------------------------------------
    model_display_name: str,
    serving_container_image_uri: str,

    # --------------------------------------------------------
    # Monitoring / continuous training state
    # --------------------------------------------------------
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    training_state_table: str,
):
    # ========================================================
    # 1. CHECK WHETHER RETRAINING IS REQUIRED
    #
    # Triggers:
    #   - new month of data
    #   - feature configuration changed
    #   - data quality alert
    #   - feature drift
    #   - prediction drift
    # ========================================================

    retraining_decision_task = (
        check_retraining_decision_component(
            project_id=project_id,
            current_data_month=current_data_month,
            current_feature_signature=(
                current_feature_signature
            ),
            data_quality_table=data_quality_table,
            feature_drift_table=feature_drift_table,
            prediction_drift_table=(
                prediction_drift_table
            ),
            training_state_table=(
                training_state_table
            ),
        )
    )

    retraining_decision_task.set_display_name(
        "Check Retraining Decision"
    )

    # ========================================================
    # 2. RUN CANDIDATE TRAINING ONLY WHEN REQUIRED
    # ========================================================

    with dsl.If(
        retraining_decision_task.output == True,
        name="retraining-required",
    ):
        # ====================================================
        # 2A. CALCULATE ROLLING TRAIN / TEST WINDOW
        #
        # Example:
        #
        # current_data_month = 2026-10-01
        #
        # TRAIN = Feb-Jul
        # TEST  = Aug-Oct
        # ====================================================

        rolling_window_task = (
            calculate_rolling_window_component(
                current_data_month=(
                    current_data_month
                ),
                training_months=training_months,
                testing_months=testing_months,
            )
        )

        rolling_window_task.set_display_name(
            "Calculate Rolling Window"
        )

        # ====================================================
        # 2B. REFRESH PROVIDER-LEVEL FEATURES
        #
        # Raw claim data is aggregated into provider-level
        # features and assigned TRAIN / TEST dataset splits.
        #
        # Use Case 2 is config-driven:
        # enabling a new feature in pipeline_config.yaml
        # automatically changes the feature SQL passed here.
        # ====================================================

        refresh_features_task = (
            refresh_provider_features_component(
                project_id=project_id,
                claims_table=claims_table,
                output_feature_table=feature_table,
                feature_sql_json=feature_sql_json,
                derived_feature_sql_json=(
                    derived_feature_sql_json
                ),
                train_start_date=(
                    rolling_window_task.outputs[
                        "train_start_date"
                    ]
                ),
                train_end_date=(
                    rolling_window_task.outputs[
                        "train_end_date"
                    ]
                ),
                test_start_date=(
                    rolling_window_task.outputs[
                        "test_start_date"
                    ]
                ),
                test_end_date=(
                    rolling_window_task.outputs[
                        "test_end_date"
                    ]
                ),
            )
        )

        refresh_features_task.set_display_name(
            "Refresh Provider Features"
        )

        # Explicit dependency for readability in Vertex UI.
        refresh_features_task.after(
            rolling_window_task
        )

        # ====================================================
        # 2C. TRAIN RANDOM FOREST CANDIDATE
        #
        # train_random_forest.py:
        #   TRAIN rows -> model fitting
        #   TEST rows  -> candidate evaluation
        #
        # The model artifact is written only to the candidate
        # location.
        # ====================================================

        train_task = (
            train_random_forest_component(
                project_id=project_id,
                feature_table=feature_table,
                feature_columns_json=(
                    feature_columns_json
                ),
                model_output_uri=model_output_uri,
            )
        )

        train_task.set_display_name(
            "Train Random Forest Candidate"
        )

        train_task.after(
            refresh_features_task
        )

        # ====================================================
        # 2D. REGISTER CANDIDATE IN VERTEX MODEL REGISTRY
        #
        # Registration does NOT mean production approval.
        # ====================================================

        register_task = (
            register_random_forest_model_component(
                project_id=project_id,
                region=region,
                model_display_name=(
                    model_display_name
                ),
                artifact_uri=(
		    model_artifact_uri
                ),
                serving_container_image_uri=(
                    serving_container_image_uri
                ),
            )
        )

        register_task.set_display_name(
            "Register Random Forest Candidate"
        )

        register_task.after(
            train_task
        )

        # ====================================================
        # 2E. MARK CANDIDATE AS PENDING APPROVAL
        #
        # Important:
        #   last_training_data_month is NOT changed here.
        #   approved_feature_signature is NOT changed here.
        #
        # Those production-state fields are updated only
        # through the manual promotion workflow.
        # ====================================================

        candidate_pending_task = (
            mark_candidate_pending_component(
                project_id=project_id,
                training_state_table=(
                    training_state_table
                ),
                current_data_month=(
                    current_data_month
                ),
                candidate_model_resource=(
                    register_task.output
                ),
                candidate_feature_signature=(
                    current_feature_signature
                ),
            )
        )

        candidate_pending_task.set_display_name(
            "Mark Candidate Pending Approval"
        )

        candidate_pending_task.after(
            register_task
        )


# ============================================================
# COMPILE PIPELINE
# ============================================================

if __name__ == "__main__":
    output_file = (
        "random_forest_training_pipeline.yaml"
    )

    compiler.Compiler().compile(
        pipeline_func=(
            random_forest_training_pipeline
        ),
        package_path=output_file,
    )

    print(
        "Pipeline compiled successfully:"
    )

    print(
        output_file
    )
