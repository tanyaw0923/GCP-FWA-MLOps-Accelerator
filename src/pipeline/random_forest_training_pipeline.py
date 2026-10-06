from kfp import compiler, dsl

from src.pipeline.components.calculate_rolling_window_component import (
    calculate_rolling_window_component,
)

from src.pipeline.components.check_retraining_decision_component import (
    check_retraining_decision_component,
)

from src.pipeline.components.refresh_provider_features_component import (
    refresh_provider_features_component,
)

from src.pipeline.components.train_random_forest_component import (
    train_random_forest_component,
)

from src.pipeline.components.register_random_forest_model_component import (
    register_random_forest_model_component,
)

from src.pipeline.components.mark_candidate_pending_component import (
    mark_candidate_pending_component,
)


@dsl.pipeline(
    name="random-forest-fraud-training-pipeline",
)
def random_forest_training_pipeline(
    project_id: str,
    region: str,

    # ---------------------------------------------------------
    # Raw data / rolling window
    # ---------------------------------------------------------
    claims_table: str,
    current_data_month: str,
    training_months: int,
    testing_months: int,

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------
    feature_table: str,
    feature_sql_json: str,
    derived_feature_sql_json: str,
    feature_columns_json: str,

    # ---------------------------------------------------------
    # Candidate model artifact
    # ---------------------------------------------------------
    model_output_uri: str,
    model_artifact_uri: str,

    # ---------------------------------------------------------
    # Vertex Model Registry
    # ---------------------------------------------------------
    model_display_name: str,
    serving_container_image_uri: str,

    # ---------------------------------------------------------
    # Monitoring / continuous training
    # ---------------------------------------------------------
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    training_state_table: str,
):
    # =========================================================
    # 1. CHECK WHETHER RETRAINING IS REQUIRED
    #
    # Retraining can be triggered by:
    #
    # - new monthly data
    # - data quality alert
    # - feature drift
    # - prediction drift
    #
    # Use Case 1:
    #
    # Change current_data_month in pipeline_config.yaml:
    #
    #   2026-09-01
    #
    # to:
    #
    #   2026-10-01
    #
    # If the last approved training month is still September,
    # this component returns should_retrain = True.
    # =========================================================

    retraining_decision_task = (
        check_retraining_decision_component(
            project_id=project_id,
            current_data_month=current_data_month,
            data_quality_table=data_quality_table,
            feature_drift_table=feature_drift_table,
            prediction_drift_table=prediction_drift_table,
            training_state_table=training_state_table,
        )
    )


    # =========================================================
    # 2. RUN CANDIDATE TRAINING ONLY WHEN A TRIGGER EXISTS
    # =========================================================

    with dsl.If(
        retraining_decision_task.output == True,
        name="retraining-required",
    ):

        # -----------------------------------------------------
        # 2A. Calculate rolling TRAIN / TEST window
        #
        # Example:
        #
        # current_data_month = 2026-10-01
        # training_months = 6
        # testing_months = 3
        #
        # TRAIN:
        # 2026-02-01 -> 2026-07-31
        #
        # TEST:
        # 2026-08-01 -> 2026-10-31
        # -----------------------------------------------------

        window_task = (
            calculate_rolling_window_component(
                current_data_month=current_data_month,
                training_months=training_months,
                testing_months=testing_months,
            )
        )


        # -----------------------------------------------------
        # 2B. Refresh provider-level features
        #
        # Raw claims do not contain dataset_split.
        #
        # This component uses service_date and the rolling
        # window calculated above to create TRAIN and TEST.
        #
        # Use Case 2:
        #
        # New provider-level features can be introduced through
        # feature_sql_json without changing this pipeline.
        # -----------------------------------------------------

        refresh_task = (
            refresh_provider_features_component(
                project_id=project_id,
                claims_table=claims_table,
                output_feature_table=feature_table,
                feature_sql_json=feature_sql_json,
                derived_feature_sql_json=derived_feature_sql_json,

                train_start_date=(
                    window_task.outputs[
                        "train_start_date"
                    ]
                ),

                train_end_date=(
                    window_task.outputs[
                        "train_end_date"
                    ]
                ),

                test_start_date=(
                    window_task.outputs[
                        "test_start_date"
                    ]
                ),

                test_end_date=(
                    window_task.outputs[
                        "test_end_date"
                    ]
                ),
            )
        )


        # -----------------------------------------------------
        # 2C. Train Random Forest candidate
        #
        # train_random_forest.py now uses:
        #
        # TRAIN -> fit Random Forest
        # TEST  -> evaluate candidate
        #
        # The configured feature list determines which provider
        # features are passed to the model.
        # -----------------------------------------------------

        train_task = (
            train_random_forest_component(
                project_id=project_id,
                feature_table=feature_table,
                feature_columns_json=feature_columns_json,
                model_output_uri=model_output_uri,
            )
        )

        train_task.after(
            refresh_task
        )


        # -----------------------------------------------------
        # 2D. Register candidate in Vertex Model Registry
        #
        # Registering a model does NOT mean it has been promoted
        # to production.
        #
        # Production continues using the currently approved
        # model while this candidate waits for review.
        # -----------------------------------------------------

        register_task = (
            register_random_forest_model_component(
                project_id=project_id,
                region=region,
                model_display_name=model_display_name,
                artifact_uri=model_artifact_uri,
                serving_container_image_uri=(
                    serving_container_image_uri
                ),
            )
        )

        register_task.after(
            train_task
        )


        # -----------------------------------------------------
        # 2E. Mark candidate as PENDING_APPROVAL
        #
        # Important:
        #
        # last_training_data_month is NOT updated here.
        #
        # It should only move forward after the candidate has
        # been manually approved for production.
        # -----------------------------------------------------

        candidate_task = (
            mark_candidate_pending_component(
                project_id=project_id,
                training_state_table=training_state_table,
                current_data_month=current_data_month,
                candidate_model_resource=(
                    register_task.output
                ),
            )
        )

        candidate_task.after(
            register_task
        )


# =============================================================
# COMPILE PIPELINE
# =============================================================

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=random_forest_training_pipeline,
        package_path=(
            "random_forest_training_pipeline.yaml"
        ),
    )

    print(
        "Pipeline compiled successfully: "
        "random_forest_training_pipeline.yaml"
    )
