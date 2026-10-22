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
from src.pipeline.components.model_promotion_gate_component import (
    model_promotion_gate_component,
)
from src.pipeline.components.refresh_provider_features_component import (
    refresh_provider_features_component,
)
from src.pipeline.components.register_model_component import (
    register_model_component,
)
from src.pipeline.components.train_random_forest_component import (
    train_random_forest_component,
)
from src.pipeline.components.train_xgboost_component import (
    train_xgboost_component,
)


@dsl.pipeline(
    name="fraud-model-training",
    description=(
        "Reusable provider-level fraud training pipeline "
        "with monitoring-driven retraining, experiment tracking, "
        "model registration, and manual promotion."
    ),
)
def model_training_pipeline(
    # Project
    project_id: str,
    region: str,

    # Model configuration
    model_type: str,
    model_name: str,
    model_parameters_json: str,
    current_model_signature: str,

    # Data and rolling window
    claims_table: str,
    current_data_month: str,
    current_feature_signature: str,
    training_months: int,
    testing_months: int,

    # Feature engineering
    feature_table: str,
    feature_sql_json: str,
    derived_feature_sql_json: str,
    feature_columns_json: str,

    # Candidate artifact
    model_output_uri: str,
    model_artifact_uri: str,

    # Vertex Model Registry
    model_display_name: str,
    serving_container_image_uri: str,

    # Experiment tracking
    experiment_runs_table: str,
    experiment_metrics_table: str,
    experiment_id: str,
    run_id: str,
    experiment_name: str,
    experiment_owner: str,
    feature_version: str,
    dataset_version: str,
    git_commit: str,
    training_start: str,
    training_end: str,

    # Monitoring
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    performance_degradation_table: str,
    training_state_table: str,

    # Model promotion gate
    min_pr_auc_improvement: float,
    min_recall_change: float,
    min_f1_change: float,
    min_precision_change: float,

    # Stable identity for the fraud-model workflow
    state_key: str = "provider_fraud_model",
):
    # Decide whether a new candidate is needed.
    retraining_task = (
        check_retraining_decision_component(
            project_id=project_id,
            current_data_month=current_data_month,
            current_feature_signature=(
                current_feature_signature
            ),
            current_model_signature=(
                current_model_signature
            ),
            current_model_type=model_type,
            data_quality_table=(
                data_quality_table
            ),
            feature_drift_table=(
                feature_drift_table
            ),
            prediction_drift_table=(
                prediction_drift_table
            ),
            performance_degradation_table=(
                performance_degradation_table
            ),
            training_state_table=(
                training_state_table
            ),
            state_key=state_key,
        )
    )

    retraining_task.set_display_name(
        "Check Retraining Decision"
    )

    # Everything below runs only when one of the retraining
    # triggers fires.
    with dsl.If(
        retraining_task.output == True,
        name="retraining-required",
    ):
        rolling_window_task = (
            calculate_rolling_window_component(
                current_data_month=(
                    current_data_month
                ),
                training_months=(
                    training_months
                ),
                testing_months=(
                    testing_months
                ),
            )
        )

        rolling_window_task.set_display_name(
            "Calculate Rolling Window"
        )

        # Rebuild the provider-level feature table for the
        # current train/test window.
        refresh_features_task = (
            refresh_provider_features_component(
                project_id=project_id,
                claims_table=claims_table,
                output_feature_table=(
                    feature_table
                ),
                feature_sql_json=(
                    feature_sql_json
                ),
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

        # This component writes to BigQuery, so we do not want
        # Vertex to reuse an old cached execution.
        refresh_features_task.set_caching_options(
            False
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        with dsl.If(
            model_type == "random_forest",
            name="random-forest-model",
        ):
            rf_train_task = (
                train_random_forest_component(
                    project_id=project_id,
                    feature_table=feature_table,
                    feature_columns_json=(
                        feature_columns_json
                    ),
                    model_parameters_json=(
                        model_parameters_json
                    ),
                    model_output_uri=(
                        model_output_uri
                    ),

                    experiment_runs_table=(
                        experiment_runs_table
                    ),
                    experiment_metrics_table=(
                        experiment_metrics_table
                    ),
                    experiment_id=experiment_id,
                    run_id=run_id,
                    experiment_name=(
                        experiment_name
                    ),
                    experiment_owner=(
                        experiment_owner
                    ),
                    model_type=model_type,
                    feature_version=(
                        feature_version
                    ),
                    dataset_version=(
                        dataset_version
                    ),
                    git_commit=git_commit,

                    training_start=(
                        training_start
                    ),
                    training_end=training_end,
                )
            )

            rf_train_task.set_display_name(
                "Train Random Forest Candidate"
            )

            rf_train_task.after(
                refresh_features_task
            )

            rf_gate_task = (
                model_promotion_gate_component(
                    project_id=project_id,
                    experiment_metrics_table=(
                        experiment_metrics_table
                    ),
                    training_state_table=(
                        training_state_table
                    ),
                    challenger_run_id=run_id,
                    min_pr_auc_improvement=(
                        min_pr_auc_improvement
                    ),
                    min_recall_change=(
                        min_recall_change
                    ),
                    min_f1_change=(
                        min_f1_change
                    ),
                    min_precision_change=(
                        min_precision_change
                    ),
                    state_key=state_key,
                )
            )

            rf_gate_task.set_display_name(
                "Model Promotion Gate"
            )

            rf_gate_task.after(
                rf_train_task
            )

            with dsl.If(
                rf_gate_task.outputs[
                    "eligible_for_promotion"
                ] == True,
                name="random-forest-promotion-eligible",
            ):
                rf_register_task = (
                    register_model_component(
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

                rf_register_task.set_display_name(
                    "Register Random Forest Candidate"
                )

                rf_register_task.after(
                    rf_gate_task
                )

                rf_pending_task = (
                    mark_candidate_pending_component(
                        project_id=project_id,
                        training_state_table=(
                            training_state_table
                        ),
                        current_data_month=(
                            current_data_month
                        ),
                        candidate_model_resource=(
                            rf_register_task.output
                        ),
                        candidate_feature_signature=(
                            current_feature_signature
                        ),
                        candidate_model_signature=(
                            current_model_signature
                        ),
                        candidate_model_type=(
                            model_type
                        ),
                        candidate_run_id=run_id,
                        state_key=state_key,
                    )
                )

                rf_pending_task.set_display_name(
                    "Mark Random Forest Candidate Pending"
                )

                rf_pending_task.after(
                    rf_register_task
                )

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        with dsl.If(
            model_type == "xgboost",
            name="xgboost-model",
        ):
            xgb_train_task = (
                train_xgboost_component(
                    project_id=project_id,
                    feature_table=feature_table,
                    feature_columns_json=(
                        feature_columns_json
                    ),
                    model_parameters_json=(
                        model_parameters_json
                    ),
                    model_output_uri=(
                        model_output_uri
                    ),

                    experiment_runs_table=(
                        experiment_runs_table
                    ),
                    experiment_metrics_table=(
                        experiment_metrics_table
                    ),
                    experiment_id=experiment_id,
                    run_id=run_id,
                    experiment_name=(
                        experiment_name
                    ),
                    experiment_owner=(
                        experiment_owner
                    ),
                    model_type=model_type,
                    feature_version=(
                        feature_version
                    ),
                    dataset_version=(
                        dataset_version
                    ),
                    git_commit=git_commit,

                    training_start=(
                        training_start
                    ),
                    training_end=training_end,
                )
            )

            xgb_train_task.set_display_name(
                "Train XGBoost Candidate"
            )

            xgb_train_task.after(
                refresh_features_task
            )

            xgb_gate_task = (
                model_promotion_gate_component(
                    project_id=project_id,
                    experiment_metrics_table=(
                        experiment_metrics_table
                    ),
                    training_state_table=(
                        training_state_table
                    ),
                    challenger_run_id=run_id,
                    min_pr_auc_improvement=(
                        min_pr_auc_improvement
                    ),
                    min_recall_change=(
                        min_recall_change
                    ),
                    min_f1_change=(
                        min_f1_change
                    ),
                    min_precision_change=(
                        min_precision_change
                    ),
                    state_key=state_key,
                )
            )

            xgb_gate_task.set_display_name(
                "Model Promotion Gate"
            )

            xgb_gate_task.after(
                xgb_train_task
            )

            with dsl.If(
                xgb_gate_task.outputs[
                    "eligible_for_promotion"
                ] == True,
                name="xgboost-promotion-eligible",
            ):
                xgb_register_task = (
                    register_model_component(
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

                xgb_register_task.set_display_name(
                    "Register XGBoost Candidate"
                )

                xgb_register_task.after(
                    xgb_gate_task
                )

                xgb_pending_task = (
                    mark_candidate_pending_component(
                        project_id=project_id,
                        training_state_table=(
                            training_state_table
                        ),
                        current_data_month=(
                            current_data_month
                        ),
                        candidate_model_resource=(
                            xgb_register_task.output
                        ),
                        candidate_feature_signature=(
                            current_feature_signature
                        ),
                        candidate_model_signature=(
                            current_model_signature
                        ),
                        candidate_model_type=(
                            model_type
                        ),
                        candidate_run_id=run_id,
                        state_key=state_key,
                    )
                )

                xgb_pending_task.set_display_name(
                    "Mark XGBoost Candidate Pending"
                )

                xgb_pending_task.after(
                    xgb_register_task
                )


if __name__ == "__main__":
    output_file = (
        "model_training_pipeline.yaml"
    )

    compiler.Compiler().compile(
        pipeline_func=(
            model_training_pipeline
        ),
        package_path=(
            output_file
        ),
    )

    print(
        f"Pipeline compiled successfully: "
        f"{output_file}"
    )
