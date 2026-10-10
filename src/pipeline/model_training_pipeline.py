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
from src.pipeline.components.register_model_component import (
    register_model_component,
)
from src.pipeline.components.train_random_forest_component import (
    train_random_forest_component,
)
from src.pipeline.components.train_xgboost_component import (
    train_xgboost_component,
)


# ============================================================
# REUSABLE FRAUD MODEL CONTINUOUS TRAINING PIPELINE
#
# Use Case 1:
#   New monthly data triggers retraining.
#
# Use Case 2:
#   A new feature configuration triggers retraining.
#
# Use Case 3:
#   A new model type or model configuration triggers retraining.
#
# Shared workflow:
#
#   Retraining Decision
#          |
#          v
#   Rolling Window
#          |
#          v
#   Provider Feature Refresh
#          |
#          +-----------------------+
#          |                       |
#          v                       v
#   Random Forest              XGBoost
#          |                       |
#          v                       v
#   Register Candidate       Register Candidate
#          |                       |
#          v                       v
#   Pending Approval        Pending Approval
#
# Production remains unchanged until manual approval.
# ============================================================


@dsl.pipeline(
    name="fraud-model-training",
    description=(
        "Reusable provider-level fraud detection pipeline "
        "supporting Random Forest and XGBoost with shared "
        "feature engineering, retraining logic, model "
        "registration, and manual approval."
    ),
)
def model_training_pipeline(
    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------
    project_id: str,
    region: str,

    # --------------------------------------------------------
    # Model configuration
    #
    # training_image is intentionally NOT passed here.
    # KFP requires container images to be known at compile time.
    # --------------------------------------------------------
    model_type: str,
    model_name: str,
    model_parameters_json: str,
    current_model_signature: str,

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
    # Candidate model artifact
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

    # --------------------------------------------------------
    # Stable ML product identity
    #
    # The state key represents the fraud ML product,
    # not the individual model algorithm.
    #
    # Example:
    #
    # approved_model_type = random_forest
    # candidate_model_type = xgboost
    #
    # Both still belong to:
    # provider_fraud_model
    # --------------------------------------------------------
    state_key: str = "provider_fraud_model",
):

    # ========================================================
    # 1. CHECK WHETHER RETRAINING IS REQUIRED
    #
    # Trigger examples:
    #
    # Use Case 1:
    #   New monthly data
    #
    # Use Case 2:
    #   Feature configuration changed
    #
    # Use Case 3:
    #   Model configuration changed
    #
    # Monitoring:
    #   Data quality alert
    #   Feature drift
    #   Prediction drift
    # ========================================================

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
            current_model_type=(
                model_type
            ),
            data_quality_table=(
                data_quality_table
            ),
            feature_drift_table=(
                feature_drift_table
            ),
            prediction_drift_table=(
                prediction_drift_table
            ),
            training_state_table=(
                training_state_table
            ),
            state_key=(
                state_key
            ),
        )
    )

    retraining_task.set_display_name(
        "Check Retraining Decision"
    )

    # ========================================================
    # 2. RUN CANDIDATE PIPELINE ONLY WHEN RETRAINING IS NEEDED
    # ========================================================

    with dsl.If(
        retraining_task.output == True,
        name="retraining-required",
    ):

        # ====================================================
        # 2A. CALCULATE ROLLING TRAIN / TEST WINDOW
        #
        # Example:
        #
        # current_data_month = 2026-10-01
        #
        # TRAIN:
        #   Feb 2026 - Jul 2026
        #
        # TEST:
        #   Aug 2026 - Oct 2026
        # ====================================================

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

        # ====================================================
        # 2B. REFRESH PROVIDER-LEVEL FEATURES
        #
        # Both Random Forest and XGBoost consume the same
        # provider feature table.
        #
        # This separates feature engineering from model choice.
        # ====================================================

        refresh_features_task = (
            refresh_provider_features_component(
                project_id=(
                    project_id
                ),
                claims_table=(
                    claims_table
                ),
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

        refresh_features_task.after(
            rolling_window_task
        )

        # ====================================================
        # 3. RANDOM FOREST BRANCH
        # ====================================================

        with dsl.If(
            model_type == "random_forest",
            name="random-forest-model",
        ):

            # ------------------------------------------------
            # 3A. TRAIN RANDOM FOREST CANDIDATE
            #
            # The Random Forest image is defined inside
            # train_random_forest_component.py at compile time.
            # ------------------------------------------------

            rf_train_task = (
                train_random_forest_component(
                    project_id=(
                        project_id
                    ),
                    feature_table=(
                        feature_table
                    ),
                    feature_columns_json=(
                        feature_columns_json
                    ),
                    model_parameters_json=(
                        model_parameters_json
                    ),
                    model_output_uri=(
                        model_output_uri
                    ),
                )
            )

            rf_train_task.set_display_name(
                "Train Random Forest Candidate"
            )

            rf_train_task.after(
                refresh_features_task
            )

            # ------------------------------------------------
            # 3B. REGISTER RANDOM FOREST CANDIDATE
            #
            # Registration does NOT mean production approval.
            # ------------------------------------------------

            rf_register_task = (
                register_model_component(
                    project_id=(
                        project_id
                    ),
                    region=(
                        region
                    ),
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
                rf_train_task
            )

            # ------------------------------------------------
            # 3C. MARK RANDOM FOREST CANDIDATE PENDING
            #
            # Candidate fields are updated.
            #
            # Approved production state is NOT changed.
            # ------------------------------------------------

            rf_pending_task = (
                mark_candidate_pending_component(
                    project_id=(
                        project_id
                    ),
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
                    state_key=(
                        state_key
                    ),
                )
            )

            rf_pending_task.set_display_name(
                "Mark Random Forest Candidate Pending"
            )

            rf_pending_task.after(
                rf_register_task
            )

        # ====================================================
        # 4. XGBOOST BRANCH
        #
        # Same:
        #   data
        #   features
        #   rolling window
        #   registration
        #   approval workflow
        #
        # Different:
        #   training algorithm
        # ====================================================

        with dsl.If(
            model_type == "xgboost",
            name="xgboost-model",
        ):

            # ------------------------------------------------
            # 4A. TRAIN XGBOOST CANDIDATE
            #
            # The XGBoost image is defined inside
            # train_xgboost_component.py at compile time.
            # ------------------------------------------------

            xgb_train_task = (
                train_xgboost_component(
                    project_id=(
                        project_id
                    ),
                    feature_table=(
                        feature_table
                    ),
                    feature_columns_json=(
                        feature_columns_json
                    ),
                    model_parameters_json=(
                        model_parameters_json
                    ),
                    model_output_uri=(
                        model_output_uri
                    ),
                )
            )

            xgb_train_task.set_display_name(
                "Train XGBoost Candidate"
            )

            xgb_train_task.after(
                refresh_features_task
            )

            # ------------------------------------------------
            # 4B. REGISTER XGBOOST CANDIDATE
            #
            # The current registration component is generic
            # despite its random_forest filename.
            # ------------------------------------------------

            xgb_register_task = (
                register_model_component(
                    project_id=(
                        project_id
                    ),
                    region=(
                        region
                    ),
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
                xgb_train_task
            )

            # ------------------------------------------------
            # 4C. MARK XGBOOST CANDIDATE PENDING
            #
            # This uses the same fraud workflow state record.
            # ------------------------------------------------

            xgb_pending_task = (
                mark_candidate_pending_component(
                    project_id=(
                        project_id
                    ),
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
                    state_key=(
                        state_key
                    ),
                )
            )

            xgb_pending_task.set_display_name(
                "Mark XGBoost Candidate Pending"
            )

            xgb_pending_task.after(
                xgb_register_task
            )


# ============================================================
# 5. COMPILE PIPELINE
#
# Keep the current filename temporarily so the existing
# launcher continues to find it.
# ============================================================

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
        "Pipeline compiled successfully:"
    )

    print(
        output_file
    )
