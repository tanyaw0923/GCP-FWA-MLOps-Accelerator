from kfp import compiler, dsl

from src.pipeline.components.postprocess_random_forest_predictions_component import (
    postprocess_random_forest_predictions_component,
)

from src.pipeline.components.run_random_forest_monitoring_component import (
    run_random_forest_monitoring_component,
)

from src.pipeline.components.check_retraining_decision_component import (
    check_retraining_decision_component,
)


@dsl.pipeline(
    name="random-forest-downstream-qa-pipeline",
)
def random_forest_downstream_qa_pipeline(
    project_id: str,

    # Existing successful Vertex raw prediction table
    raw_prediction_table: str,

    # Feature / experiment metadata
    feature_table: str,
    feature_columns_json: str,
    experiment_runs_table: str,

    # Curated prediction output
    provider_prediction_table: str,

    # Monitoring
    batch_input_table: str,
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    feature_shift_threshold: float,
    prediction_shift_threshold: float,
    baseline_positive_rate: float,

    # Continuous Training
    current_data_month: str,
    training_state_table: str,
):
    # ---------------------------------------------------------
    # 1. Post-process existing raw Vertex predictions
    # ---------------------------------------------------------
    postprocess_task = (
        postprocess_random_forest_predictions_component(
            project_id=project_id,
            raw_prediction_table=raw_prediction_table,
            feature_table=feature_table,
            experiment_runs_table=experiment_runs_table,
            output_table=provider_prediction_table,
            feature_columns_json=feature_columns_json,
        )
    )

    # ---------------------------------------------------------
    # 2. Run monitoring
    # ---------------------------------------------------------
    monitoring_task = (
        run_random_forest_monitoring_component(
            project_id=project_id,
            feature_table=feature_table,
            batch_input_table=batch_input_table,
            provider_prediction_table=provider_prediction_table,
            feature_columns_json=feature_columns_json,
            data_quality_table=data_quality_table,
            feature_drift_table=feature_drift_table,
            prediction_drift_table=prediction_drift_table,
            feature_shift_threshold=feature_shift_threshold,
            prediction_shift_threshold=prediction_shift_threshold,
            baseline_positive_rate=baseline_positive_rate,
        )
    )

    monitoring_task.after(
        postprocess_task
    )

    # ---------------------------------------------------------
    # 3. Check retraining decision
    #
    # Trigger if:
    # - Data Quality ALERT
    # - Feature Drift ALERT
    # - Prediction Drift ALERT
    # - New monthly data available
    # ---------------------------------------------------------
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

    retraining_decision_task.after(
        monitoring_task
    )


# ---------------------------------------------------------
# Compile pipeline
# ---------------------------------------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=random_forest_downstream_qa_pipeline,
        package_path="random_forest_downstream_qa_pipeline.yaml",
    )

    print(
        "QA pipeline compiled successfully: "
        "random_forest_downstream_qa_pipeline.yaml"
    )
