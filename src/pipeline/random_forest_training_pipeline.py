from kfp import compiler, dsl

from src.pipeline.components.train_random_forest_component import (
    train_random_forest_component,
)

from src.pipeline.components.register_random_forest_model_component import (
    register_random_forest_model_component,
)

from src.pipeline.components.run_random_forest_batch_prediction_component import (
    run_random_forest_batch_prediction_component,
)

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
    name="random-forest-fraud-training-pipeline",
)
def random_forest_training_pipeline(
    project_id: str,
    region: str,

    # Training data
    feature_table: str,
    feedback_table: str,
    feature_columns_json: str,

    # Initial training/model artifact
    model_output_uri: str,
    model_artifact_uri: str,

    # Model Registry
    model_display_name: str,
    serving_container_image_uri: str,

    # Batch prediction
    batch_input_table: str,
    batch_output_dataset: str,

    # Post-processing
    experiment_runs_table: str,
    provider_prediction_table: str,

    # Monitoring
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    feature_shift_threshold: float,
    prediction_shift_threshold: float,
    baseline_positive_rate: float,

    # Continuous training
    current_data_month: str,
    training_state_table: str,
):
    # ---------------------------------------------------------
    # 1. Train Random Forest
    # ---------------------------------------------------------
    train_task = train_random_forest_component(
        project_id=project_id,
        feature_table=feature_table,
        feedback_table=feedback_table,
        feature_columns_json=feature_columns_json,
        model_output_uri=model_output_uri,
    )

    # ---------------------------------------------------------
    # 2. Register model
    # ---------------------------------------------------------
    register_task = register_random_forest_model_component(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
    )

    register_task.after(
        train_task
    )

    # ---------------------------------------------------------
    # 3. Batch prediction
    # ---------------------------------------------------------
    batch_task = run_random_forest_batch_prediction_component(
        project_id=project_id,
        region=region,
        model_resource_name=register_task.output,
        batch_input_table=batch_input_table,
        batch_output_dataset=batch_output_dataset,
        feature_columns_json=feature_columns_json,
    )

    batch_task.after(
        register_task
    )

    # ---------------------------------------------------------
    # 4. Post-process raw predictions
    # ---------------------------------------------------------
    postprocess_task = (
        postprocess_random_forest_predictions_component(
            project_id=project_id,
            raw_prediction_table=batch_task.output,
            feature_table=feature_table,
            experiment_runs_table=experiment_runs_table,
            output_table=provider_prediction_table,
            feature_columns_json=feature_columns_json,
        )
    )

    postprocess_task.after(
        batch_task
    )

    # ---------------------------------------------------------
    # 5. Monitoring
    # ---------------------------------------------------------
    monitoring_task = run_random_forest_monitoring_component(
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

    monitoring_task.after(
        postprocess_task
    )

    # ---------------------------------------------------------
    # 6. Continuous-training decision
    #
    # Retrain if:
    # - data quality alert
    # - feature drift alert
    # - prediction drift alert
    # - new monthly data available
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
        pipeline_func=random_forest_training_pipeline,
        package_path="random_forest_training_pipeline.yaml",
    )

    print(
        "Pipeline compiled successfully: "
        "random_forest_training_pipeline.yaml"
    )
