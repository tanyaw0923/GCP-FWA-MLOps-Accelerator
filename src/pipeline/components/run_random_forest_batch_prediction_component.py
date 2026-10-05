from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
)
def run_random_forest_batch_prediction_component(
    project_id: str,
    region: str,
    model_resource_name: str,
    batch_input_table: str,
    batch_output_dataset: str,
    feature_columns_json: str,
) -> str:
    import json
    import time

    from google.cloud import aiplatform_v1


    # ---------------------------------------------------------
    # 1. Parse feature columns
    # ---------------------------------------------------------
    feature_columns = json.loads(
        feature_columns_json
    )

    if not feature_columns:
        raise ValueError(
            "feature_columns_json is empty."
        )


    # ---------------------------------------------------------
    # 2. Vertex AI client
    # ---------------------------------------------------------
    client = (
        aiplatform_v1.JobServiceClient(
            client_options={
                "api_endpoint":
                f"{region}-aiplatform.googleapis.com"
            }
        )
    )


    # ---------------------------------------------------------
    # 3. BigQuery source
    # ---------------------------------------------------------
    bigquery_source = (
        aiplatform_v1.types.BigQuerySource(
            input_uri=(
                f"bq://{batch_input_table}"
            )
        )
    )


    # ---------------------------------------------------------
    # 4. BigQuery destination
    #
    # batch_output_dataset should be:
    # project.dataset
    # ---------------------------------------------------------
    bigquery_destination = (
        aiplatform_v1.types.BigQueryDestination(
            output_uri=(
                f"bq://{batch_output_dataset}"
            )
        )
    )


    # ---------------------------------------------------------
    # 5. Instance configuration
    #
    # Keep provider_id in the BigQuery source table,
    # but send only configured model features to sklearn.
    # ---------------------------------------------------------
    instance_config = (
        aiplatform_v1.types.BatchPredictionJob.InstanceConfig(
            instance_type="array",
            included_fields=feature_columns,
        )
    )


    # ---------------------------------------------------------
    # 6. Compute resources
    # ---------------------------------------------------------
    dedicated_resources = (
        aiplatform_v1.types.BatchDedicatedResources(
            machine_spec=(
                aiplatform_v1.types.MachineSpec(
                    machine_type="n1-standard-2"
                )
            ),
            starting_replica_count=1,
            max_replica_count=1,
        )
    )


    # ---------------------------------------------------------
    # 7. Create batch prediction job
    # ---------------------------------------------------------
    batch_job = (
        aiplatform_v1.types.BatchPredictionJob(
            display_name=(
                "random-forest-provider-batch-prediction"
            ),
            model=model_resource_name,

            input_config=(
                aiplatform_v1.types.BatchPredictionJob.InputConfig(
                    instances_format="bigquery",
                    bigquery_source=bigquery_source,
                )
            ),

            output_config=(
                aiplatform_v1.types.BatchPredictionJob.OutputConfig(
                    predictions_format="bigquery",
                    bigquery_destination=bigquery_destination,
                )
            ),

            instance_config=instance_config,

            dedicated_resources=dedicated_resources,
        )
    )


    # ---------------------------------------------------------
    # 8. Submit batch prediction job
    # ---------------------------------------------------------
    parent = (
        f"projects/{project_id}/"
        f"locations/{region}"
    )

    response = client.create_batch_prediction_job(
        parent=parent,
        batch_prediction_job=batch_job,
    )

    job_name = response.name

    print("=" * 60)
    print("Random Forest Batch Prediction")
    print("=" * 60)

    print(
        f"Batch job resource: "
        f"{job_name}"
    )

    print(
        f"Model resource: "
        f"{model_resource_name}"
    )

    print(
        f"Input table: "
        f"{batch_input_table}"
    )

    print(
        f"Output dataset: "
        f"{batch_output_dataset}"
    )

    print(
        "Features sent to model:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )


    # ---------------------------------------------------------
    # 9. Wait for batch prediction to finish
    # ---------------------------------------------------------
    terminal_states = {
        aiplatform_v1.types.JobState.JOB_STATE_SUCCEEDED,
        aiplatform_v1.types.JobState.JOB_STATE_FAILED,
        aiplatform_v1.types.JobState.JOB_STATE_CANCELLED,
        aiplatform_v1.types.JobState.JOB_STATE_EXPIRED,
    }

    while True:
        job = client.get_batch_prediction_job(
            name=job_name
        )

        print(
            f"Batch prediction state: "
            f"{job.state.name}"
        )

        if job.state in terminal_states:
            break

        time.sleep(30)


    # ---------------------------------------------------------
    # 10. Validate job result
    # ---------------------------------------------------------
    if (
        job.state
        != aiplatform_v1.types.JobState.JOB_STATE_SUCCEEDED
    ):
        raise RuntimeError(
            "Batch prediction failed. "
            f"State: {job.state.name}. "
            f"Error: {job.error}"
        )


    # ---------------------------------------------------------
    # 11. Get raw BigQuery prediction table
    # ---------------------------------------------------------
    raw_prediction_table = (
        job.output_info.bigquery_output_table
    )

    if not raw_prediction_table:
        raise RuntimeError(
            "Batch prediction succeeded, but Vertex "
            "did not return a BigQuery output table."
        )


    print(
        "\nVertex returned raw output table:"
    )

    print(
        raw_prediction_table
    )


    # ---------------------------------------------------------
    # 12. Normalize output table name
    #
    # Possible Vertex responses:
    #
    #   predictions_xxx
    #
    #   fraud_experiments.predictions_xxx
    #
    #   fwa-mlops-accelerator-demo.
    #       fraud_experiments.predictions_xxx
    #
    # Downstream BigQuery SQL requires:
    #
    #   project.dataset.table
    # ---------------------------------------------------------
    raw_prediction_table = (
        raw_prediction_table
        .replace("bq://", "")
        .replace("bigquery://", "")
        .strip()
    )

    table_parts = (
        raw_prediction_table.split(".")
    )


    # Vertex returned only:
    # predictions_xxx
    if len(table_parts) == 1:

        normalized_prediction_table = (
            f"{batch_output_dataset}."
            f"{raw_prediction_table}"
        )


    # Vertex returned:
    # dataset.predictions_xxx
    elif len(table_parts) == 2:

        normalized_prediction_table = (
            f"{project_id}."
            f"{raw_prediction_table}"
        )


    # Vertex already returned:
    # project.dataset.table
    elif len(table_parts) == 3:

        normalized_prediction_table = (
            raw_prediction_table
        )


    else:
        raise ValueError(
            "Unexpected BigQuery prediction table "
            f"format returned by Vertex: "
            f"{raw_prediction_table}"
        )


    # ---------------------------------------------------------
    # 13. Final validation
    # ---------------------------------------------------------
    normalized_parts = (
        normalized_prediction_table.split(".")
    )

    if len(normalized_parts) != 3:
        raise ValueError(
            "Prediction table is not fully qualified: "
            f"{normalized_prediction_table}"
        )


    print(
        "\nFully qualified prediction table:"
    )

    print(
        normalized_prediction_table
    )


    # ---------------------------------------------------------
    # 14. Return fully qualified table to post-processing
    # ---------------------------------------------------------
    return normalized_prediction_table
