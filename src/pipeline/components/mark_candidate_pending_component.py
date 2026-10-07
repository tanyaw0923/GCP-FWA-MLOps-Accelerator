from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def mark_candidate_pending_component(
    project_id: str,
    training_state_table: str,
    current_data_month: str,
    candidate_model_resource: str,
    candidate_feature_signature: str,
    model_name: str = "random_forest",
):
    from google.cloud import bigquery

    bq = bigquery.Client(project=project_id)

    query = f"""
    MERGE `{training_state_table}` AS target
    USING (
      SELECT
        @model_name AS model_name,
        DATE(@current_data_month) AS candidate_data_month,
        @candidate_model_resource AS candidate_model_resource,
        @candidate_feature_signature AS candidate_feature_signature
    ) AS source
    ON target.model_name = source.model_name

    WHEN MATCHED THEN
      UPDATE SET
        candidate_status = 'PENDING_APPROVAL',
        candidate_model_resource = source.candidate_model_resource,
        candidate_data_month = source.candidate_data_month,
        candidate_feature_signature = source.candidate_feature_signature,
        candidate_created_at = CURRENT_TIMESTAMP()

    WHEN NOT MATCHED THEN
      INSERT (
        model_name,
        candidate_status,
        candidate_model_resource,
        candidate_data_month,
        candidate_feature_signature,
        candidate_created_at
      )
      VALUES (
        source.model_name,
        'PENDING_APPROVAL',
        source.candidate_model_resource,
        source.candidate_data_month,
        source.candidate_feature_signature,
        CURRENT_TIMESTAMP()
      )
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "model_name",
                "STRING",
                model_name,
            ),
            bigquery.ScalarQueryParameter(
                "current_data_month",
                "STRING",
                current_data_month,
            ),
            bigquery.ScalarQueryParameter(
                "candidate_model_resource",
                "STRING",
                candidate_model_resource,
            ),
            bigquery.ScalarQueryParameter(
                "candidate_feature_signature",
                "STRING",
                candidate_feature_signature,
            ),
        ]
    )

    bq.query(
        query,
        job_config=job_config,
    ).result()

    print("=" * 70)
    print("Candidate Model Pending Approval")
    print("=" * 70)

    print(
        f"Model: "
        f"{model_name}"
    )

    print(
        f"Candidate data month: "
        f"{current_data_month}"
    )

    print(
        f"Candidate model resource: "
        f"{candidate_model_resource}"
    )

    print(
        f"Candidate feature signature: "
        f"{candidate_feature_signature}"
    )

    print(
        "Candidate status: "
        "PENDING_APPROVAL"
    )

    print(
        "\nApproved production state "
        "has not been changed."
    )
