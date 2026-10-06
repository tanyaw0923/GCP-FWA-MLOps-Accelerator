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
    model_name: str = "random_forest",
):
    from datetime import date

    from google.cloud import bigquery

    client = bigquery.Client(
        project=project_id
    )

    current_data_month_date = (
        date.fromisoformat(
            current_data_month
        )
    )

    # ---------------------------------------------------------
    # Store candidate information without updating
    # last_training_data_month.
    #
    # The month is considered accepted only after the model
    # receives manual production approval.
    # ---------------------------------------------------------
    query = f"""
    MERGE `{training_state_table}` AS target

    USING (
      SELECT
        @model_name AS model_name
    ) AS source

    ON
      target.model_name
      = source.model_name

    WHEN MATCHED THEN
      UPDATE SET
        candidate_status =
          'PENDING_APPROVAL',

        candidate_model_resource =
          @candidate_model_resource,

        candidate_data_month =
          @current_data_month,

        candidate_created_at =
          CURRENT_TIMESTAMP()

    WHEN NOT MATCHED THEN
      INSERT (
        model_name,
        candidate_status,
        candidate_model_resource,
        candidate_data_month,
        candidate_created_at
      )

      VALUES (
        @model_name,
        'PENDING_APPROVAL',
        @candidate_model_resource,
        @current_data_month,
        CURRENT_TIMESTAMP()
      )
    """

    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "model_name",
                    "STRING",
                    model_name,
                ),
                bigquery.ScalarQueryParameter(
                    "candidate_model_resource",
                    "STRING",
                    candidate_model_resource,
                ),
                bigquery.ScalarQueryParameter(
                    "current_data_month",
                    "DATE",
                    current_data_month_date,
                ),
            ]
        )
    )

    client.query(
        query,
        job_config=job_config,
    ).result()

    print("=" * 60)
    print("Candidate Model Ready")
    print("=" * 60)

    print(
        f"Model: "
        f"{model_name}"
    )

    print(
        f"Candidate data month: "
        f"{current_data_month}"
    )

    print(
        f"Candidate resource: "
        f"{candidate_model_resource}"
    )

    print(
        "\nStatus: "
        "PENDING_APPROVAL"
    )

    print(
        "Production model remains unchanged."
    )
