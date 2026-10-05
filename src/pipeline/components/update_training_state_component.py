from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def update_training_state_component(
    project_id: str,
    training_state_table: str,
    current_data_month: str,
    model_name: str = "random_forest",
):
    from datetime import date
    from google.cloud import bigquery

    # ---------------------------------------------------------
    # 1. Validate current_data_month
    # ---------------------------------------------------------
    current_data_month_date = date.fromisoformat(
        current_data_month
    )

    # ---------------------------------------------------------
    # 2. BigQuery client
    # ---------------------------------------------------------
    client = bigquery.Client(
        project=project_id
    )

    # ---------------------------------------------------------
    # 3. Update training state
    #
    # One row per model.
    # If the model already exists -> update it.
    # Otherwise -> insert a new row.
    # ---------------------------------------------------------
    query = f"""
    MERGE `{training_state_table}` AS target

    USING (
        SELECT
            @model_name AS model_name,
            @current_data_month AS last_training_data_month,
            CURRENT_TIMESTAMP() AS last_training_timestamp
    ) AS source

    ON target.model_name = source.model_name

    WHEN MATCHED THEN
      UPDATE SET
        last_training_data_month =
            source.last_training_data_month,
        last_training_timestamp =
            source.last_training_timestamp

    WHEN NOT MATCHED THEN
      INSERT (
        model_name,
        last_training_data_month,
        last_training_timestamp
      )
      VALUES (
        source.model_name,
        source.last_training_data_month,
        source.last_training_timestamp
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
                "DATE",
                current_data_month_date,
            ),
        ]
    )

    client.query(
        query,
        job_config=job_config,
    ).result()

    # ---------------------------------------------------------
    # 4. Verification
    # ---------------------------------------------------------
    verify_query = f"""
    SELECT
      model_name,
      last_training_data_month,
      last_training_timestamp
    FROM `{training_state_table}`
    WHERE model_name = @model_name
    """

    verify_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "model_name",
                "STRING",
                model_name,
            )
        ]
    )

    rows = list(
        client.query(
            verify_query,
            job_config=verify_config,
        ).result()
    )

    if not rows:
        raise RuntimeError(
            "Training state update completed, "
            "but no state row was found."
        )

    row = rows[0]

    print("=" * 60)
    print("Training State Updated")
    print("=" * 60)

    print(
        f"Model: "
        f"{row.model_name}"
    )

    print(
        f"Last training data month: "
        f"{row.last_training_data_month}"
    )

    print(
        f"Last training timestamp: "
        f"{row.last_training_timestamp}"
    )
