from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def postprocess_random_forest_predictions_component(
    project_id: str,
    raw_prediction_table: str,
    feature_table: str,
    experiment_runs_table: str,
    output_table: str,
    feature_columns_json: str,
):
    import json
    import re

    from google.cloud import bigquery


    # ---------------------------------------------------------
    # 1. Parse feature configuration
    # ---------------------------------------------------------
    feature_columns = json.loads(
        feature_columns_json
    )

    if not feature_columns:
        raise ValueError(
            "feature_columns_json cannot be empty."
        )


    # ---------------------------------------------------------
    # 2. Validate feature names
    #
    # Feature names are inserted into SQL identifiers,
    # so only allow standard BigQuery-safe column names.
    # ---------------------------------------------------------
    valid_column_pattern = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    invalid_features = [
        feature
        for feature in feature_columns
        if not valid_column_pattern.match(feature)
    ]

    if invalid_features:
        raise ValueError(
            "Invalid feature column names: "
            f"{invalid_features}"
        )


    # ---------------------------------------------------------
    # 3. Normalize Vertex BigQuery output table URI
    #
    # Vertex can return:
    # bq://project.dataset.table
    #
    # BigQuery SQL expects:
    # project.dataset.table
    # ---------------------------------------------------------
    raw_prediction_table = (
        raw_prediction_table
        .replace("bq://", "")
    )


    # ---------------------------------------------------------
    # 4. Dynamically build TRAIN feature means
    #
    # Example:
    # AVG(high_acuity_pct) AS mean_0,
    # AVG(pct_99214) AS mean_1,
    # ...
    # ---------------------------------------------------------
    train_mean_expressions = []

    for index, feature in enumerate(
        feature_columns
    ):
        train_mean_expressions.append(
            f"AVG(`{feature}`) AS mean_{index}"
        )

    train_mean_select = ",\n        ".join(
        train_mean_expressions
    )


    # ---------------------------------------------------------
    # 5. Dynamically build reason-code structures
    #
    # For each feature:
    #
    # provider feature value
    # ----------------------
    # TRAIN feature mean
    #
    # Then rank the ratios and keep the top 3.
    # ---------------------------------------------------------
    reason_structs = []

    for index, feature in enumerate(
        feature_columns
    ):
        reason_structs.append(
            f"""
            STRUCT(
              '{feature}' AS feature_name,
              SAFE_DIVIDE(
                r.`{feature}`,
                m.mean_{index}
              ) AS ratio,
              {index + 1} AS feature_order
            )
            """.strip()
        )

    reason_struct_array = ",\n            ".join(
        reason_structs
    )


    # ---------------------------------------------------------
    # 6. Create BigQuery client
    # ---------------------------------------------------------
    bq = bigquery.Client(
        project=project_id
    )


    # ---------------------------------------------------------
    # 7. Build post-processing SQL
    # ---------------------------------------------------------
    query = f"""
    CREATE OR REPLACE TABLE `{output_table}` AS

    WITH latest_run AS (
      SELECT
        experiment_id,
        run_id,
        experiment_name
      FROM `{experiment_runs_table}`
      WHERE model_type = 'random_forest'
      ORDER BY created_at DESC
      LIMIT 1
    ),

    train_means AS (
      SELECT
        {train_mean_select}
      FROM `{feature_table}`
      WHERE dataset_split = 'TRAIN'
    ),

    scored AS (
      SELECT
        p.*,

        CASE
          WHEN CAST(p.prediction AS STRING) = '1'
            THEN 1.0
          ELSE 0.0
        END AS fraud_score

      FROM `{raw_prediction_table}` p
    ),

    ranked AS (
      SELECT
        *,

        ROW_NUMBER() OVER (
          ORDER BY
            fraud_score DESC,
            provider_id
        ) AS risk_rank,

        CASE
          WHEN fraud_score = 1
            THEN 'HIGH'
          ELSE 'LOW'
        END AS risk_category

      FROM scored
    ),

    reason_codes AS (
      SELECT
        r.*,

        ARRAY(
          SELECT AS STRUCT
            feature_name,
            ratio

          FROM UNNEST([
            {reason_struct_array}
          ])

          WHERE ratio IS NOT NULL

          ORDER BY
            ratio DESC,
            feature_order

          LIMIT 3
        ) AS reasons

      FROM ranked r
      CROSS JOIN train_means m
    )

    SELECT
      CURRENT_DATE() AS prediction_date,

      p.provider_id,

      lr.experiment_id,
      lr.run_id,

      'random_forest' AS model_name,
      lr.experiment_name,

      p.fraud_score,
      p.risk_rank,
      p.risk_category,

      CASE
        WHEN p.fraud_score = 1
          THEN 'Random Forest predicted provider as suspicious'
        ELSE 'Random Forest predicted provider as normal'
      END AS reason,

      CURRENT_TIMESTAMP() AS created_at,

      p.reasons[SAFE_OFFSET(0)].feature_name
        AS reason_1,

      p.reasons[SAFE_OFFSET(1)].feature_name
        AS reason_2,

      p.reasons[SAFE_OFFSET(2)].feature_name
        AS reason_3

    FROM reason_codes p
    CROSS JOIN latest_run lr
    """


    # ---------------------------------------------------------
    # 8. Print configuration
    # ---------------------------------------------------------
    print("=" * 60)
    print("Post-processing Random Forest Predictions")
    print("=" * 60)

    print(
        f"Project: "
        f"{project_id}"
    )

    print(
        f"Raw prediction table: "
        f"{raw_prediction_table}"
    )

    print(
        f"Feature table: "
        f"{feature_table}"
    )

    print(
        f"Experiment runs table: "
        f"{experiment_runs_table}"
    )

    print(
        f"Output table: "
        f"{output_table}"
    )

    print("\nFeatures used for reason codes:")

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )


    # ---------------------------------------------------------
    # 9. Execute post-processing
    # ---------------------------------------------------------
    job = bq.query(
        query
    )

    job.result()


    # ---------------------------------------------------------
    # 10. Success
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Post-processing Completed Successfully")
    print("=" * 60)

    print(
        f"Curated prediction table: "
        f"{output_table}"
    )

