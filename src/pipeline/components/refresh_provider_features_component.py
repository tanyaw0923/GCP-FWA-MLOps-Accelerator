from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def refresh_provider_features_component(
    project_id: str,
    claims_table: str,
    output_feature_table: str,
    feature_sql_json: str,
    derived_feature_sql_json: str,
    train_start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
):
    import json
    import re
    from google.cloud import bigquery

    # ---------------------------------------------------------
    # 1. Parse configurable feature definitions
    # ---------------------------------------------------------
    feature_sql = json.loads(
        feature_sql_json
    )

    derived_feature_sql = json.loads(
        derived_feature_sql_json
    )

    if not feature_sql:
        raise ValueError(
            "feature_sql_json is empty."
        )

    # ---------------------------------------------------------
    # 2. Validate feature names
    # ---------------------------------------------------------
    feature_name_pattern = (
        r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    for feature_name in list(
        feature_sql.keys()
    ) + list(
        derived_feature_sql.keys()
    ):

        if not re.fullmatch(
            feature_name_pattern,
            feature_name,
        ):
            raise ValueError(
                f"Invalid feature name: "
                f"{feature_name}"
            )

    # ---------------------------------------------------------
    # 3. Validate table identifiers
    # ---------------------------------------------------------
    table_pattern = (
        r"^[A-Za-z0-9_-]+\."
        r"[A-Za-z0-9_]+\."
        r"[A-Za-z0-9_]+$"
    )

    for table_name in [
        claims_table,
        output_feature_table,
    ]:
        if not re.fullmatch(
            table_pattern,
            table_name,
        ):
            raise ValueError(
                f"Invalid BigQuery table: "
                f"{table_name}"
            )

    client = bigquery.Client(
        project=project_id
    )

    # ---------------------------------------------------------
    # 4. Build dynamic aggregate feature SQL
    #
    # Example generated:
    #
    # AVG(CAST(is_99215 AS FLOAT64))
    #   AS pct_99215
    # ---------------------------------------------------------
    aggregate_feature_expressions = []

    for (
        feature_name,
        sql_expression,
    ) in feature_sql.items():

        aggregate_feature_expressions.append(
            f"""
            {sql_expression}
              AS {feature_name}
            """
        )

    aggregate_feature_sql = ",\n".join(
        aggregate_feature_expressions
    )

    # ---------------------------------------------------------
    # 5. Features needed for split-level baselines
    #
    # high_acuity_oe_ratio currently needs the split mean of
    # high_acuity_pct.
    #
    # Baseline calculations remain metadata, not model features.
    # ---------------------------------------------------------
    baseline_sql = """
        AVG(high_acuity_pct)
          AS expected_high_acuity_pct
    """

    # ---------------------------------------------------------
    # 6. Build dynamic final derived features
    # ---------------------------------------------------------
    derived_expressions = []

    for (
        feature_name,
        sql_expression,
    ) in derived_feature_sql.items():

        derived_expressions.append(
            f"""
            {sql_expression}
              AS {feature_name}
            """
        )

    if derived_expressions:
        derived_select_sql = (
            ",\n"
            + ",\n".join(
                derived_expressions
            )
        )
    else:
        derived_select_sql = ""

    # ---------------------------------------------------------
    # 7. Dynamically select base features
    # ---------------------------------------------------------
    base_feature_select = []

    for feature_name in feature_sql.keys():
        base_feature_select.append(
            f"p.{feature_name}"
        )

    base_feature_select_sql = ",\n".join(
        base_feature_select
    )

    # ---------------------------------------------------------
    # 8. Build rolling-window provider feature table
    # ---------------------------------------------------------
    query = f"""
    CREATE OR REPLACE TABLE
    `{output_feature_table}` AS

    WITH claim_windows AS (

      SELECT
        *,

        CASE
          WHEN claim_date BETWEEN
               DATE(@train_start_date)
               AND DATE(@train_end_date)
            THEN 'TRAIN'

          WHEN claim_date BETWEEN
               DATE(@test_start_date)
               AND DATE(@test_end_date)
            THEN 'TEST'

          ELSE NULL
        END AS rolling_dataset_split

      FROM `{claims_table}`

      WHERE claim_date BETWEEN
        DATE(@train_start_date)
        AND DATE(@test_end_date)
    ),

    provider_base AS (

      SELECT
        provider_id,

        ANY_VALUE(
          provider_name
        ) AS provider_name,

        rolling_dataset_split
          AS dataset_split,

        COUNT(*)
          AS claim_count,

        {aggregate_feature_sql}

      FROM claim_windows

      WHERE
        rolling_dataset_split
        IS NOT NULL

      GROUP BY
        provider_id,
        rolling_dataset_split
    ),

    split_baseline AS (

      SELECT
        dataset_split,

        {baseline_sql}

      FROM provider_base

      GROUP BY
        dataset_split
    )

    SELECT
      p.provider_id,
      p.provider_name,
      p.dataset_split,
      p.claim_count,

      {base_feature_select_sql}

      {derived_select_sql}

    FROM provider_base p

    LEFT JOIN split_baseline b
      USING(dataset_split)

    ORDER BY
      dataset_split,
      provider_id
    """

    # ---------------------------------------------------------
    # 9. Execute feature refresh
    # ---------------------------------------------------------
    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "train_start_date",
                    "STRING",
                    train_start_date,
                ),
                bigquery.ScalarQueryParameter(
                    "train_end_date",
                    "STRING",
                    train_end_date,
                ),
                bigquery.ScalarQueryParameter(
                    "test_start_date",
                    "STRING",
                    test_start_date,
                ),
                bigquery.ScalarQueryParameter(
                    "test_end_date",
                    "STRING",
                    test_end_date,
                ),
            ]
        )
    )

    print("=" * 65)
    print("REFRESHING PROVIDER FEATURES")
    print("=" * 65)

    print(
        f"TRAIN: "
        f"{train_start_date} "
        f"to {train_end_date}"
    )

    print(
        f"TEST: "
        f"{test_start_date} "
        f"to {test_end_date}"
    )

    print(
        "\nConfigured base features:"
    )

    for feature_name in feature_sql:
        print(
            f"  - {feature_name}"
        )

    print(
        "\nConfigured derived features:"
    )

    for feature_name in derived_feature_sql:
        print(
            f"  - {feature_name}"
        )

    client.query(
        query,
        job_config=job_config,
    ).result()

    # ---------------------------------------------------------
    # 10. Verify output
    # ---------------------------------------------------------
    verify_query = f"""
    SELECT
      dataset_split,
      COUNT(*) AS provider_rows,
      SUM(claim_count) AS claims_used
    FROM `{output_feature_table}`
    GROUP BY dataset_split
    ORDER BY dataset_split
    """

    rows = list(
        client.query(
            verify_query
        ).result()
    )

    if not rows:
        raise RuntimeError(
            "Feature refresh produced no rows."
        )

    print(
        "\nFeature refresh complete:"
    )

    for row in rows:

        print(
            f"{row.dataset_split}: "
            f"providers={row.provider_rows}, "
            f"claims={row.claims_used}"
        )
