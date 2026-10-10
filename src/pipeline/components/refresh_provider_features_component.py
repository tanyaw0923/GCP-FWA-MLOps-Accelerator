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


    # ============================================================
    # 1. PARSE FEATURE DEFINITIONS
    #
    # Base features and derived features are passed into the
    # component as JSON so the same pipeline can support different
    # feature sets without rewriting this component.
    #
    # Example:
    #
    # {
    #   "high_acuity_pct":
    #       "AVG(CAST(is_high_acuity_em AS FLOAT64))",
    #
    #   "avg_member_provider_distance":
    #       "AVG(member_provider_distance)"
    # }
    #
    # Use Case 2 simply adds a new feature definition to the config.
    # ============================================================

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


    # ============================================================
    # 2. VALIDATE FEATURE NAMES
    #
    # Feature names are used when dynamically building SQL, so only
    # standard SQL-safe column names are allowed.
    # ============================================================

    feature_name_pattern = (
        r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    all_feature_names = (
        list(feature_sql.keys())
        + list(derived_feature_sql.keys())
    )

    for feature_name in all_feature_names:
        if not re.fullmatch(
            feature_name_pattern,
            feature_name,
        ):
            raise ValueError(
                f"Invalid feature name: "
                f"{feature_name}"
            )


    # ============================================================
    # 3. VALIDATE BIGQUERY TABLE NAMES
    # ============================================================

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


    # ============================================================
    # 4. BUILD BASE FEATURE SQL
    #
    # Each configured claim-level expression is aggregated to the
    # provider level.
    #
    # Examples:
    #
    # AVG(CAST(is_99215 AS FLOAT64))
    #     AS pct_99215
    #
    # AVG(member_provider_distance)
    #     AS avg_member_provider_distance
    #
    # The second example is the feature introduced in Use Case 2.
    # ============================================================

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


    # ============================================================
    # 5. SPLIT-LEVEL BASELINE
    #
    # high_acuity_oe_ratio depends on a peer baseline calculated
    # separately for TRAIN and TEST.
    #
    # These baseline values are supporting metadata rather than
    # direct Random Forest inputs.
    # ============================================================

    baseline_sql = """
        AVG(high_acuity_pct)
          AS expected_high_acuity_pct
    """


    # ============================================================
    # 6. BUILD DERIVED FEATURE SQL
    #
    # Derived features are calculated after provider aggregation.
    #
    # Example:
    #
    # SAFE_DIVIDE(
    #     p.high_acuity_pct,
    #     b.expected_high_acuity_pct
    # )
    #
    #     AS high_acuity_oe_ratio
    # ============================================================

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


    # ============================================================
    # 7. BUILD FINAL BASE FEATURE SELECT
    # ============================================================

    base_feature_select = []

    for feature_name in feature_sql.keys():
        base_feature_select.append(
            f"p.{feature_name}"
        )

    base_feature_select_sql = ",\n".join(
        base_feature_select
    )


    # ============================================================
    # 8. CREATE ROLLING PROVIDER FEATURE TABLE
    #
    # Use Case 1:
    #
    # TRAIN and TEST are assigned dynamically from service_date.
    #
    # Example for:
    #
    # current_data_month = 2026-10-01
    #
    # TRAIN:
    # 2026-02-01 -> 2026-07-31
    #
    # TEST:
    # 2026-08-01 -> 2026-10-31
    #
    # There are no hard-coded split dates in this component.
    #
    #
    # Use Case 2:
    #
    # member_provider_distance already exists in raw claims.
    # Adding:
    #
    # AVG(member_provider_distance)
    #
    # to feature_sql automatically creates the new provider-level
    # feature without changing this component.
    #
    #
    # fraud_provider_label is synthetic ground truth used only for
    # this demo. It allows Random Forest to train on TRAIN and
    # evaluate on TEST.
    # ============================================================

    query = f"""
    CREATE OR REPLACE TABLE
    `{output_feature_table}` AS

    WITH claim_windows AS (

      SELECT
        *,

        CASE
          WHEN service_date BETWEEN
               DATE(@train_start_date)
               AND DATE(@train_end_date)
            THEN 'TRAIN'

          WHEN service_date BETWEEN
               DATE(@test_start_date)
               AND DATE(@test_end_date)
            THEN 'TEST'

          ELSE NULL
        END AS rolling_dataset_split

      FROM `{claims_table}`

      WHERE service_date BETWEEN
        DATE(@train_start_date)
        AND DATE(@test_end_date)
    ),

    provider_base AS (

      SELECT
        provider_id,

        ANY_VALUE(
          provider_name
        ) AS provider_name,

        ANY_VALUE(
          provider_specialty
        ) AS provider_specialty,

        ANY_VALUE(
          provider_city
        ) AS provider_city,

        ANY_VALUE(
          provider_state
        ) AS provider_state,

        rolling_dataset_split
          AS dataset_split,

        MAX(
          fraud_provider_label
        ) AS fraud_label,

        COUNT(*)
          AS claim_count,

        COUNT(
          DISTINCT member_id
        ) AS unique_members,

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
      p.provider_specialty,
      p.provider_city,
      p.provider_state,

      p.dataset_split,
      p.fraud_label,

      p.claim_count,
      p.unique_members,

      {base_feature_select_sql}

      {derived_select_sql}

    FROM provider_base p

    LEFT JOIN split_baseline b
      USING(dataset_split)

    ORDER BY
      dataset_split,
      provider_id
    """


    # ============================================================
    # 9. EXECUTE FEATURE REFRESH
    # ============================================================

    job_config = bigquery.QueryJobConfig(
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

    print("=" * 65)
    print("REFRESH PROVIDER FEATURES")
    print("=" * 65)

    print(
        f"TRAIN: "
        f"{train_start_date} "
        f"to {train_end_date}"
    )

    print(
        f"TEST:  "
        f"{test_start_date} "
        f"to {test_end_date}"
    )

    print(
        "\nBase features:"
    )

    for feature_name in feature_sql:
        print(
            f"  - {feature_name}"
        )

    if derived_feature_sql:
        print(
            "\nDerived features:"
        )

        for feature_name in derived_feature_sql:
            print(
                f"  - {feature_name}"
            )

    client.query(
        query,
        job_config=job_config,
    ).result()


    # ============================================================
    # 10. VERIFY OUTPUT
    #
    # Confirm both rolling splits were created and show:
    #
    # - number of providers
    # - number of claims
    # - fraud / normal provider counts
    #
    # This output is useful during the Vertex Pipeline demo.
    # ============================================================

    verify_query = f"""
    SELECT
      dataset_split,

      COUNT(*) AS provider_rows,

      SUM(claim_count)
        AS claims_used,

      COUNTIF(
        fraud_label = 1
      ) AS fraud_providers,

      COUNTIF(
        fraud_label = 0
      ) AS normal_providers

    FROM `{output_feature_table}`

    GROUP BY
      dataset_split

    ORDER BY
      dataset_split
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

    split_names = {
        row.dataset_split
        for row in rows
    }

    required_splits = {
        "TRAIN",
        "TEST",
    }

    missing_splits = (
        required_splits
        - split_names
    )

    if missing_splits:
        raise RuntimeError(
            "Feature refresh is missing "
            f"dataset split(s): "
            f"{sorted(missing_splits)}"
        )


    # ============================================================
    # 11. PRINT FEATURE REFRESH SUMMARY
    # ============================================================

    print(
        "\nFeature refresh complete:"
    )

    for row in rows:
        print(
            f"\n{row.dataset_split}"
        )

        print(
            f"  Providers: "
            f"{row.provider_rows}"
        )

        print(
            f"  Claims used: "
            f"{row.claims_used}"
        )

        print(
            f"  Fraud providers: "
            f"{row.fraud_providers}"
        )

        print(
            f"  Normal providers: "
            f"{row.normal_providers}"
        )


    # ============================================================
    # 12. FEATURE VALIDATION
    #
    # Check whether any generated feature contains NULL values.
    #
    # This prevents Random Forest training from starting with an
    # invalid provider-level dataset.
    # ============================================================

    all_model_features = (
        list(feature_sql.keys())
        + list(
            derived_feature_sql.keys()
        )
    )

    null_checks = []

    for feature_name in all_model_features:
        null_checks.append(
            f"""
            COUNTIF(
              {feature_name} IS NULL
            ) AS {feature_name}_nulls
            """
        )

    null_check_sql = ",\n".join(
        null_checks
    )

    validation_query = f"""
    SELECT
      {null_check_sql}
    FROM `{output_feature_table}`
    """

    validation_rows = list(
        client.query(
            validation_query
        ).result()
    )

    if not validation_rows:
        raise RuntimeError(
            "Feature validation query "
            "returned no result."
        )

    validation_row = (
        validation_rows[0]
    )

    null_feature_counts = {}

    for feature_name in all_model_features:
        null_count = getattr(
            validation_row,
            f"{feature_name}_nulls",
        )

        null_feature_counts[
            feature_name
        ] = null_count

    features_with_nulls = {
        feature_name: null_count
        for (
            feature_name,
            null_count,
        ) in null_feature_counts.items()
        if null_count > 0
    }

    if features_with_nulls:
        raise RuntimeError(
            "NULL values found in "
            f"provider features: "
            f"{features_with_nulls}"
        )

    print(
        "\nFeature validation: PASS"
    )

    print(
        "No NULL values found in "
        "configured model features."
    )

