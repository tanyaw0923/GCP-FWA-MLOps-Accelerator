from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def run_random_forest_monitoring_component(
    project_id: str,
    feature_table: str,
    batch_input_table: str,
    provider_prediction_table: str,
    feature_columns_json: str,
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    feature_shift_threshold: float,
    prediction_shift_threshold: float,
    baseline_positive_rate: float,
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
            f"Invalid feature names: {invalid_features}"
        )


    # ---------------------------------------------------------
    # 3. Create BigQuery client
    # ---------------------------------------------------------
    bq = bigquery.Client(
        project=project_id
    )


    print("=" * 60)
    print("Random Forest Monitoring")
    print("=" * 60)

    print(
        f"Feature table: "
        f"{feature_table}"
    )

    print(
        f"Batch input table: "
        f"{batch_input_table}"
    )

    print(
        f"Prediction table: "
        f"{provider_prediction_table}"
    )

    print(
        f"Feature shift threshold: "
        f"{feature_shift_threshold:.2%}"
    )

    print(
        f"Prediction shift threshold: "
        f"{prediction_shift_threshold:.2%}"
    )

    print("\nFeatures:")

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )


    # =========================================================
    # A. DATA QUALITY
    #
    # No row-count or unique-provider-count checks.
    #
    # We only check:
    #   1. feature NULLs
    #   2. duplicate provider IDs
    # =========================================================

    null_count_expressions = [
        f"COUNTIF(`{feature}` IS NULL)"
        for feature in feature_columns
    ]

    total_null_expression = " + ".join(
        null_count_expressions
    )


    data_quality_query = f"""
    CREATE OR REPLACE TABLE `{data_quality_table}` AS

    WITH null_summary AS (
      SELECT
        {total_null_expression}
          AS total_feature_null_count

      FROM `{batch_input_table}`
    ),

    duplicate_summary AS (
      SELECT
        COUNT(*) AS duplicate_provider_count

      FROM (
        SELECT
          provider_id

        FROM `{batch_input_table}`

        GROUP BY
          provider_id

        HAVING
          COUNT(*) > 1
      )
    )

    SELECT
      CURRENT_TIMESTAMP()
        AS monitoring_timestamp,

      'random_forest'
        AS model_name,

      n.total_feature_null_count,

      d.duplicate_provider_count,

      CASE
        WHEN n.total_feature_null_count > 0
          THEN 'ALERT'

        WHEN d.duplicate_provider_count > 0
          THEN 'ALERT'

        ELSE 'PASS'
      END AS monitoring_status

    FROM null_summary n
    CROSS JOIN duplicate_summary d
    """


    print(
        "\nRunning data quality monitoring..."
    )

    bq.query(
        data_quality_query
    ).result()

    print(
        f"Data quality results written to: "
        f"{data_quality_table}"
    )


    # =========================================================
    # B. FEATURE DRIFT
    #
    # Relative mean shift:
    #
    # |current_mean - baseline_mean|
    # --------------------------------
    #       |baseline_mean|
    #
    # Example:
    #
    # TRAIN mean = 0.50
    # Current mean = 0.56
    #
    # shift = 0.06 / 0.50
    #       = 12%
    #
    # > 10% => ALERT
    # =========================================================

    feature_drift_queries = []


    for feature in feature_columns:

        feature_query = f"""
        SELECT
          '{feature}'
            AS feature_name,

          baseline_mean,

          current_mean,

          CASE
            WHEN baseline_mean = 0
              AND current_mean = 0
              THEN 0.0

            WHEN baseline_mean = 0
              THEN 1.0

            ELSE SAFE_DIVIDE(
              ABS(
                current_mean
                - baseline_mean
              ),
              ABS(
                baseline_mean
              )
            )
          END AS relative_mean_shift

        FROM (
          SELECT

            (
              SELECT
                AVG(`{feature}`)

              FROM `{feature_table}`

              WHERE
                dataset_split = 'TRAIN'
            ) AS baseline_mean,

            (
              SELECT
                AVG(`{feature}`)

              FROM `{batch_input_table}`
            ) AS current_mean
        )
        """

        feature_drift_queries.append(
            feature_query
        )


    feature_drift_union = (
        "\nUNION ALL\n".join(
            feature_drift_queries
        )
    )


    feature_drift_query = f"""
    CREATE OR REPLACE TABLE `{feature_drift_table}` AS

    WITH drift AS (

      {feature_drift_union}

    )

    SELECT
      CURRENT_TIMESTAMP()
        AS monitoring_timestamp,

      'random_forest'
        AS model_name,

      feature_name,

      baseline_mean,

      current_mean,

      relative_mean_shift,

      {feature_shift_threshold}
        AS alert_threshold,

      CASE
        WHEN relative_mean_shift
             > {feature_shift_threshold}
          THEN 'ALERT'

        ELSE 'PASS'
      END AS monitoring_status

    FROM drift

    ORDER BY
      relative_mean_shift DESC
    """


    print(
        "\nRunning feature drift monitoring..."
    )

    bq.query(
        feature_drift_query
    ).result()

    print(
        f"Feature drift results written to: "
        f"{feature_drift_table}"
    )


    # =========================================================
    # C. PREDICTION DRIFT
    #
    # Absolute positive-rate shift:
    #
    # |current positive rate - baseline positive rate|
    #
    # Example:
    #
    # baseline = 10%
    # current  = 23%
    #
    # shift = 13 percentage points
    #
    # > 10% => ALERT
    # =========================================================

    prediction_drift_query = f"""
    CREATE OR REPLACE TABLE `{prediction_drift_table}` AS

    WITH prediction_summary AS (

      SELECT
        AVG(
          CASE
            WHEN fraud_score = 1.0
              THEN 1.0
            ELSE 0.0
          END
        ) AS current_positive_rate

      FROM `{provider_prediction_table}`
    )

    SELECT
      CURRENT_TIMESTAMP()
        AS monitoring_timestamp,

      'random_forest'
        AS model_name,

      {baseline_positive_rate}
        AS baseline_positive_rate,

      current_positive_rate,

      ABS(
        current_positive_rate
        - {baseline_positive_rate}
      ) AS prediction_rate_shift,

      {prediction_shift_threshold}
        AS alert_threshold,

      CASE
        WHEN ABS(
          current_positive_rate
          - {baseline_positive_rate}
        ) > {prediction_shift_threshold}
          THEN 'ALERT'

        ELSE 'PASS'
      END AS monitoring_status

    FROM prediction_summary
    """


    print(
        "\nRunning prediction drift monitoring..."
    )

    bq.query(
        prediction_drift_query
    ).result()

    print(
        f"Prediction drift results written to: "
        f"{prediction_drift_table}"
    )


    # =========================================================
    # D. MONITORING SUMMARY
    # =========================================================

    data_quality_result = list(
        bq.query(
            f"""
            SELECT
              monitoring_status

            FROM `{data_quality_table}`
            """
        ).result()
    )[0]


    feature_drift_result = list(
        bq.query(
            f"""
            SELECT
              COUNTIF(
                monitoring_status = 'ALERT'
              ) AS alert_features

            FROM `{feature_drift_table}`
            """
        ).result()
    )[0]


    prediction_drift_result = list(
        bq.query(
            f"""
            SELECT
              monitoring_status

            FROM `{prediction_drift_table}`
            """
        ).result()
    )[0]


    print("\n" + "=" * 60)
    print("Monitoring Summary")
    print("=" * 60)

    print(
        "Data Quality: "
        f"{data_quality_result.monitoring_status}"
    )

    print(
        "Feature Drift: "
        f"{feature_drift_result.alert_features} "
        "feature(s) above threshold"
    )

    print(
        "Prediction Drift: "
        f"{prediction_drift_result.monitoring_status}"
    )


    # ---------------------------------------------------------
    # Overall status
    # ---------------------------------------------------------
    overall_alert = (
        data_quality_result.monitoring_status
        == "ALERT"

        or feature_drift_result.alert_features
        > 0

        or prediction_drift_result.monitoring_status
        == "ALERT"
    )


    if overall_alert:
        print(
            "\nOVERALL MONITORING STATUS: ALERT"
        )
    else:
        print(
            "\nOVERALL MONITORING STATUS: PASS"
        )


    print("\n" + "=" * 60)
    print("Random Forest Monitoring Completed")
    print("=" * 60)
