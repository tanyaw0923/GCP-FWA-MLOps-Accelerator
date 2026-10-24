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
    provider_feature_history_table: str,
    output_feature_table: str,
    feature_sql_json: str,
    derived_feature_sql_json: str,
    feature_columns_json: str,
    train_start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
):
    """
    Build the model TRAIN / TEST table from the canonical timestamped
    provider feature history.

    The feature history is the source of model features for both
    training and batch scoring. Raw claims are used here only for
    synthetic demo labels and provider metadata.

    The short-term 7-day features intentionally remain outside the
    current model feature list until their modeling design is finalized.
    """
    import json
    import re

    from google.cloud import bigquery

    feature_columns = json.loads(feature_columns_json)
    feature_sql = json.loads(feature_sql_json)
    derived_feature_sql = json.loads(derived_feature_sql_json)

    if not feature_columns:
        raise ValueError("feature_columns_json is empty.")

    safe_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for feature_name in feature_columns:
        if not safe_name.fullmatch(feature_name):
            raise ValueError(
                f"Invalid configured feature name: {feature_name}"
            )

    configured_feature_names = (
        set(feature_sql)
        | set(derived_feature_sql)
    )

    missing_definitions = (
        set(feature_columns)
        - configured_feature_names
    )

    if missing_definitions:
        raise ValueError(
            "Configured model features do not have a matching "
            "feature definition: "
            f"{sorted(missing_definitions)}"
        )

    table_pattern = re.compile(
        r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$"
    )

    for table_name in [
        claims_table,
        provider_feature_history_table,
        output_feature_table,
    ]:
        if not table_pattern.fullmatch(table_name):
            raise ValueError(
                f"Invalid BigQuery table: {table_name}"
            )

    bq = bigquery.Client(project=project_id)

    history_table = bq.get_table(
        provider_feature_history_table
    )

    history_columns = {
        field.name
        for field in history_table.schema
    }

    required_history_columns = {
        "provider_id",
        "feature_timestamp",
        *feature_columns,
    }

    missing_history_columns = (
        required_history_columns
        - history_columns
    )

    if missing_history_columns:
        raise RuntimeError(
            "Canonical provider feature history is missing "
            "required training columns: "
            f"{sorted(missing_history_columns)}"
        )

    feature_select_sql = ",\n        ".join(
        f"h.{feature_name}"
        for feature_name in feature_columns
    )

    query = f"""
    CREATE OR REPLACE TABLE
      `{output_feature_table}` AS

    WITH monthly_history AS (
      SELECT
        provider_id,
        DATE(feature_timestamp) AS feature_snapshot_date,
        {", ".join(feature_columns)}

      FROM `{provider_feature_history_table}`

      WHERE DATE(feature_timestamp) BETWEEN
        DATE(@train_start_date)
        AND DATE(@test_end_date)

      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
          provider_id,
          DATE_TRUNC(
            DATE(feature_timestamp),
            MONTH
          )
        ORDER BY feature_timestamp DESC
      ) = 1
    ),

    labeled_history AS (
      SELECT
        h.*,

        CASE
          WHEN h.feature_snapshot_date BETWEEN
            DATE(@train_start_date)
            AND DATE(@train_end_date)
            THEN 'TRAIN'

          WHEN h.feature_snapshot_date BETWEEN
            DATE(@test_start_date)
            AND DATE(@test_end_date)
            THEN 'TEST'

          ELSE NULL
        END AS dataset_split

      FROM monthly_history h
    ),

    provider_metadata AS (
      SELECT
        CAST(provider_id AS STRING) AS provider_id,
        ANY_VALUE(provider_name) AS provider_name,
        ANY_VALUE(provider_specialty) AS provider_specialty,
        ANY_VALUE(provider_city) AS provider_city,
        ANY_VALUE(provider_state) AS provider_state,
        MAX(CAST(fraud_provider_label AS INT64)) AS fraud_label

      FROM `{claims_table}`

      GROUP BY provider_id
    ),

    monthly_claim_stats AS (
      SELECT
        CAST(provider_id AS STRING) AS provider_id,
        DATE_TRUNC(service_date, MONTH) AS service_month,
        COUNT(*) AS claim_count,
        COUNT(DISTINCT member_id) AS unique_members

      FROM `{claims_table}`

      WHERE service_date BETWEEN
        DATE(@train_start_date)
        AND DATE(@test_end_date)

      GROUP BY
        provider_id,
        service_month
    )

    SELECT
      h.provider_id,
      m.provider_name,
      m.provider_specialty,
      m.provider_city,
      m.provider_state,

      h.feature_snapshot_date,
      h.dataset_split,
      m.fraud_label,

      COALESCE(c.claim_count, 0) AS claim_count,
      COALESCE(c.unique_members, 0) AS unique_members,

      {feature_select_sql}

    FROM labeled_history h

    JOIN provider_metadata m
      USING(provider_id)

    LEFT JOIN monthly_claim_stats c
      ON h.provider_id = c.provider_id
      AND DATE_TRUNC(
        h.feature_snapshot_date,
        MONTH
      ) = c.service_month

    WHERE h.dataset_split IS NOT NULL

    ORDER BY
      h.dataset_split,
      h.feature_snapshot_date,
      h.provider_id
    """

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

    print("=" * 70)
    print("BUILD POINT-IN-TIME MODEL DATASET")
    print("=" * 70)
    print(
        "Canonical feature source: "
        f"{provider_feature_history_table}"
    )
    print(
        f"TRAIN: {train_start_date} -> {train_end_date}"
    )
    print(
        f"TEST:  {test_start_date} -> {test_end_date}"
    )
    print(
        f"Configured model features: {feature_columns}"
    )

    bq.query(
        query,
        job_config=job_config,
    ).result()

    null_checks = ",\n".join(
        (
            "COUNTIF("
            f"{feature_name} IS NULL"
            ") AS "
            f"{feature_name}_nulls"
        )
        for feature_name in feature_columns
    )

    validation_query = f"""
    SELECT
      dataset_split,
      COUNT(*) AS snapshot_rows,
      COUNT(DISTINCT provider_id) AS providers,
      COUNT(DISTINCT feature_snapshot_date) AS snapshot_dates,
      COUNTIF(fraud_label = 1) AS fraud_rows,
      COUNTIF(fraud_label = 0) AS normal_rows,
      {null_checks}

    FROM `{output_feature_table}`

    GROUP BY dataset_split

    ORDER BY dataset_split
    """

    rows = list(
        bq.query(validation_query).result()
    )

    if {
        row.dataset_split
        for row in rows
    } != {"TRAIN", "TEST"}:
        raise RuntimeError(
            "Point-in-time feature dataset must contain "
            "both TRAIN and TEST splits."
        )

    for row in rows:
        for feature_name in feature_columns:
            if getattr(
                row,
                f"{feature_name}_nulls",
            ) > 0:
                raise RuntimeError(
                    "NULL values found in canonical feature "
                    f"{feature_name} for {row.dataset_split}."
                )

        print(
            f"{row.dataset_split}: "
            f"{row.snapshot_rows} rows, "
            f"{row.providers} providers, "
            f"{row.snapshot_dates} snapshots"
        )

    print(
        "Point-in-time feature validation: PASS"
    )
