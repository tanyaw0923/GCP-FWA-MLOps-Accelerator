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
    train_end_date: str,
    test_end_date: str,
    provider_training_fraction: float,
):
    """
    Build a provider-disjoint TRAIN / TEST dataset from canonical
    point-in-time feature history.

    Design:
      - one row per provider
      - TRAIN and TEST providers never overlap
      - split is deterministic and stratified by demo fraud label
      - TRAIN features use the train cutoff snapshot
      - TEST features use the later test cutoff snapshot
      - raw claims supply synthetic labels and metadata only

    The current 7-day short-term features intentionally remain outside
    the active model contract until their modeling design is finalized.
    """
    import json
    import re

    from google.cloud import bigquery

    feature_columns = json.loads(
        feature_columns_json
    )

    feature_sql = json.loads(
        feature_sql_json
    )

    derived_feature_sql = json.loads(
        derived_feature_sql_json
    )

    if not feature_columns:
        raise ValueError(
            "feature_columns_json is empty."
        )

    if not (
        0.0
        < provider_training_fraction
        < 1.0
    ):
        raise ValueError(
            "provider_training_fraction must be "
            "between 0 and 1."
        )

    safe_name = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    for feature_name in feature_columns:
        if not safe_name.fullmatch(
            feature_name
        ):
            raise ValueError(
                "Invalid configured feature name: "
                f"{feature_name}"
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
            "Configured model features do not have "
            "a matching feature definition: "
            f"{sorted(missing_definitions)}"
        )

    table_pattern = re.compile(
        r"^[A-Za-z0-9_-]+\."
        r"[A-Za-z0-9_]+\."
        r"[A-Za-z0-9_]+$"
    )

    for table_name in [
        claims_table,
        provider_feature_history_table,
        output_feature_table,
    ]:
        if not table_pattern.fullmatch(
            table_name
        ):
            raise ValueError(
                f"Invalid BigQuery table: "
                f"{table_name}"
            )

    bq = bigquery.Client(
        project=project_id
    )

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
            "Canonical provider feature history is "
            "missing required training columns: "
            f"{sorted(missing_history_columns)}"
        )

    feature_select_sql = ",\n      ".join(
        f"s.{feature_name}"
        for feature_name in feature_columns
    )

    query = f"""
    CREATE OR REPLACE TABLE
      `{output_feature_table}` AS

    WITH provider_metadata AS (
      SELECT
        CAST(provider_id AS STRING)
          AS provider_id,

        ANY_VALUE(provider_name)
          AS provider_name,

        ANY_VALUE(provider_specialty)
          AS provider_specialty,

        ANY_VALUE(provider_city)
          AS provider_city,

        ANY_VALUE(provider_state)
          AS provider_state,

        MAX(
          CAST(
            fraud_provider_label
            AS INT64
          )
        ) AS fraud_label

      FROM
        `{claims_table}`

      GROUP BY provider_id
    ),

    ranked_providers AS (
      SELECT
        *,

        ROW_NUMBER() OVER (
          PARTITION BY fraud_label
          ORDER BY
            FARM_FINGERPRINT(
              CONCAT(
                provider_id,
                '|fwa-provider-split-v1'
              )
            ),
            provider_id
        ) AS class_rank,

        COUNT(*) OVER (
          PARTITION BY fraud_label
        ) AS class_count

      FROM provider_metadata
    ),

    provider_split AS (
      SELECT
        provider_id,
        provider_name,
        provider_specialty,
        provider_city,
        provider_state,
        fraud_label,

        CASE
          WHEN class_rank <= LEAST(
            class_count - 1,
            GREATEST(
              1,
              CAST(
                FLOOR(
                  class_count
                  * @provider_training_fraction
                )
                AS INT64
              )
            )
          )
          THEN 'TRAIN'

          ELSE 'TEST'
        END AS dataset_split

      FROM ranked_providers
    ),

    train_snapshot AS (
      SELECT
        CAST(provider_id AS STRING)
          AS provider_id,

        DATE(feature_timestamp)
          AS feature_snapshot_date,

        {", ".join(feature_columns)}

      FROM
        `{provider_feature_history_table}`

      WHERE
        DATE(feature_timestamp)
        = DATE(@train_end_date)

      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY provider_id
        ORDER BY feature_timestamp DESC
      ) = 1
    ),

    test_snapshot AS (
      SELECT
        CAST(provider_id AS STRING)
          AS provider_id,

        DATE(feature_timestamp)
          AS feature_snapshot_date,

        {", ".join(feature_columns)}

      FROM
        `{provider_feature_history_table}`

      WHERE
        DATE(feature_timestamp)
        = DATE(@test_end_date)

      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY provider_id
        ORDER BY feature_timestamp DESC
      ) = 1
    ),

    selected_snapshot AS (
      SELECT
        p.provider_id,
        p.provider_name,
        p.provider_specialty,
        p.provider_city,
        p.provider_state,
        p.fraud_label,
        p.dataset_split,

        t.feature_snapshot_date,

        {", ".join(
            f"t.{feature_name}"
            for feature_name in feature_columns
        )}

      FROM provider_split p

      JOIN train_snapshot t
        USING(provider_id)

      WHERE
        p.dataset_split = 'TRAIN'

      UNION ALL

      SELECT
        p.provider_id,
        p.provider_name,
        p.provider_specialty,
        p.provider_city,
        p.provider_state,
        p.fraud_label,
        p.dataset_split,

        t.feature_snapshot_date,

        {", ".join(
            f"t.{feature_name}"
            for feature_name in feature_columns
        )}

      FROM provider_split p

      JOIN test_snapshot t
        USING(provider_id)

      WHERE
        p.dataset_split = 'TEST'
    ),

    claim_stats AS (
      SELECT
        CAST(provider_id AS STRING)
          AS provider_id,

        COUNTIF(
          service_date
          <= DATE(@test_end_date)
        ) AS claim_count,

        COUNT(
          DISTINCT IF(
            service_date
            <= DATE(@test_end_date),
            member_id,
            NULL
          )
        ) AS unique_members

      FROM
        `{claims_table}`

      GROUP BY provider_id
    )

    SELECT
      s.provider_id,
      s.provider_name,
      s.provider_specialty,
      s.provider_city,
      s.provider_state,

      s.feature_snapshot_date,
      s.dataset_split,
      s.fraud_label,

      COALESCE(
        c.claim_count,
        0
      ) AS claim_count,

      COALESCE(
        c.unique_members,
        0
      ) AS unique_members,

      {feature_select_sql}

    FROM selected_snapshot s

    LEFT JOIN claim_stats c
      USING(provider_id)

    ORDER BY
      s.dataset_split,
      s.provider_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "train_end_date",
                "STRING",
                train_end_date,
            ),
            bigquery.ScalarQueryParameter(
                "test_end_date",
                "STRING",
                test_end_date,
            ),
            bigquery.ScalarQueryParameter(
                "provider_training_fraction",
                "FLOAT64",
                provider_training_fraction,
            ),
        ]
    )

    print("=" * 70)
    print("BUILD PROVIDER-DISJOINT POINT-IN-TIME DATASET")
    print("=" * 70)

    print(
        "Canonical feature source: "
        f"{provider_feature_history_table}"
    )

    print(
        "Provider training fraction: "
        f"{provider_training_fraction:.2%}"
    )

    print(
        "TRAIN feature cutoff: "
        f"{train_end_date}"
    )

    print(
        "TEST feature cutoff: "
        f"{test_end_date}"
    )

    print(
        "Configured model features: "
        f"{feature_columns}"
    )

    bq.query(
        query,
        job_config=job_config,
    ).result()

    null_checks = ",\n      ".join(
        (
            "COUNTIF("
            f"{feature_name} IS NULL"
            ") AS "
            f"{feature_name}_nulls"
        )
        for feature_name in feature_columns
    )

    validation_query = f"""
    WITH split_summary AS (
      SELECT
        dataset_split,
        COUNT(*) AS provider_rows,
        COUNT(DISTINCT provider_id)
          AS unique_providers,
        COUNT(DISTINCT feature_snapshot_date)
          AS snapshot_dates,
        MIN(feature_snapshot_date)
          AS min_snapshot_date,
        MAX(feature_snapshot_date)
          AS max_snapshot_date,
        COUNTIF(fraud_label = 1)
          AS fraud_providers,
        COUNTIF(fraud_label = 0)
          AS normal_providers,
        {null_checks}

      FROM
        `{output_feature_table}`

      GROUP BY dataset_split
    ),

    overlap AS (
      SELECT
        COUNT(*) AS overlapping_providers

      FROM (
        SELECT provider_id

        FROM
          `{output_feature_table}`

        GROUP BY provider_id

        HAVING
          COUNT(DISTINCT dataset_split)
          > 1
      )
    )

    SELECT
      s.*,
      o.overlapping_providers

    FROM split_summary s

    CROSS JOIN overlap o

    ORDER BY dataset_split
    """

    rows = list(
        bq.query(
            validation_query
        ).result()
    )

    split_names = {
        row.dataset_split
        for row in rows
    }

    if split_names != {
        "TRAIN",
        "TEST",
    }:
        raise RuntimeError(
            "Provider dataset must contain "
            "both TRAIN and TEST."
        )

    for row in rows:
        if (
            row.provider_rows
            != row.unique_providers
        ):
            raise RuntimeError(
                "Each provider must appear exactly "
                "once in each selected split."
            )

        if row.overlapping_providers > 0:
            raise RuntimeError(
                "Provider leakage detected: "
                f"{row.overlapping_providers} provider(s) "
                "appear in both TRAIN and TEST."
            )

        if (
            row.fraud_providers == 0
            or row.normal_providers == 0
        ):
            raise RuntimeError(
                f"{row.dataset_split} must contain "
                "both fraud classes."
            )

        for feature_name in feature_columns:
            null_count = getattr(
                row,
                f"{feature_name}_nulls",
            )

            if null_count > 0:
                raise RuntimeError(
                    "NULL values found in canonical "
                    f"feature {feature_name} for "
                    f"{row.dataset_split}."
                )

        expected_snapshot_date = (
            train_end_date
            if row.dataset_split == "TRAIN"
            else test_end_date
        )

        actual_snapshot_date = str(
            row.max_snapshot_date
        )

        if (
            row.snapshot_dates != 1
            or actual_snapshot_date
            != expected_snapshot_date
        ):
            raise RuntimeError(
                f"{row.dataset_split} did not use "
                "the expected point-in-time snapshot. "
                f"Expected {expected_snapshot_date}; "
                f"found {row.min_snapshot_date} "
                f"to {row.max_snapshot_date}."
            )

        print(
            f"{row.dataset_split}: "
            f"{row.provider_rows} providers, "
            f"{row.fraud_providers} fraud, "
            f"{row.normal_providers} normal, "
            f"snapshot={row.max_snapshot_date}"
        )

    print(
        "Provider overlap: 0"
    )

    print(
        "Point-in-time provider split validation: PASS"
    )
