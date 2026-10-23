import argparse
import calendar
from datetime import date

from google.cloud import bigquery

from src.config.load_config import load_config


def month_end(value: str) -> date:
    year, month, _ = map(int, value.split("-"))
    return date(
        year,
        month,
        calendar.monthrange(year, month)[1],
    )


def refresh_provider_feature_history(
    snapshot_date: date,
):
    config = load_config()

    project_id = config["project_id"]
    claims_table = config["data"]["claims_table"]

    feature_store_config = config["feature_store"]
    history_table = feature_store_config["history_table"]

    refresh_config = feature_store_config["refresh"]
    long_term_months = int(
        refresh_config["long_term_months"]
    )
    short_term_days = int(
        refresh_config["short_term_days"]
    )

    if long_term_months <= 0:
        raise ValueError(
            "long_term_months must be positive."
        )

    if short_term_days <= 0:
        raise ValueError(
            "short_term_days must be positive."
        )

    client = bigquery.Client(
        project=project_id
    )

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS
      `{history_table}`
    (
      entity_id STRING,
      provider_id STRING,
      feature_timestamp TIMESTAMP,

      high_acuity_pct FLOAT64,
      pct_99214 FLOAT64,
      pct_99215 FLOAT64,
      referral_rate FLOAT64,
      eye_procedure_pct FLOAT64,
      avg_paid_per_claim FLOAT64,
      high_acuity_oe_ratio FLOAT64,
      avg_member_provider_distance FLOAT64,

      claim_count_7d INT64,
      high_acuity_pct_7d FLOAT64,
      referral_rate_7d FLOAT64,
      avg_paid_per_claim_7d FLOAT64,
      avg_member_provider_distance_7d FLOAT64,
      high_acuity_oe_ratio_7d FLOAT64
    )
    PARTITION BY DATE(feature_timestamp)
    CLUSTER BY provider_id
    """

    client.query(
        create_table_sql
    ).result()

    # Backward-compatible schema updates for an existing history table.
    # CREATE TABLE IF NOT EXISTS does not add newly introduced columns
    # when the table already exists, so explicitly add every short-term
    # feature used by the MERGE.
    for column_name, column_type in [
        ("claim_count_7d", "INT64"),
        ("high_acuity_pct_7d", "FLOAT64"),
        ("referral_rate_7d", "FLOAT64"),
        ("avg_paid_per_claim_7d", "FLOAT64"),
        ("avg_member_provider_distance_7d", "FLOAT64"),
        ("high_acuity_oe_ratio_7d", "FLOAT64"),
    ]:
        client.query(
            f"""
            ALTER TABLE `{history_table}`
            ADD COLUMN IF NOT EXISTS
              {column_name} {column_type}
            """
        ).result()

    query = f"""
    MERGE `{history_table}` AS target

    USING (

      WITH scoped_claims AS (

        SELECT
          CAST(provider_id AS STRING) AS provider_id,
          service_date,
          is_high_acuity_em,
          is_99214,
          is_99215,
          is_referral,
          is_eye_procedure,
          total_paid_amount,
          member_provider_distance

        FROM `{claims_table}`

        WHERE service_date BETWEEN
          DATE_ADD(
            DATE_SUB(
              @snapshot_date,
              INTERVAL {long_term_months} MONTH
            ),
            INTERVAL 1 DAY
          )
          AND @snapshot_date
      ),

      provider_long_term AS (

        SELECT
          provider_id,

          AVG(
            CAST(is_high_acuity_em AS FLOAT64)
          ) AS high_acuity_pct,

          AVG(
            CAST(is_99214 AS FLOAT64)
          ) AS pct_99214,

          AVG(
            CAST(is_99215 AS FLOAT64)
          ) AS pct_99215,

          AVG(
            CAST(is_referral AS FLOAT64)
          ) AS referral_rate,

          AVG(
            CAST(is_eye_procedure AS FLOAT64)
          ) AS eye_procedure_pct,

          AVG(
            total_paid_amount
          ) AS avg_paid_per_claim,

          AVG(
            member_provider_distance
          ) AS avg_member_provider_distance,

          COUNTIF(
            service_date BETWEEN
              DATE_SUB(
                @snapshot_date,
                INTERVAL {short_term_days - 1} DAY
              )
              AND @snapshot_date
          ) AS claim_count_7d,

          AVG(
            IF(
              service_date BETWEEN
                DATE_SUB(
                  @snapshot_date,
                  INTERVAL {short_term_days - 1} DAY
                )
                AND @snapshot_date,
              CAST(is_high_acuity_em AS FLOAT64),
              NULL
            )
          ) AS high_acuity_pct_7d,

          AVG(
            IF(
              service_date BETWEEN
                DATE_SUB(
                  @snapshot_date,
                  INTERVAL {short_term_days - 1} DAY
                )
                AND @snapshot_date,
              CAST(is_referral AS FLOAT64),
              NULL
            )
          ) AS referral_rate_7d,

          AVG(
            IF(
              service_date BETWEEN
                DATE_SUB(
                  @snapshot_date,
                  INTERVAL {short_term_days - 1} DAY
                )
                AND @snapshot_date,
              total_paid_amount,
              NULL
            )
          ) AS avg_paid_per_claim_7d,

          AVG(
            IF(
              service_date BETWEEN
                DATE_SUB(
                  @snapshot_date,
                  INTERVAL {short_term_days - 1} DAY
                )
                AND @snapshot_date,
              member_provider_distance,
              NULL
            )
          ) AS avg_member_provider_distance_7d

        FROM scoped_claims

        GROUP BY provider_id
      ),

      peer_baseline AS (

        SELECT
          AVG(high_acuity_pct)
            AS expected_high_acuity_pct,

          AVG(high_acuity_pct_7d)
            AS expected_high_acuity_pct_7d

        FROM provider_long_term
      )

      SELECT
        p.provider_id AS entity_id,
        p.provider_id,

        TIMESTAMP(
          DATETIME(
            @snapshot_date,
            TIME(23, 59, 59)
          )
        ) AS feature_timestamp,

        p.high_acuity_pct,
        p.pct_99214,
        p.pct_99215,
        p.referral_rate,
        p.eye_procedure_pct,
        p.avg_paid_per_claim,

        SAFE_DIVIDE(
          p.high_acuity_pct,
          b.expected_high_acuity_pct
        ) AS high_acuity_oe_ratio,

        p.avg_member_provider_distance,

        p.claim_count_7d,
        p.high_acuity_pct_7d,
        p.referral_rate_7d,
        p.avg_paid_per_claim_7d,
        p.avg_member_provider_distance_7d,

        SAFE_DIVIDE(
          p.high_acuity_pct_7d,
          b.expected_high_acuity_pct_7d
        ) AS high_acuity_oe_ratio_7d

      FROM provider_long_term p
      CROSS JOIN peer_baseline b

    ) AS source

    ON
      target.entity_id = source.entity_id
      AND target.feature_timestamp =
          source.feature_timestamp

    WHEN MATCHED THEN
      UPDATE SET
        high_acuity_pct =
          source.high_acuity_pct,
        pct_99214 =
          source.pct_99214,
        pct_99215 =
          source.pct_99215,
        referral_rate =
          source.referral_rate,
        eye_procedure_pct =
          source.eye_procedure_pct,
        avg_paid_per_claim =
          source.avg_paid_per_claim,
        high_acuity_oe_ratio =
          source.high_acuity_oe_ratio,
        avg_member_provider_distance =
          source.avg_member_provider_distance,
        claim_count_7d =
          source.claim_count_7d,
        high_acuity_pct_7d =
          source.high_acuity_pct_7d,
        referral_rate_7d =
          source.referral_rate_7d,
        avg_paid_per_claim_7d =
          source.avg_paid_per_claim_7d,
        avg_member_provider_distance_7d =
          source.avg_member_provider_distance_7d,
        high_acuity_oe_ratio_7d =
          source.high_acuity_oe_ratio_7d

    WHEN NOT MATCHED THEN
      INSERT (
        entity_id,
        provider_id,
        feature_timestamp,
        high_acuity_pct,
        pct_99214,
        pct_99215,
        referral_rate,
        eye_procedure_pct,
        avg_paid_per_claim,
        high_acuity_oe_ratio,
        avg_member_provider_distance,
        claim_count_7d,
        high_acuity_pct_7d,
        referral_rate_7d,
        avg_paid_per_claim_7d,
        avg_member_provider_distance_7d,
        high_acuity_oe_ratio_7d
      )
      VALUES (
        source.entity_id,
        source.provider_id,
        source.feature_timestamp,
        source.high_acuity_pct,
        source.pct_99214,
        source.pct_99215,
        source.referral_rate,
        source.eye_procedure_pct,
        source.avg_paid_per_claim,
        source.high_acuity_oe_ratio,
        source.avg_member_provider_distance,
        source.claim_count_7d,
        source.high_acuity_pct_7d,
        source.referral_rate_7d,
        source.avg_paid_per_claim_7d,
        source.avg_member_provider_distance_7d,
        source.high_acuity_oe_ratio_7d
      )
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "snapshot_date",
                "DATE",
                snapshot_date,
            )
        ]
    )

    print("=" * 70)
    print("REFRESH PROVIDER FEATURE HISTORY")
    print("=" * 70)

    print(
        f"Snapshot date: {snapshot_date}"
    )

    print(
        "Long-term window: "
        f"{long_term_months} months"
    )

    print(
        "Short-term window: "
        f"{short_term_days} days"
    )

    client.query(
        query,
        job_config=job_config,
    ).result()

    verify_query = f"""
    SELECT
      COUNT(*) AS providers,
      MIN(feature_timestamp) AS min_ts,
      MAX(feature_timestamp) AS max_ts

    FROM `{history_table}`

    WHERE DATE(feature_timestamp) =
      @snapshot_date
    """

    row = next(
        iter(
            client.query(
                verify_query,
                job_config=job_config,
            ).result()
        )
    )

    print(
        f"Providers written: {row.providers}"
    )

    print(
        f"Snapshot timestamp: {row.max_ts}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the monthly provider feature snapshot "
            "used by BigQuery batch serving and Vertex AI "
            "Feature Store."
        )
    )

    parser.add_argument(
        "--snapshot-date",
        required=False,
        help=(
            "Snapshot date in YYYY-MM-DD. "
            "Defaults to the month end of "
            "data.current_data_month."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    if args.snapshot_date:
        snapshot_date = date.fromisoformat(
            args.snapshot_date
        )
    else:
        snapshot_date = month_end(
            config["data"][
                "current_data_month"
            ]
        )

    refresh_provider_feature_history(
        snapshot_date=snapshot_date
    )


if __name__ == "__main__":
    main()
