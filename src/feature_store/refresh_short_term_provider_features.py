import argparse
from datetime import date, timedelta

from google.cloud import bigquery

from src.config.load_config import load_config


def refresh_short_term_provider_features(snapshot_date: date):
    config = load_config()

    project_id = config["project_id"]
    claims_table = config["data"]["claims_table"]
    history_table = config["feature_store"]["history_table"]
    short_term_days = int(
        config["feature_store"]["refresh"]["short_term"]["short_term_days"]
    )

    client = bigquery.Client(project=project_id)

    query = f"""
    MERGE `{history_table}` AS target

    USING (

      WITH recent_claims AS (

        SELECT
          CAST(provider_id AS STRING) AS provider_id,
          service_date,
          is_high_acuity_em,
          is_referral,
          total_paid_amount,
          member_provider_distance

        FROM `{claims_table}`

        WHERE service_date BETWEEN
          DATE_SUB(
            @snapshot_date,
            INTERVAL {short_term_days - 1} DAY
          )
          AND @snapshot_date
      ),

      short_term AS (

        SELECT
          provider_id,
          COUNT(*) AS claim_count_7d,
          AVG(CAST(is_high_acuity_em AS FLOAT64))
            AS high_acuity_pct_7d,
          AVG(CAST(is_referral AS FLOAT64))
            AS referral_rate_7d,
          AVG(total_paid_amount)
            AS avg_paid_per_claim_7d,
          AVG(member_provider_distance)
            AS avg_member_provider_distance_7d

        FROM recent_claims
        GROUP BY provider_id
      ),

      peer_baseline AS (

        SELECT
          AVG(high_acuity_pct_7d)
            AS expected_high_acuity_pct_7d

        FROM short_term
      ),

      latest_long_term AS (

        SELECT
          entity_id,
          provider_id,
          high_acuity_pct,
          pct_99214,
          pct_99215,
          referral_rate,
          eye_procedure_pct,
          avg_paid_per_claim,
          high_acuity_oe_ratio,
          avg_member_provider_distance

        FROM `{history_table}`

        WHERE DATE(feature_timestamp) <= @snapshot_date
          AND high_acuity_pct IS NOT NULL

        QUALIFY ROW_NUMBER() OVER (
          PARTITION BY entity_id
          ORDER BY feature_timestamp DESC
        ) = 1
      )

      SELECT
        s.provider_id AS entity_id,
        s.provider_id,

        TIMESTAMP(
          DATETIME(
            @snapshot_date,
            TIME(23, 59, 59)
          )
        ) AS feature_timestamp,

        l.high_acuity_pct,
        l.pct_99214,
        l.pct_99215,
        l.referral_rate,
        l.eye_procedure_pct,
        l.avg_paid_per_claim,
        l.high_acuity_oe_ratio,
        l.avg_member_provider_distance,

        s.claim_count_7d,
        s.high_acuity_pct_7d,
        s.referral_rate_7d,
        s.avg_paid_per_claim_7d,
        s.avg_member_provider_distance_7d,

        SAFE_DIVIDE(
          s.high_acuity_pct_7d,
          p.expected_high_acuity_pct_7d
        ) AS high_acuity_oe_ratio_7d

      FROM short_term s
      LEFT JOIN latest_long_term l
        ON s.provider_id = l.provider_id
      CROSS JOIN peer_baseline p

    ) AS source

    ON
      target.entity_id = source.entity_id
      AND target.feature_timestamp = source.feature_timestamp

    WHEN MATCHED THEN
      UPDATE SET
        claim_count_7d = source.claim_count_7d,
        high_acuity_pct_7d = source.high_acuity_pct_7d,
        referral_rate_7d = source.referral_rate_7d,
        avg_paid_per_claim_7d = source.avg_paid_per_claim_7d,
        avg_member_provider_distance_7d =
          source.avg_member_provider_distance_7d,
        high_acuity_oe_ratio_7d = source.high_acuity_oe_ratio_7d

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
    print("REFRESH DAILY SHORT-TERM PROVIDER FEATURES")
    print("=" * 70)
    print(f"Snapshot date: {snapshot_date}")
    print(f"Short-term window: {short_term_days} days")

    client.query(
        query,
        job_config=job_config,
    ).result()

    print("Daily short-term feature refresh completed.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Refresh 7-day provider behavioral features daily and carry "
            "forward the latest monthly long-term provider profile."
        )
    )

    parser.add_argument(
        "--snapshot-date",
        required=False,
        help=(
            "Snapshot date in YYYY-MM-DD. If omitted, defaults to "
            "yesterday so the daily refresh uses a completed day."
        ),
    )

    args = parser.parse_args()

    if args.snapshot_date:
        snapshot_date = date.fromisoformat(args.snapshot_date)
    else:
        snapshot_date = date.today() - timedelta(days=1)

    refresh_short_term_provider_features(
        snapshot_date=snapshot_date
    )


if __name__ == "__main__":
    main()
