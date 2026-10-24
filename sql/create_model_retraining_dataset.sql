-- ============================================================
-- INVESTIGATOR-LABELED RETRAINING DATASET
--
-- Each completed investigation is joined to the latest provider
-- feature snapshot that existed ON OR BEFORE the investigation
-- date. This prevents using future feature values when rebuilding
-- a supervised training set from feedback.
-- ============================================================

CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.model_retraining_dataset`
AS

WITH latest_feedback AS (
  SELECT
    provider_id,
    confirmed_fraud,
    investigation_result,
    investigator,
    investigation_date,
    run_id,
    model_name,
    model_version,
    fraud_amount,
    created_at

  FROM
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`

  WHERE confirmed_fraud IS NOT NULL

  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY provider_id
    ORDER BY
      investigation_date DESC,
      created_at DESC
  ) = 1
),

point_in_time_features AS (
  SELECT
    f.provider_id,
    f.feature_timestamp,

    f.high_acuity_pct,
    f.pct_99214,
    f.pct_99215,
    f.referral_rate,
    f.eye_procedure_pct,
    f.avg_paid_per_claim,
    f.high_acuity_oe_ratio,
    f.avg_member_provider_distance,

    feedback.confirmed_fraud,
    feedback.investigation_result,
    feedback.investigator,
    feedback.investigation_date,
    feedback.run_id,
    feedback.model_name,
    feedback.model_version,
    feedback.fraud_amount

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_feature_history` f

  JOIN latest_feedback feedback
    USING(provider_id)

  WHERE
    DATE(f.feature_timestamp)
    <= feedback.investigation_date

  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY f.provider_id
    ORDER BY f.feature_timestamp DESC
  ) = 1
)

SELECT
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

  CAST(
    confirmed_fraud
    AS INT64
  ) AS fraud_label,

  investigation_result,
  investigator,
  investigation_date,
  run_id AS source_run_id,
  model_name AS source_model_name,
  model_version AS source_model_version,
  fraud_amount

FROM point_in_time_features;


-- QA
SELECT
  fraud_label,
  COUNT(*) AS providers,
  MIN(feature_timestamp) AS earliest_feature_snapshot,
  MAX(feature_timestamp) AS latest_feature_snapshot

FROM
  `fwa-mlops-accelerator-demo.fraud_features.model_retraining_dataset`

GROUP BY fraud_label

ORDER BY fraud_label;
