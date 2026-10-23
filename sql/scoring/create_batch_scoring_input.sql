-- ============================================================
-- CANONICAL BATCH SCORING INPUT
--
-- Training and inference use the same provider feature history.
-- The latest complete provider snapshot is selected for scoring.
--
-- provider_id is preserved as lineage metadata. It is not a model
-- feature; the serving layer removes it before constructing the
-- numeric model matrix.
-- ============================================================

CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.xgboost_batch_input`
AS

WITH latest_snapshot AS (
  SELECT
    CAST(provider_id AS STRING) AS provider_id,
    feature_timestamp,

    CAST(high_acuity_pct AS FLOAT64)
      AS high_acuity_pct,
    CAST(pct_99214 AS FLOAT64)
      AS pct_99214,
    CAST(pct_99215 AS FLOAT64)
      AS pct_99215,
    CAST(referral_rate AS FLOAT64)
      AS referral_rate,
    CAST(eye_procedure_pct AS FLOAT64)
      AS eye_procedure_pct,
    CAST(avg_paid_per_claim AS FLOAT64)
      AS avg_paid_per_claim,
    CAST(high_acuity_oe_ratio AS FLOAT64)
      AS high_acuity_oe_ratio

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_feature_history`

  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY provider_id
    ORDER BY feature_timestamp DESC
  ) = 1
)

SELECT
  provider_id,
  high_acuity_pct,
  pct_99214,
  pct_99215,
  referral_rate,
  eye_procedure_pct,
  avg_paid_per_claim,
  high_acuity_oe_ratio

FROM latest_snapshot;


-- Random Forest uses the same canonical feature contract.
CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`
AS

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_features.xgboost_batch_input`;


-- QA
SELECT
  COUNT(*) AS providers,
  COUNT(DISTINCT provider_id) AS unique_providers,
  COUNTIF(provider_id IS NULL) AS missing_provider_id

FROM
  `fwa-mlops-accelerator-demo.fraud_features.xgboost_batch_input`;
