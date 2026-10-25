-- ============================================================
-- BATCH SCORING INPUT
--
-- Preserve provider_id as a key for downstream lineage while the
-- model receives only the configured numeric feature columns.
--
-- run_batch_prediction.py submits provider_id as Vertex key_field,
-- so Vertex excludes the key from the model instance and writes it
-- back to the prediction output as "key".
-- ============================================================

CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.xgboost_batch_input`
AS

SELECT
  CAST(provider_id AS STRING) AS provider_id,

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
  `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

WHERE
  dataset_split = 'TEST';


-- Random Forest uses the same scoring contract.
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
