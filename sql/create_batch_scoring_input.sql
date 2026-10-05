#score the test dataset with the xgboost model with the features used for training
CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.xgboost_batch_input`
AS

SELECT
  CAST(high_acuity_pct AS FLOAT64) AS high_acuity_pct,
  CAST(pct_99214 AS FLOAT64) AS pct_99214,
  CAST(pct_99215 AS FLOAT64) AS pct_99215,
  CAST(referral_rate AS FLOAT64) AS referral_rate,
  CAST(eye_procedure_pct AS FLOAT64) AS eye_procedure_pct,
  CAST(avg_paid_per_claim AS FLOAT64) AS avg_paid_per_claim,
  CAST(high_acuity_oe_ratio AS FLOAT64) AS high_acuity_oe_ratio

FROM
  `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

WHERE
  dataset_split = 'TEST';

