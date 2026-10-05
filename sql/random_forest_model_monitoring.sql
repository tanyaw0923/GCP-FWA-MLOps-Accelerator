--confirm random forest prediction column type and distribution
SELECT
  prediction,
  COUNT(*) AS count
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T13_08_45_116Z_589`
GROUP BY prediction
ORDER BY prediction;

-- Random Forest batch-input data quality checks
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT provider_id) AS unique_providers,

  COUNTIF(provider_id IS NULL) AS null_provider_id,

  COUNTIF(high_acuity_pct IS NULL) AS null_high_acuity_pct,
  COUNTIF(pct_99214 IS NULL) AS null_pct_99214,
  COUNTIF(pct_99215 IS NULL) AS null_pct_99215,
  COUNTIF(referral_rate IS NULL) AS null_referral_rate,
  COUNTIF(eye_procedure_pct IS NULL) AS null_eye_procedure_pct,
  COUNTIF(avg_paid_per_claim IS NULL) AS null_avg_paid_per_claim,
  COUNTIF(high_acuity_oe_ratio IS NULL) AS null_high_acuity_oe_ratio,

  COUNTIF(high_acuity_pct < 0 OR high_acuity_pct > 1)
    AS invalid_high_acuity_pct,

  COUNTIF(pct_99214 < 0 OR pct_99214 > 1)
    AS invalid_pct_99214,

  COUNTIF(pct_99215 < 0 OR pct_99215 > 1)
    AS invalid_pct_99215,

  COUNTIF(referral_rate < 0 OR referral_rate > 1)
    AS invalid_referral_rate,

  COUNTIF(eye_procedure_pct < 0 OR eye_procedure_pct > 1)
    AS invalid_eye_procedure_pct,

  COUNTIF(avg_paid_per_claim < 0)
    AS invalid_avg_paid_per_claim,

  COUNTIF(high_acuity_oe_ratio < 0)
    AS invalid_high_acuity_oe_ratio

FROM
  `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`;


--create a monitoring snapshot table
CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.random_forest_data_quality`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  dataset_name STRING,

  total_rows INT64,
  unique_providers INT64,
  duplicate_rows INT64,

  null_feature_values INT64,
  invalid_feature_values INT64,

  data_quality_status STRING
);

--then insert into data quality check table with testing data
INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.random_forest_data_quality`

WITH dq AS (

  SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT provider_id) AS unique_providers,

    COUNT(*) - COUNT(DISTINCT provider_id)
      AS duplicate_rows,

    COUNTIF(high_acuity_pct IS NULL)
    + COUNTIF(pct_99214 IS NULL)
    + COUNTIF(pct_99215 IS NULL)
    + COUNTIF(referral_rate IS NULL)
    + COUNTIF(eye_procedure_pct IS NULL)
    + COUNTIF(avg_paid_per_claim IS NULL)
    + COUNTIF(high_acuity_oe_ratio IS NULL)
      AS null_feature_values,

    COUNTIF(high_acuity_pct < 0 OR high_acuity_pct > 1)
    + COUNTIF(pct_99214 < 0 OR pct_99214 > 1)
    + COUNTIF(pct_99215 < 0 OR pct_99215 > 1)
    + COUNTIF(referral_rate < 0 OR referral_rate > 1)
    + COUNTIF(eye_procedure_pct < 0 OR eye_procedure_pct > 1)
    + COUNTIF(avg_paid_per_claim < 0)
    + COUNTIF(high_acuity_oe_ratio < 0)
      AS invalid_feature_values

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`
)

SELECT
  CURRENT_TIMESTAMP(),
  'random_forest',
  'TEST',

  total_rows,
  unique_providers,
  duplicate_rows,
  null_feature_values,
  invalid_feature_values,

  CASE
    WHEN total_rows != 200 THEN 'FAIL'
    WHEN duplicate_rows > 0 THEN 'FAIL'
    WHEN null_feature_values > 0 THEN 'FAIL'
    WHEN invalid_feature_values > 0 THEN 'FAIL'
    ELSE 'PASS'
  END AS data_quality_status

FROM dq;

--qa check updated data quality table
SELECT *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.random_forest_data_quality`
ORDER BY monitoring_timestamp DESC;

--monitor feature drift using training dataset as baseline
WITH baseline AS (

  SELECT
    'high_acuity_pct' AS feature_name,
    AVG(high_acuity_pct) AS baseline_mean,
    STDDEV(high_acuity_pct) AS baseline_std

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'

  UNION ALL

  SELECT
    'pct_99214',
    AVG(pct_99214),
    STDDEV(pct_99214)

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'

  UNION ALL

  SELECT
    'pct_99215',
    AVG(pct_99215),
    STDDEV(pct_99215)

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'

  UNION ALL

  SELECT
    'referral_rate',
    AVG(referral_rate),
    STDDEV(referral_rate)

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'

  UNION ALL

  SELECT
    'eye_procedure_pct',
    AVG(eye_procedure_pct),
    STDDEV(eye_procedure_pct)

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'

  UNION ALL

  SELECT
    'avg_paid_per_claim',
    AVG(avg_paid_per_claim),
    STDDEV(avg_paid_per_claim)

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'

  UNION ALL

  SELECT
    'high_acuity_oe_ratio',
    AVG(high_acuity_oe_ratio),
    STDDEV(high_acuity_oe_ratio)

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  WHERE dataset_split = 'TRAIN'
),

current_data AS (

  SELECT
    'high_acuity_pct' AS feature_name,
    AVG(high_acuity_pct) AS current_mean

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL

  SELECT 'pct_99214', AVG(pct_99214)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL

  SELECT 'pct_99215', AVG(pct_99215)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL

  SELECT 'referral_rate', AVG(referral_rate)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL

  SELECT 'eye_procedure_pct', AVG(eye_procedure_pct)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL

  SELECT 'avg_paid_per_claim', AVG(avg_paid_per_claim)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL

  SELECT 'high_acuity_oe_ratio', AVG(high_acuity_oe_ratio)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`
)

SELECT
  b.feature_name,
  b.baseline_mean,
  c.current_mean,
  b.baseline_std,

  SAFE_DIVIDE(
    ABS(c.current_mean - b.baseline_mean),
    NULLIF(b.baseline_std, 0)
  ) AS standardized_mean_shift

FROM baseline b

JOIN current_data c
USING(feature_name)

ORDER BY standardized_mean_shift DESC;

--check prediction tables from batch inference
SELECT
  table_name,
  creation_time
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.INFORMATION_SCHEMA.TABLES`
WHERE
  STARTS_WITH(table_name, 'predictions')
ORDER BY creation_time DESC;

--check top 5 records for the random forest model prediction results
SELECT *
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T13_08_45_116Z_589`
LIMIT 5;

--check top 5 records for the isolation forest model prediction results
SELECT *
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T12_58_14_815Z_669`
LIMIT 5;

--monitor feature-drift
CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.feature_drift_summary`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  feature_name STRING,
  baseline_mean FLOAT64,
  current_mean FLOAT64,
  baseline_std FLOAT64,
  standardized_mean_shift FLOAT64,
  drift_status STRING
);


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.feature_drift_summary`

WITH baseline AS (
  SELECT 'high_acuity_pct' feature_name,
         AVG(high_acuity_pct) baseline_mean,
         STDDEV(high_acuity_pct) baseline_std
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'

  UNION ALL
  SELECT 'pct_99214', AVG(pct_99214), STDDEV(pct_99214)
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'

  UNION ALL
  SELECT 'pct_99215', AVG(pct_99215), STDDEV(pct_99215)
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'

  UNION ALL
  SELECT 'referral_rate', AVG(referral_rate), STDDEV(referral_rate)
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'

  UNION ALL
  SELECT 'eye_procedure_pct', AVG(eye_procedure_pct), STDDEV(eye_procedure_pct)
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'

  UNION ALL
  SELECT 'avg_paid_per_claim', AVG(avg_paid_per_claim), STDDEV(avg_paid_per_claim)
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'

  UNION ALL
  SELECT 'high_acuity_oe_ratio', AVG(high_acuity_oe_ratio), STDDEV(high_acuity_oe_ratio)
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'
),

current_data AS (
  SELECT 'high_acuity_pct' feature_name, AVG(high_acuity_pct) current_mean
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL
  SELECT 'pct_99214', AVG(pct_99214)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL
  SELECT 'pct_99215', AVG(pct_99215)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL
  SELECT 'referral_rate', AVG(referral_rate)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL
  SELECT 'eye_procedure_pct', AVG(eye_procedure_pct)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL
  SELECT 'avg_paid_per_claim', AVG(avg_paid_per_claim)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`

  UNION ALL
  SELECT 'high_acuity_oe_ratio', AVG(high_acuity_oe_ratio)
  FROM `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input`
),

drift AS (
  SELECT
    b.feature_name,
    b.baseline_mean,
    c.current_mean,
    b.baseline_std,
    SAFE_DIVIDE(
      ABS(c.current_mean - b.baseline_mean),
      NULLIF(b.baseline_std, 0)
    ) AS standardized_mean_shift
  FROM baseline b
  JOIN current_data c USING(feature_name)
)

SELECT
  CURRENT_TIMESTAMP(),
  'random_forest',
  feature_name,
  baseline_mean,
  current_mean,
  baseline_std,
  standardized_mean_shift,

  CASE
    WHEN standardized_mean_shift >= 1.0 THEN 'FAIL'
    WHEN standardized_mean_shift >= 0.5 THEN 'WARNING'
    ELSE 'PASS'
  END AS drift_status

FROM drift;

--create one standardized monitoring table:
CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  total_predictions INT64,
  positive_count INT64,
  negative_count INT64,
  positive_rate FLOAT64,
  baseline_rate FLOAT64,
  rate_change FLOAT64,
  monitoring_status STRING
);
INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`

WITH metrics AS (
  SELECT
    COUNT(*) AS total_predictions,
    COUNTIF(prediction = '1') AS positive_count,
    COUNTIF(prediction = '0') AS negative_count,
    SAFE_DIVIDE(COUNTIF(prediction = '1'), COUNT(*)) AS positive_rate
  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T13_08_45_116Z_589`
)

SELECT
  CURRENT_TIMESTAMP(),
  'random_forest',
  total_predictions,
  positive_count,
  negative_count,
  positive_rate,

  0.15 AS baseline_rate,

  ABS(positive_rate - 0.15) AS rate_change,

  CASE
    WHEN ABS(positive_rate - 0.15) >= 0.15 THEN 'FAIL'
    WHEN ABS(positive_rate - 0.15) >= 0.05 THEN 'WARNING'
    ELSE 'PASS'
  END

FROM metrics;

--qa monitoring table results
SELECT *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
ORDER BY monitoring_timestamp DESC;
