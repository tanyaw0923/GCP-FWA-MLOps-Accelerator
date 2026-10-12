-- ============================================================
-- GENERIC MODEL MONITORING
-- FWA MLOps Accelerator
--
-- Shared across Random Forest, XGBoost, or future model types.
-- ============================================================


-- ============================================================
-- 1. DATA QUALITY MONITORING
--
-- Source:
--   provider-level TEST dataset
--
-- Target:
--   fraud_monitoring.model_data_quality
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  dataset_name STRING,

  total_rows INT64,
  unique_providers INT64,
  duplicate_rows INT64,

  null_feature_values INT64,
  invalid_feature_values INT64,

  monitoring_status STRING
);


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`

WITH test_data AS (

  SELECT
    *
  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE
    dataset_split = 'TEST'

),

dq AS (

  SELECT
    COUNT(*) AS total_rows,

    COUNT(
      DISTINCT provider_id
    ) AS unique_providers,

    COUNT(*)
      - COUNT(
          DISTINCT provider_id
        )
      AS duplicate_rows,

    (
      COUNTIF(
        high_acuity_pct IS NULL
      )
      + COUNTIF(
          pct_99214 IS NULL
        )
      + COUNTIF(
          pct_99215 IS NULL
        )
      + COUNTIF(
          referral_rate IS NULL
        )
      + COUNTIF(
          eye_procedure_pct IS NULL
        )
      + COUNTIF(
          avg_paid_per_claim IS NULL
        )
      + COUNTIF(
          high_acuity_oe_ratio IS NULL
        )
      + COUNTIF(
          avg_member_provider_distance IS NULL
        )
    ) AS null_feature_values,

    (
      COUNTIF(
        high_acuity_pct < 0
        OR high_acuity_pct > 1
      )

      + COUNTIF(
          pct_99214 < 0
          OR pct_99214 > 1
        )

      + COUNTIF(
          pct_99215 < 0
          OR pct_99215 > 1
        )

      + COUNTIF(
          referral_rate < 0
          OR referral_rate > 1
        )

      + COUNTIF(
          eye_procedure_pct < 0
          OR eye_procedure_pct > 1
        )

      + COUNTIF(
          avg_paid_per_claim < 0
        )

      + COUNTIF(
          high_acuity_oe_ratio < 0
        )

      + COUNTIF(
          avg_member_provider_distance < 0
        )
    ) AS invalid_feature_values

  FROM
    test_data
)

SELECT
  CURRENT_TIMESTAMP()
    AS monitoring_timestamp,

  'provider_fraud_model'
    AS model_name,

  'TEST'
    AS dataset_name,

  total_rows,
  unique_providers,
  duplicate_rows,
  null_feature_values,
  invalid_feature_values,

  CASE
    WHEN total_rows = 0
      THEN 'ALERT'

    WHEN duplicate_rows > 0
      THEN 'ALERT'

    WHEN null_feature_values > 0
      THEN 'ALERT'

    WHEN invalid_feature_values > 0
      THEN 'ALERT'

    ELSE 'PASS'
  END AS monitoring_status

FROM
  dq;



-- ============================================================
-- 2. FEATURE DRIFT MONITORING
--
-- TRAIN = baseline
-- TEST  = current
--
-- Target:
--   fraud_monitoring.model_feature_drift
--
-- Uses standardized mean shift:
--
--   |current mean - baseline mean|
--   --------------------------------
--            baseline std
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  feature_name STRING,

  baseline_mean FLOAT64,
  current_mean FLOAT64,
  baseline_std FLOAT64,

  standardized_mean_shift FLOAT64,

  monitoring_status STRING
);


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`

WITH feature_values AS (

  SELECT
    dataset_split,
    feature_name,
    feature_value

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  UNPIVOT (
    feature_value
    FOR feature_name IN (
      high_acuity_pct,
      pct_99214,
      pct_99215,
      referral_rate,
      eye_procedure_pct,
      avg_paid_per_claim,
      high_acuity_oe_ratio,
      avg_member_provider_distance
    )
  )

),

baseline AS (

  SELECT
    feature_name,

    AVG(
      feature_value
    ) AS baseline_mean,

    STDDEV(
      feature_value
    ) AS baseline_std

  FROM
    feature_values

  WHERE
    dataset_split = 'TRAIN'

  GROUP BY
    feature_name

),

current_data AS (

  SELECT
    feature_name,

    AVG(
      feature_value
    ) AS current_mean

  FROM
    feature_values

  WHERE
    dataset_split = 'TEST'

  GROUP BY
    feature_name

),

drift AS (

  SELECT
    b.feature_name,
    b.baseline_mean,
    c.current_mean,
    b.baseline_std,

    SAFE_DIVIDE(
      ABS(
        c.current_mean
        - b.baseline_mean
      ),
      NULLIF(
        b.baseline_std,
        0
      )
    ) AS standardized_mean_shift

  FROM
    baseline b

  JOIN
    current_data c

  USING (
    feature_name
  )

)

SELECT
  CURRENT_TIMESTAMP()
    AS monitoring_timestamp,

  'provider_fraud_model'
    AS model_name,

  feature_name,

  baseline_mean,
  current_mean,
  baseline_std,

  standardized_mean_shift,

  CASE
    WHEN standardized_mean_shift >= 1.0
      THEN 'ALERT'

    ELSE 'PASS'
  END AS monitoring_status

FROM
  drift;



-- ============================================================
-- 3. PREDICTION DRIFT MONITORING
--
-- Assumes scored provider predictions are stored in:
--
--   fraud_experiments.provider_predictions
--
-- Expected fields:
--   prediction
--
-- Target:
--   fraud_monitoring.model_prediction_drift
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
(
  check_date DATE,
  model_name STRING,

  total_predictions INT64,
  positive_count INT64,
  negative_count INT64,

  positive_rate FLOAT64,
  baseline_rate FLOAT64,
  rate_change FLOAT64,

  status STRING
);


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`

WITH prediction_metrics AS (

  SELECT
    COUNT(*)
      AS total_predictions,

    COUNTIF(
      CAST(prediction AS STRING)
      = '1'
    ) AS positive_count,

    COUNTIF(
      CAST(prediction AS STRING)
      = '0'
    ) AS negative_count,

    SAFE_DIVIDE(
      COUNTIF(
        CAST(prediction AS STRING)
        = '1'
      ),
      COUNT(*)
    ) AS positive_rate

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`

),

drift AS (

  SELECT
    total_predictions,
    positive_count,
    negative_count,
    positive_rate,

    0.10
      AS baseline_rate,

    ABS(
      positive_rate
      - 0.10
    ) AS rate_change

  FROM
    prediction_metrics

)

SELECT
  CURRENT_DATE()
    AS check_date,

  'provider_fraud_model'
    AS model_name,

  total_predictions,
  positive_count,
  negative_count,
  positive_rate,
  baseline_rate,
  rate_change,

  CASE
    WHEN total_predictions = 0
      THEN 'ALERT'

    WHEN rate_change >= 0.10
      THEN 'ALERT'

    ELSE 'PASS'
  END AS status

FROM
  drift;



-- ============================================================
-- 4. QA: LATEST DATA QUALITY RESULT
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`

ORDER BY
  monitoring_timestamp DESC

LIMIT 10;



-- ============================================================
-- 5. QA: LATEST FEATURE DRIFT RUN
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`

WHERE
  monitoring_timestamp = (
    SELECT
      MAX(
        monitoring_timestamp
      )
    FROM
      `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`
  )

ORDER BY
  standardized_mean_shift DESC;



-- ============================================================
-- 6. QA: LATEST PREDICTION DRIFT RESULT
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`

ORDER BY
  check_date DESC

LIMIT 10;
