-- ============================================================
-- GENERIC MODEL RETRAINING TRIGGER CHECK
-- FWA MLOps Accelerator
--
-- Shared across Random Forest, XGBoost, and future model types.
--
-- Retraining can be triggered by:
--   1. enough new confirmed investigation labels
--   2. data-quality alert
--   3. feature-drift alert
--   4. prediction-drift alert
--
-- Note:
-- The pipeline itself also checks:
--   - new monthly data
--   - feature signature changes
--   - model configuration changes
--
-- This SQL is mainly a monitoring / demo summary table.
-- ============================================================


-- ============================================================
-- 1. CREATE GENERIC RETRAINING STATUS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_retraining_status`
(
  check_timestamp TIMESTAMP,

  model_name STRING,

  labeled_provider_count INT64,
  confirmed_fraud_count INT64,
  confirmed_normal_count INT64,

  new_label_threshold INT64,

  data_quality_trigger BOOL,
  feature_drift_trigger BOOL,
  prediction_drift_trigger BOOL,

  monitoring_trigger BOOL,

  retraining_required BOOL,

  trigger_reason STRING
);


-- ============================================================
-- 2. EVALUATE CURRENT RETRAINING STATUS
-- ============================================================

INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_retraining_status`

WITH labels AS (

  SELECT
    COUNT(*) AS labeled_provider_count,

    COUNTIF(
      fraud_label = 1
    ) AS confirmed_fraud_count,

    COUNTIF(
      fraud_label = 0
    ) AS confirmed_normal_count

  FROM
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`

  WHERE
    fraud_label IS NOT NULL

),


latest_data_quality AS (

  SELECT
    monitoring_status

  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`

  WHERE
    model_name = 'provider_fraud_model'

  ORDER BY
    monitoring_timestamp DESC

  LIMIT 1

),


latest_feature_drift_run AS (

  SELECT
    MAX(
      monitoring_timestamp
    ) AS monitoring_timestamp

  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`

  WHERE
    model_name = 'provider_fraud_model'

),


feature_drift AS (

  SELECT
    COUNTIF(
      monitoring_status = 'ALERT'
    ) > 0 AS feature_drift_trigger

  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`

  WHERE
    model_name = 'provider_fraud_model'

    AND monitoring_timestamp = (
      SELECT
        monitoring_timestamp
      FROM
        latest_feature_drift_run
    )

),


latest_prediction_drift AS (

  SELECT
    status

  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`

  WHERE
    model_name = 'provider_fraud_model'

  ORDER BY
    check_date DESC

  LIMIT 1

),


trigger_summary AS (

  SELECT
    l.labeled_provider_count,
    l.confirmed_fraud_count,
    l.confirmed_normal_count,

    20 AS new_label_threshold,

    COALESCE(
      dq.monitoring_status = 'ALERT',
      FALSE
    ) AS data_quality_trigger,

    COALESCE(
      fd.feature_drift_trigger,
      FALSE
    ) AS feature_drift_trigger,

    COALESCE(
      pd.status = 'ALERT',
      FALSE
    ) AS prediction_drift_trigger

  FROM
    labels l

  LEFT JOIN
    latest_data_quality dq
  ON TRUE

  LEFT JOIN
    feature_drift fd
  ON TRUE

  LEFT JOIN
    latest_prediction_drift pd
  ON TRUE

)


SELECT
  CURRENT_TIMESTAMP()
    AS check_timestamp,

  'provider_fraud_model'
    AS model_name,

  labeled_provider_count,
  confirmed_fraud_count,
  confirmed_normal_count,

  new_label_threshold,

  data_quality_trigger,
  feature_drift_trigger,
  prediction_drift_trigger,

  (
    data_quality_trigger
    OR feature_drift_trigger
    OR prediction_drift_trigger
  ) AS monitoring_trigger,

  (
    labeled_provider_count
      >= new_label_threshold

    OR data_quality_trigger

    OR feature_drift_trigger

    OR prediction_drift_trigger
  ) AS retraining_required,

  CASE

    WHEN
      labeled_provider_count
      >= new_label_threshold
      AND data_quality_trigger
      AND feature_drift_trigger
      AND prediction_drift_trigger
    THEN
      'New labels + data quality + feature drift + prediction drift'

    WHEN
      labeled_provider_count
      >= new_label_threshold
      AND (
        data_quality_trigger
        OR feature_drift_trigger
        OR prediction_drift_trigger
      )
    THEN
      'New labels + monitoring alert'

    WHEN
      labeled_provider_count
      >= new_label_threshold
    THEN
      'Enough confirmed investigation labels available'

    WHEN
      data_quality_trigger
      AND feature_drift_trigger
      AND prediction_drift_trigger
    THEN
      'Data quality + feature drift + prediction drift alerts'

    WHEN
      data_quality_trigger
      AND feature_drift_trigger
    THEN
      'Data quality + feature drift alerts'

    WHEN
      data_quality_trigger
      AND prediction_drift_trigger
    THEN
      'Data quality + prediction drift alerts'

    WHEN
      feature_drift_trigger
      AND prediction_drift_trigger
    THEN
      'Feature drift + prediction drift alerts'

    WHEN
      data_quality_trigger
    THEN
      'Data quality alert'

    WHEN
      feature_drift_trigger
    THEN
      'Feature drift alert'

    WHEN
      prediction_drift_trigger
    THEN
      'Prediction drift alert'

    ELSE
      'No retraining trigger'

  END AS trigger_reason

FROM
  trigger_summary;



-- ============================================================
-- 3. QA: CHECK LATEST RETRAINING STATUS
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_retraining_status`

ORDER BY
  check_timestamp DESC

LIMIT 10;
