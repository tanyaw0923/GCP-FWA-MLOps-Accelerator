--for retraining two conditions have to meet:
-- Retrain Random Forest when either:
-- 1. at least 20 new confirmed investigator labels are available
-- OR
-- 2. monitoring status is WARNING / FAIL

--create a retraining-status table
CREATE TABLE IF NOT EXISTS
`fwa-mlops-accelerator-demo.fraud_monitoring.random_forest_retraining_status` (
  check_timestamp TIMESTAMP,
  labeled_provider_count INT64,
  confirmed_fraud_count INT64,
  confirmed_normal_count INT64,
  new_label_threshold INT64,
  drift_trigger BOOL,
  retraining_required BOOL,
  trigger_reason STRING
);

--evaluate the current status
INSERT INTO
`fwa-mlops-accelerator-demo.fraud_monitoring.random_forest_retraining_status`

WITH labels AS (
  SELECT
    COUNT(*) AS labeled_provider_count,
    COUNTIF(fraud_label = 1) AS confirmed_fraud_count,
    COUNTIF(fraud_label = 0) AS confirmed_normal_count
  FROM
    `fwa-mlops-accelerator-demo.fraud_features.random_forest_retraining_dataset`
),

feature_drift AS (
  SELECT
    COUNTIF(drift_status IN ('WARNING', 'FAIL')) > 0 AS feature_drift_trigger
  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.feature_drift_summary`
  WHERE model_name = 'random_forest'
),

prediction_drift AS (
  SELECT
    COUNTIF(monitoring_status IN ('WARNING', 'FAIL')) > 0 AS prediction_drift_trigger
  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
  WHERE model_name = 'random_forest'
)

SELECT
  CURRENT_TIMESTAMP() AS check_timestamp,
  l.labeled_provider_count,
  l.confirmed_fraud_count,
  l.confirmed_normal_count,

  20 AS new_label_threshold,

  (f.feature_drift_trigger OR p.prediction_drift_trigger)
    AS drift_trigger,

  CASE
    WHEN l.labeled_provider_count >= 20 THEN TRUE
    WHEN f.feature_drift_trigger THEN TRUE
    WHEN p.prediction_drift_trigger THEN TRUE
    ELSE FALSE
  END AS retraining_required,

  CASE
    WHEN l.labeled_provider_count >= 20
      THEN 'Enough confirmed investigator labels available'
    WHEN f.feature_drift_trigger
      THEN 'Feature drift detected'
    WHEN p.prediction_drift_trigger
      THEN 'Prediction drift detected'
    ELSE 'No retraining trigger'
  END AS trigger_reason

FROM labels l
CROSS JOIN feature_drift f
CROSS JOIN prediction_drift p;


--check the output
select *
from fwa-mlops-accelerator-demo.fraud_monitoring.random_forest_retraining_status
