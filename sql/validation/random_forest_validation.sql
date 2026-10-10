-- Random Forest experiment metadata, confirm model runs successfully

SELECT
  experiment_id,
  run_id,
  model_type,
  feature_version,
  dataset_version,
  status,
  created_at
FROM `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs`
WHERE model_type = 'random_forest'
ORDER BY created_at DESC;


-- Random Forest metrics

SELECT
  run_id,
  metric_name,
  metric_value
FROM `fwa-mlops-accelerator-demo.fraud_experiments.experiment_metrics`
WHERE experiment_id = 'rf_provider_eye_fraud'
ORDER BY created_at DESC, metric_name;


-- Latest Random Forest top providers

WITH latest_run AS (
  SELECT run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name = 'random_forest'
  GROUP BY run_id
  ORDER BY MAX(created_at) DESC
  LIMIT 1
)

SELECT
  provider_id,
  fraud_score,
  risk_rank,
  risk_category,
  reason_1,
  reason_2,
  reason_3
FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
WHERE model_name = 'random_forest'
  AND run_id = (SELECT run_id FROM latest_run)
ORDER BY risk_rank
LIMIT 20;
