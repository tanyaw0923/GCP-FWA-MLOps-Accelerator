#no data returned->in this synthetic run, XGBoost did not identify any providers as HIGH risk that were outside the original Isolation Forest top-50 queue.
WITH latest_if_run AS (
  SELECT run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name = 'isolation_forest'
  GROUP BY run_id
  ORDER BY MAX(created_at) DESC
  LIMIT 1
),

latest_xgb_run AS (
  SELECT run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name = 'xgboost'
  GROUP BY run_id
  ORDER BY MAX(created_at) DESC
  LIMIT 1
),

if_predictions AS (
  SELECT
    provider_id,
    fraud_score AS isolation_forest_score,
    risk_rank AS isolation_forest_rank,
    risk_category AS isolation_forest_risk
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name = 'isolation_forest'
    AND run_id = (SELECT run_id FROM latest_if_run)
),

xgb_predictions AS (
  SELECT
    provider_id,
    fraud_score AS xgboost_score,
    risk_rank AS xgboost_rank,
    risk_category AS xgboost_risk
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name = 'xgboost'
    AND run_id = (SELECT run_id FROM latest_xgb_run)
)

SELECT
  i.provider_id,

  i.isolation_forest_score,
  i.isolation_forest_rank,
  i.isolation_forest_risk,

  x.xgboost_score,
  x.xgboost_rank,
  x.xgboost_risk

FROM if_predictions i
JOIN xgb_predictions x
  USING (provider_id)

WHERE
  i.isolation_forest_rank > 50
  AND x.xgboost_risk = 'HIGH'

ORDER BY
  x.xgboost_score DESC;
