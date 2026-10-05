-- Model Comparison
-- Compare the latest runs of Isolation Forest,
-- XGBoost and Random Forest.


-- Pull out latest run for each model

-- Compare supervised model metrics
-- XGBoost and Random Forest can be compared directly because
-- they use the same labeled population and evaluation method.


WITH latest_runs AS (
  SELECT
    r.model_type,
    r.run_id,
    r.created_at
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs` r
  WHERE r.model_type IN ('xgboost', 'random_forest')
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY r.model_type
    ORDER BY r.created_at DESC
  ) = 1
)

SELECT
  r.model_type,
  m.metric_name,
  ROUND(m.metric_value, 4) AS metric_value
FROM latest_runs r
JOIN `fwa-mlops-accelerator-demo.fraud_experiments.experiment_metrics` m
  ON r.run_id = m.run_id
WHERE m.metric_name IN (
  'cv_pr_auc',
  'cv_roc_auc',
  'cv_precision',
  'cv_recall',
  'fraud_prevalence',
  'labeled_provider_count'
)
ORDER BY
  m.metric_name,
  r.model_type;

-- Pivot supervised metrics for easier comparison

WITH latest_runs AS (
  SELECT
    model_type,
    run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs`
  WHERE model_type IN ('xgboost', 'random_forest')
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_type
    ORDER BY created_at DESC
  ) = 1
),

metrics AS (
  SELECT
    r.model_type,
    m.metric_name,
    m.metric_value
  FROM latest_runs r
  JOIN `fwa-mlops-accelerator-demo.fraud_experiments.experiment_metrics` m
    ON r.run_id = m.run_id
)

SELECT
  metric_name,
  ROUND(MAX(IF(model_type = 'xgboost', metric_value, NULL)), 4)
    AS xgboost,
  ROUND(MAX(IF(model_type = 'random_forest', metric_value, NULL)), 4)
    AS random_forest
FROM metrics
WHERE metric_name IN (
  'cv_pr_auc',
  'cv_roc_auc',
  'cv_precision',
  'cv_recall'
)
GROUP BY metric_name
ORDER BY metric_name;


-- Compare risk-category distributions across all 3 models

WITH latest_runs AS (
  SELECT
    model_name,
    run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name IN (
    'isolation_forest',
    'xgboost',
    'random_forest'
  )
  GROUP BY model_name, run_id
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_name
    ORDER BY MAX(created_at) DESC
  ) = 1
)

SELECT
  p.model_name,
  p.risk_category,
  COUNT(*) AS provider_count,
  ROUND(
    100 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY p.model_name),
    2
  ) AS provider_pct
FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p
JOIN latest_runs r
  ON p.model_name = r.model_name
 AND p.run_id = r.run_id
GROUP BY
  p.model_name,
  p.risk_category
ORDER BY
  p.model_name,
  p.risk_category;


-- Compare Top 20 provider rankings across models

WITH latest_runs AS (
  SELECT
    model_name,
    run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name IN (
    'isolation_forest',
    'xgboost',
    'random_forest'
  )
  GROUP BY model_name, run_id
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_name
    ORDER BY MAX(created_at) DESC
  ) = 1
),

latest_predictions AS (
  SELECT
    p.model_name,
    p.provider_id,
    p.fraud_score,
    p.risk_rank,
    p.risk_category
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p
  JOIN latest_runs r
    ON p.model_name = r.model_name
   AND p.run_id = r.run_id
)

SELECT
  provider_id,

  MAX(IF(
    model_name = 'isolation_forest',
    risk_rank,
    NULL
  )) AS isolation_forest_rank,

  MAX(IF(
    model_name = 'xgboost',
    risk_rank,
    NULL
  )) AS xgboost_rank,

  MAX(IF(
    model_name = 'random_forest',
    risk_rank,
    NULL
  )) AS random_forest_rank,

  MAX(IF(
    model_name = 'xgboost',
    fraud_score,
    NULL
  )) AS xgboost_score,

  MAX(IF(
    model_name = 'random_forest',
    fraud_score,
    NULL
  )) AS random_forest_score

FROM latest_predictions

GROUP BY provider_id

HAVING
  xgboost_rank <= 20
  OR random_forest_rank <= 20
  OR isolation_forest_rank <= 20

ORDER BY
  LEAST(
    COALESCE(isolation_forest_rank, 9999),
    COALESCE(xgboost_rank, 9999),
    COALESCE(random_forest_rank, 9999)
  );


-- Top-20 overlap between XGBoost and Random Forest

WITH latest_runs AS (
  SELECT
    model_name,
    run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name IN ('xgboost', 'random_forest')
  GROUP BY model_name, run_id
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_name
    ORDER BY MAX(created_at) DESC
  ) = 1
),

top20 AS (
  SELECT
    p.model_name,
    p.provider_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p
  JOIN latest_runs r
    ON p.model_name = r.model_name
   AND p.run_id = r.run_id
  WHERE p.risk_rank <= 20
),

xgb AS (
  SELECT provider_id
  FROM top20
  WHERE model_name = 'xgboost'
),

rf AS (
  SELECT provider_id
  FROM top20
  WHERE model_name = 'random_forest'
)

SELECT
  COUNT(*) AS common_top_20_providers
FROM xgb
JOIN rf USING (provider_id);


-- Compare model scores for confirmed fraud providers

WITH latest_runs AS (
  SELECT
    model_name,
    run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name IN ('xgboost', 'random_forest')
  GROUP BY model_name, run_id
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_name
    ORDER BY MAX(created_at) DESC
  ) = 1
),

confirmed_fraud AS (
  SELECT DISTINCT provider_id
  FROM `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
  WHERE confirmed_fraud = TRUE
)

SELECT
  p.model_name,
  COUNT(*) AS confirmed_fraud_scored,
  ROUND(AVG(p.fraud_score), 4) AS avg_fraud_score,
  COUNTIF(p.risk_category = 'HIGH') AS high_risk_count
FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p
JOIN latest_runs r
  ON p.model_name = r.model_name
 AND p.run_id = r.run_id
JOIN confirmed_fraud f
  USING (provider_id)
GROUP BY p.model_name
ORDER BY p.model_name;
