-- ============================================================
-- CURRENT SUPERVISED MODEL COMPARISON
--
-- Random Forest and XGBoost are directly comparable because they
-- use the same point-in-time provider feature source, labels, and
-- test-window evaluation contract.
-- ============================================================

WITH latest_runs AS (
  SELECT
    model_type,
    run_id,
    created_at

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs`

  WHERE
    model_type IN (
      'xgboost',
      'random_forest'
    )
    AND status = 'SUCCEEDED'

  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_type
    ORDER BY created_at DESC
  ) = 1
),

metrics AS (
  SELECT
    r.model_type,
    r.run_id,
    m.metric_name,
    m.metric_value

  FROM latest_runs r

  JOIN
    `fwa-mlops-accelerator-demo.fraud_experiments.experiment_metrics` m
    USING(run_id)

  WHERE m.metric_name IN (
    'pr_auc',
    'roc_auc',
    'precision',
    'recall',
    'f1_score'
  )
)

SELECT
  metric_name,

  ROUND(
    MAX(
      IF(
        model_type = 'xgboost',
        metric_value,
        NULL
      )
    ),
    4
  ) AS xgboost,

  ROUND(
    MAX(
      IF(
        model_type = 'random_forest',
        metric_value,
        NULL
      )
    ),
    4
  ) AS random_forest

FROM metrics

GROUP BY metric_name

ORDER BY
  CASE metric_name
    WHEN 'pr_auc' THEN 1
    WHEN 'roc_auc' THEN 2
    WHEN 'precision' THEN 3
    WHEN 'recall' THEN 4
    WHEN 'f1_score' THEN 5
    ELSE 99
  END;


-- ============================================================
-- RANKING OVERLAP FOR LATEST SCORED RUNS
-- ============================================================

WITH latest_scored_runs AS (
  SELECT
    model_name,
    model_version,
    run_id,
    MAX(created_at) AS created_at

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`

  WHERE model_name IN (
    'xgboost',
    'random_forest'
  )

  GROUP BY
    model_name,
    model_version,
    run_id

  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_name
    ORDER BY created_at DESC
  ) = 1
),

top20 AS (
  SELECT
    p.model_name,
    p.provider_id,
    p.fraud_score,
    p.risk_rank

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

  JOIN latest_scored_runs r
    ON p.model_name = r.model_name
    AND p.model_version = r.model_version
    AND p.run_id = r.run_id

  WHERE p.risk_rank <= 20
),

rf AS (
  SELECT provider_id
  FROM top20
  WHERE model_name = 'random_forest'
),

xgb AS (
  SELECT provider_id
  FROM top20
  WHERE model_name = 'xgboost'
)

SELECT
  20 AS investigation_capacity,

  COUNT(*) AS common_providers,

  SAFE_DIVIDE(
    COUNT(*),
    20
  ) AS top20_overlap_rate

FROM rf
JOIN xgb
  USING(provider_id);
