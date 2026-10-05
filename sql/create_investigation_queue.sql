--only send suspicious providers highlighted by isolation forest to investigation queue
CREATE OR REPLACE TABLE
`fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue` AS

WITH latest_if_run AS (
  SELECT run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.isolation_forest_provider_predictions`
  ORDER BY created_at DESC
  LIMIT 1
)

SELECT
  provider_id,
  experiment_id,
  run_id,
  model_name,
  experiment_name,
  fraud_score,
  risk_rank,
  risk_category,
  reason,
  reason_1,
  reason_2,
  reason_3,
  'P1_ISOLATION_FOREST' AS investigation_priority,
  CURRENT_TIMESTAMP() AS queue_created_at

FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.isolation_forest_provider_predictions`

WHERE
  run_id = (SELECT run_id FROM latest_if_run)
  AND risk_category = 'HIGH';

  --qa queue results
  SELECT
  COUNT(*) AS investigation_count
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`;


  --inspect queue results
  SELECT *
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`
ORDER BY risk_rank;

