
--recreate the open queue table for isolation forest model
CREATE OR REPLACE VIEW
`fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue` AS

SELECT q.*
FROM `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue` q
LEFT JOIN `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results` f
  ON q.provider_id = f.provider_id
  AND q.run_id = f.run_id
WHERE f.provider_id IS NULL;

SELECT COUNT(*) AS open_investigation_count
FROM `fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue`;

--simulate investigator outcomes for a subset of those open Isolation Forest cases.
INSERT INTO
`fwa-mlops-accelerator-demo.fraud_feedback.investigation_results` (
  investigation_id,
  provider_id,
  run_id,
  model_version,
  fraud_score,
  investigation_result,
  confirmed_fraud,
  fraud_amount,
  investigation_date,
  investigator,
  created_at
)

WITH cases AS (
  SELECT
    provider_id,
    run_id,
    fraud_score,
    risk_rank,
    ROW_NUMBER() OVER (ORDER BY risk_rank) AS rn
  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue`
  ORDER BY risk_rank
  LIMIT 10
)

SELECT
  GENERATE_UUID(),
  provider_id,
  run_id,
  'isolation_forest_vertex_batch',
  fraud_score,

  CASE
    WHEN rn <= 3 THEN 'CONFIRMED_FRAUD'
    ELSE 'CONFIRMED_NORMAL'
  END,

  rn <= 3,

  CASE
    WHEN rn <= 3
      THEN CAST(ROUND(5000 + RAND() * 20000, 2) AS NUMERIC)
    ELSE CAST(0 AS NUMERIC)
  END,

  CURRENT_DATE(),
  'synthetic_investigation_team',
  CURRENT_TIMESTAMP()

FROM cases;

--qa results
SELECT
  run_id,
  investigation_result,
  COUNT(*) AS provider_count,
  SUM(fraud_amount) AS total_fraud_amount
FROM
  `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
WHERE run_id IS NOT NULL
GROUP BY run_id, investigation_result
ORDER BY run_id, investigation_result;


--confirm content in open queue, now it decreases to 12
SELECT COUNT(*) AS remaining_open_cases
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue`;

--check fraud label distribution
SELECT
  fraud_label,
  COUNT(*) AS provider_count
FROM
  `fwa-mlops-accelerator-demo.fraud_features.random_forest_retraining_dataset`
GROUP BY fraud_label
ORDER BY fraud_label;
