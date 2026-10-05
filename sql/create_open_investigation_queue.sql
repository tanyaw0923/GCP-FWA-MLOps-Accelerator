
--exclude providers have been investigated already
CREATE OR REPLACE VIEW
`fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue` AS

SELECT q.*
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue` q

LEFT JOIN (
  SELECT DISTINCT provider_id
  FROM
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
) f
USING(provider_id)

WHERE f.provider_id IS NULL;

--check open investigation queue
SELECT
  COUNT(*) AS open_investigation_count
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue`;


--inspect the cases
SELECT *
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue`
ORDER BY risk_rank;


--simulate investigation results
INSERT INTO
`fwa-mlops-accelerator-demo.fraud_feedback.investigation_results` (
  investigation_id,
  provider_id,
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
    fraud_score,
    ROW_NUMBER() OVER (ORDER BY risk_rank) AS rn
  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.open_investigation_queue`
  ORDER BY risk_rank
  LIMIT 10
)

SELECT
  GENERATE_UUID() AS investigation_id,
  provider_id,
  'isolation_forest_vertex_batch' AS model_version,
  fraud_score,

  CASE
    WHEN rn <= 3 THEN 'CONFIRMED_FRAUD'
    ELSE 'CONFIRMED_NORMAL'
  END AS investigation_result,

  CASE
    WHEN rn <= 3 THEN TRUE
    ELSE FALSE
  END AS confirmed_fraud,

  CASE
    WHEN rn <= 3 THEN CAST(ROUND(5000 + RAND() * 20000, 2) AS NUMERIC)
    ELSE CAST(0 AS NUMERIC)
  END AS fraud_amount,

  CURRENT_DATE() AS investigation_date,
  'synthetic_investigation_team' AS investigator,
  CURRENT_TIMESTAMP() AS created_at

FROM cases;

--check investigation results
SELECT
  investigation_result,
  COUNT(*) AS provider_count,
  SUM(fraud_amount) AS total_fraud_amount
FROM
  `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
GROUP BY investigation_result;


--add run ID to investigation results table
ALTER TABLE
`fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
ADD COLUMN IF NOT EXISTS run_id STRING;
