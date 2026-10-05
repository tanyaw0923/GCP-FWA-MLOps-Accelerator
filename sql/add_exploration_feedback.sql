#find the latest_run_id from the model
DECLARE latest_run_id STRING;

#clean up from the previous debugging
DELETE FROM `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
WHERE investigator = 'synthetic_exploration_team';

SET latest_run_id = (
  SELECT run_id
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
  WHERE model_name = 'isolation_forest'
  GROUP BY run_id
  ORDER BY MAX(created_at) DESC
  LIMIT 1
);
-- select providers already investigated
CREATE TEMP TABLE already_investigated AS
SELECT DISTINCT provider_id
FROM `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`;


-- Exploration candidates

CREATE TEMP TABLE candidates AS
SELECT
  p.provider_id,
  p.model_version,
  p.fraud_score,
  p.risk_rank,
  p.risk_category
FROM `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p
LEFT JOIN already_investigated i USING (provider_id)
WHERE p.model_name = 'isolation_forest'
  AND p.run_id = latest_run_id
  AND i.provider_id IS NULL;



-- Stratified exploration sample from LOW cateogry
CREATE TEMP TABLE low_exploration_sample AS
SELECT
  *,
  ROW_NUMBER() OVER (ORDER BY RAND()) AS exploration_random_rank
FROM (
  SELECT *
  FROM candidates
  WHERE risk_category = 'LOW'
  ORDER BY RAND()
  LIMIT 30
);
-- Randomly rank LOW-risk exploration providers
--
-- 3 LOW-risk providers will be simulate as confirmed fraud to avoid selection bias for supervised training
CREATE TEMP TABLE exploration_feedback AS
SELECT
  GENERATE_UUID() AS investigation_id,
  provider_id,
  model_version,
  fraud_score,
  CASE
    WHEN exploration_random_rank <= 3 THEN 'CONFIRMED_FRAUD'
    ELSE 'CONFIRMED_NORMAL'
  END AS investigation_result,
  exploration_random_rank <= 3 AS confirmed_fraud,
  CASE
    WHEN exploration_random_rank <= 3 THEN
      CAST(
        10000 + MOD(ABS(FARM_FINGERPRINT(provider_id)), 140001)
        AS NUMERIC
      )
    ELSE CAST(0 AS NUMERIC)
  END AS fraud_amount,
  DATE('2026-10-20') AS investigation_date,
  'synthetic_exploration_team' AS investigator,
  CURRENT_TIMESTAMP() AS created_at
FROM low_exploration_sample;
INSERT INTO `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results` (
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
SELECT
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
FROM exploration_feedback;

SELECT
  confirmed_fraud,
  COUNT(*) AS provider_count
FROM exploration_feedback
GROUP BY confirmed_fraud
ORDER BY confirmed_fraud;

