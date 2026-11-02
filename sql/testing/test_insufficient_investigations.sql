-- Isolated BigQuery regression test for the insufficient-investigations case.
-- No persistent tables are created or modified.
-- Run: bq query --use_legacy_sql=false < sql/testing/test_insufficient_investigations.sql

DECLARE min_completed_cases INT64 DEFAULT 20;
DECLARE evaluated_cases INT64;
DECLARE actual_status STRING;

CREATE TEMP TABLE synthetic_claims AS
SELECT
  FORMAT('TEST_PROVIDER_%03d', id) AS provider_id,
  FORMAT('TEST_CLAIM_%03d', id) AS claim_id,
  DATE '2026-09-01' AS service_date,
  150.0 + id * 25.0 AS paid_amount
FROM UNNEST(GENERATE_ARRAY(1, 15)) AS id;

CREATE TEMP TABLE synthetic_predictions AS
SELECT
  provider_id,
  'random_forest' AS model_name,
  'random_forest:test-model-001' AS model_version,
  'insufficient-data-test-run' AS run_id,
  CAST(ROUND(0.05 + MOD(ABS(FARM_FINGERPRINT(provider_id)), 85) / 100.0, 2) AS FLOAT64) AS fraud_score
FROM synthetic_claims;

CREATE TEMP TABLE synthetic_feedback AS
SELECT
  provider_id,
  model_name,
  model_version,
  run_id,
  IF(MOD(CAST(SUBSTR(provider_id, -3) AS INT64), 5) = 0,
     'CONFIRMED_FRAUD', 'CONFIRMED_NORMAL') AS investigation_result,
  TRUE AS completed
FROM synthetic_predictions;

SET evaluated_cases = (
  SELECT COUNT(*)
  FROM synthetic_predictions p
  JOIN synthetic_feedback f
    ON p.provider_id = f.provider_id
   AND p.model_name = f.model_name
   AND p.model_version = f.model_version
   AND p.run_id = f.run_id
  WHERE f.completed
    AND f.investigation_result IN ('CONFIRMED_FRAUD', 'CONFIRMED_NORMAL')
);

SET actual_status = IF(
  evaluated_cases < min_completed_cases,
  'INSUFFICIENT_DATA',
  'EVALUATION_READY'
);

ASSERT (SELECT COUNT(*) FROM synthetic_claims) = 15
  AS 'Expected 15 synthetic claims';
ASSERT evaluated_cases = 15
  AS 'Expected exactly 15 completed and matched investigations';
ASSERT actual_status = 'INSUFFICIENT_DATA'
  AS 'Expected the insufficient-data branch';

SELECT
  evaluated_cases AS completed_investigations,
  min_completed_cases AS minimum_required,
  actual_status AS monitoring_status,
  'Not enough completed investigation outcomes' AS alert_reason,
  COUNTIF(investigation_result = 'CONFIRMED_FRAUD') AS confirmed_fraud_cases,
  COUNTIF(investigation_result = 'CONFIRMED_NORMAL') AS confirmed_normal_cases
FROM synthetic_feedback;
