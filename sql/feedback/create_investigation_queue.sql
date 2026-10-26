-- ============================================================
-- PRODUCTION INVESTIGATION QUEUE
--
-- Build the queue from the model that is actually approved in
-- fraud_monitoring.model_training_state.
--
-- This prevents a newer challenger / experiment run from
-- accidentally becoming the investigation source.
--
-- The approved production identity is:
--   approved_model_type
--   approved_run_id
--   approved_model_resource (when available)
--
-- Investigation capacity is represented by the top 50 providers.
-- ============================================================

CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`
AS

WITH production_state AS (
  SELECT
    approved_model_type AS model_name,
    approved_run_id AS run_id,
    approved_model_resource,

    CASE
      WHEN approved_model_resource IS NOT NULL
        THEN CONCAT(
          approved_model_type,
          ':',
          REGEXP_EXTRACT(
            approved_model_resource,
            r'([^/]+)$'
          )
        )
      ELSE NULL
    END AS expected_model_version

  FROM
    `fwa-mlops-accelerator-demo.fraud_monitoring.model_training_state`

  WHERE
    model_name = 'provider_fraud_model'

  LIMIT 1
),

resolved_production_version AS (
  SELECT
    p.model_name,
    p.model_version,
    p.run_id,
    MAX(p.created_at) AS latest_prediction_timestamp

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

  CROSS JOIN
    production_state s

  WHERE
    p.model_name = s.model_name
    AND p.run_id = s.run_id

    AND (
      s.expected_model_version IS NULL
      OR p.model_version = s.expected_model_version
    )

  GROUP BY
    p.model_name,
    p.model_version,
    p.run_id

  QUALIFY ROW_NUMBER() OVER (
    ORDER BY latest_prediction_timestamp DESC
  ) = 1
),

ranked AS (
  SELECT
    p.provider_id,
    p.experiment_id,
    p.run_id,
    p.model_name,
    p.model_version,
    p.fraud_score,
    p.risk_rank,
    p.risk_category,
    p.reason,

    ROW_NUMBER() OVER (
      ORDER BY
        p.fraud_score DESC,
        p.provider_id
    ) AS investigation_rank

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

  INNER JOIN
    resolved_production_version r

    ON p.run_id = r.run_id
    AND p.model_name = r.model_name
    AND p.model_version = r.model_version
)

SELECT
  provider_id,
  experiment_id,
  run_id,
  model_name,
  model_version,
  fraud_score,
  risk_rank,
  risk_category,
  reason,
  investigation_rank,

  CASE
    WHEN investigation_rank <= 10
      THEN 'P1'
    WHEN investigation_rank <= 25
      THEN 'P2'
    ELSE 'P3'
  END AS investigation_priority,

  CURRENT_TIMESTAMP()
    AS queue_created_at

FROM ranked

WHERE
  investigation_rank <= 50

ORDER BY
  investigation_rank;


-- QA
SELECT
  model_name,
  model_version,
  run_id,
  COUNT(*) AS investigation_count,
  MIN(fraud_score) AS lowest_queued_score,
  MAX(fraud_score) AS highest_queued_score

FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`

GROUP BY
  model_name,
  model_version,
  run_id;
