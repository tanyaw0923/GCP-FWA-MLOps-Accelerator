-- ============================================================
-- MODEL-INDEPENDENT INVESTIGATION QUEUE
--
-- The queue is created from the latest curated provider prediction
-- run, regardless of whether the approved model is Random Forest
-- or XGBoost.
--
-- In production, investigation capacity should be configured from
-- SIU operations. The demo uses the top 20 providers.
-- ============================================================

CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`
AS

WITH latest_run AS (
  SELECT
    run_id,
    model_name,
    model_version

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`

  QUALIFY ROW_NUMBER() OVER (
    ORDER BY created_at DESC
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

  JOIN latest_run r
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
    WHEN investigation_rank <= 5
      THEN 'P1'
    WHEN investigation_rank <= 10
      THEN 'P2'
    ELSE 'P3'
  END AS investigation_priority,

  CURRENT_TIMESTAMP() AS queue_created_at

FROM ranked

WHERE investigation_rank <= 20

ORDER BY investigation_rank;


-- QA
SELECT
  model_name,
  model_version,
  COUNT(*) AS investigation_count,
  MIN(fraud_score) AS lowest_queued_score,
  MAX(fraud_score) AS highest_queued_score

FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`

GROUP BY
  model_name,
  model_version;
