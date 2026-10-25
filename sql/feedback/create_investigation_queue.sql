-- ============================================================
-- MODEL-INDEPENDENT INVESTIGATION QUEUE
--
-- Select providers from the newest curated production-style
-- scoring run, regardless of whether the model is Random Forest
-- or XGBoost.
--
-- Investigation capacity is represented by the top 50 providers.
-- In production this would come from SIU staffing/case capacity.
-- ============================================================

CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_experiments.investigation_queue`
AS

WITH latest_run AS (
  SELECT
    model_name,
    model_version,
    run_id,
    MAX(created_at) AS latest_prediction_timestamp

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`

  WHERE
    run_id IS NOT NULL
    AND model_name IS NOT NULL
    AND model_version IS NOT NULL

  GROUP BY
    model_name,
    model_version,
    run_id

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
