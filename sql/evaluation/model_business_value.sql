-- ============================================================
-- FWA OPERATIONAL / BUSINESS VALUE EVALUATION
--
-- Model quality should reflect investigator capacity, not only a
-- global classification threshold.
--
-- This demo evaluates the latest scored run at K = 10, 20, 50.
--
-- Because this repository uses synthetic claims, recall@K and lift@K
-- use synthetic provider truth from the claims table.
--
-- Captured fraud dollars use completed investigation feedback and are
-- therefore only available for providers that were investigated.
-- ============================================================

WITH latest_run AS (
  SELECT
    model_name,
    model_version,
    run_id,
    MAX(created_at) AS created_at

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`

  GROUP BY
    model_name,
    model_version,
    run_id

  QUALIFY ROW_NUMBER() OVER (
    ORDER BY created_at DESC
  ) = 1
),

truth AS (
  SELECT
    CAST(provider_id AS STRING) AS provider_id,

    MAX(
      CAST(
        fraud_provider_label
        AS INT64
      )
    ) AS fraud_label

  FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

  GROUP BY provider_id
),

ranked AS (
  SELECT
    p.provider_id,
    p.model_name,
    p.model_version,
    p.run_id,
    p.fraud_score,

    ROW_NUMBER() OVER (
      ORDER BY
        p.fraud_score DESC,
        p.provider_id
    ) AS rank_position,

    t.fraud_label

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

  JOIN latest_run r
    ON p.model_name = r.model_name
    AND p.model_version = r.model_version
    AND p.run_id = r.run_id

  JOIN truth t
    USING(provider_id)
),

population AS (
  SELECT
    COUNT(*) AS total_providers,
    SUM(fraud_label) AS total_fraud_providers,
    AVG(fraud_label) AS fraud_prevalence

  FROM ranked
),

capacity AS (
  SELECT investigation_capacity
  FROM UNNEST([10, 20, 50]) AS investigation_capacity
),

rank_metrics AS (
  SELECT
    c.investigation_capacity,

    COUNTIF(
      r.rank_position
      <= c.investigation_capacity
    ) AS providers_selected,

    COUNTIF(
      r.rank_position
      <= c.investigation_capacity
      AND r.fraud_label = 1
    ) AS fraud_captured

  FROM capacity c

  CROSS JOIN ranked r

  GROUP BY c.investigation_capacity
),

feedback AS (
  SELECT
    provider_id,

    MAX(
      IF(
        confirmed_fraud,
        fraud_amount,
        0
      )
    ) AS confirmed_fraud_amount

  FROM
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`

  GROUP BY provider_id
),

dollar_metrics AS (
  SELECT
    c.investigation_capacity,

    SUM(
      IF(
        r.rank_position <= c.investigation_capacity,
        COALESCE(f.confirmed_fraud_amount, 0),
        0
      )
    ) AS confirmed_fraud_dollars_captured

  FROM capacity c

  CROSS JOIN ranked r

  LEFT JOIN feedback f
    USING(provider_id)

  GROUP BY c.investigation_capacity
)

SELECT
  r.model_name,
  r.model_version,
  r.run_id,

  m.investigation_capacity,
  m.providers_selected,
  m.fraud_captured,

  SAFE_DIVIDE(
    m.fraud_captured,
    m.providers_selected
  ) AS precision_at_k,

  SAFE_DIVIDE(
    m.fraud_captured,
    p.total_fraud_providers
  ) AS recall_at_k,

  SAFE_DIVIDE(
    SAFE_DIVIDE(
      m.fraud_captured,
      m.providers_selected
    ),
    p.fraud_prevalence
  ) AS lift_at_k,

  d.confirmed_fraud_dollars_captured,

  p.fraud_prevalence,
  p.total_fraud_providers,
  p.total_providers

FROM rank_metrics m

CROSS JOIN population p

CROSS JOIN latest_run r

JOIN dollar_metrics d
  USING(investigation_capacity)

ORDER BY
  m.investigation_capacity;
