CREATE OR REPLACE TABLE
`fwa-mlops-accelerator-demo.fraud_features.random_forest_retraining_dataset` AS

WITH latest_feedback AS (
  SELECT *
  FROM (
    SELECT
      provider_id,
      confirmed_fraud,
      investigation_result,
      investigator,
      investigation_date,
      run_id,
      created_at,

      ROW_NUMBER() OVER (
        PARTITION BY provider_id
        ORDER BY investigation_date DESC, created_at DESC
      ) AS rn

    FROM
      `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`

    WHERE confirmed_fraud IS NOT NULL
  )

  WHERE rn = 1
)

SELECT
  f.provider_id,

  f.high_acuity_pct,
  f.pct_99214,
  f.pct_99215,
  f.referral_rate,
  f.eye_procedure_pct,
  f.avg_paid_per_claim,
  f.high_acuity_oe_ratio,

  CAST(feedback.confirmed_fraud AS INT64) AS fraud_label,

  feedback.investigation_result,
  feedback.investigator,
  feedback.investigation_date,
  feedback.run_id AS source_run_id

FROM
  `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features` f

JOIN latest_feedback feedback
USING(provider_id)

WHERE
  f.dataset_split = 'TEST';


--check re-train dataset label distribution
SELECT
  fraud_label,
  COUNT(*) AS provider_count
FROM
  `fwa-mlops-accelerator-demo.fraud_features.random_forest_retraining_dataset`
GROUP BY fraud_label
ORDER BY fraud_label;
