--create random forest provider prediction table to store all the prediction results
CREATE TABLE IF NOT EXISTS
`fwa-mlops-accelerator-demo.fraud_experiments.random_forest_provider_predictions` (
  prediction_date DATE,
  provider_id STRING,
  experiment_id STRING,
  run_id STRING,
  model_name STRING,
  experiment_name STRING,
  fraud_score FLOAT64,
  risk_rank INT64,
  risk_category STRING,
  reason STRING,
  created_at TIMESTAMP,
  reason_1 STRING,
  reason_2 STRING,
  reason_3 STRING
);

INSERT INTO
`fwa-mlops-accelerator-demo.fraud_experiments.random_forest_provider_predictions` (
  prediction_date,
  provider_id,
  experiment_id,
  run_id,
  model_name,
  experiment_name,
  fraud_score,
  risk_rank,
  risk_category,
  reason,
  created_at,
  reason_1,
  reason_2,
  reason_3
)

WITH latest_rf_run AS (
  SELECT experiment_id, run_id, experiment_name
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs`
  WHERE model_type = 'random_forest'
  ORDER BY created_at DESC
  LIMIT 1
),

train_means AS (
  SELECT
    AVG(high_acuity_pct) AS mean_high_acuity_pct,
    AVG(pct_99214) AS mean_pct_99214,
    AVG(pct_99215) AS mean_pct_99215,
    AVG(referral_rate) AS mean_referral_rate,
    AVG(eye_procedure_pct) AS mean_eye_procedure_pct,
    AVG(avg_paid_per_claim) AS mean_avg_paid_per_claim,
    AVG(high_acuity_oe_ratio) AS mean_high_acuity_oe_ratio
  FROM `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE dataset_split = 'TRAIN'
),

rf_raw AS (
  SELECT
    provider_id,
    SAFE_CAST(prediction AS INT64) AS predicted_class
  FROM `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T13_08_45_116Z_589`
),

--calculate reasons
rf_features AS (
  SELECT
    p.provider_id,
    p.predicted_class,
    f.high_acuity_pct,
    f.pct_99214,
    f.pct_99215,
    f.referral_rate,
    f.eye_procedure_pct,
    f.avg_paid_per_claim,
    f.high_acuity_oe_ratio
  FROM rf_raw p
  JOIN `fwa-mlops-accelerator-demo.fraud_features.random_forest_batch_input` f
  USING(provider_id)
),

rf_deviations AS (
  SELECT
    r.*,
    ARRAY(
      SELECT AS STRUCT feature_name, deviation_ratio
      FROM UNNEST([
        STRUCT(
          'high_acuity_pct' AS feature_name,
          1 AS feature_order,
          IF(t.mean_high_acuity_pct != 0,
             r.high_acuity_pct / t.mean_high_acuity_pct, 0) AS deviation_ratio
        ),
        STRUCT(
          'pct_99214',
          2,
          IF(t.mean_pct_99214 != 0,
             r.pct_99214 / t.mean_pct_99214, 0)
        ),
        STRUCT(
          'pct_99215',
          3,
          IF(t.mean_pct_99215 != 0,
             r.pct_99215 / t.mean_pct_99215, 0)
        ),
        STRUCT(
          'referral_rate',
          4,
          IF(t.mean_referral_rate != 0,
             r.referral_rate / t.mean_referral_rate, 0)
        ),
        STRUCT(
          'eye_procedure_pct',
          5,
          IF(t.mean_eye_procedure_pct != 0,
             r.eye_procedure_pct / t.mean_eye_procedure_pct, 0)
        ),
        STRUCT(
          'avg_paid_per_claim',
          6,
          IF(t.mean_avg_paid_per_claim != 0,
             r.avg_paid_per_claim / t.mean_avg_paid_per_claim, 0)
        ),
        STRUCT(
          'high_acuity_oe_ratio',
          7,
          IF(t.mean_high_acuity_oe_ratio != 0,
             r.high_acuity_oe_ratio / t.mean_high_acuity_oe_ratio, 0)
        )
      ])
      ORDER BY deviation_ratio DESC, feature_order
    ) AS ranked_reasons
  FROM rf_features r
  CROSS JOIN train_means t
),

rf_ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      ORDER BY predicted_class DESC, provider_id
    ) AS risk_rank
  FROM rf_deviations
)

--qa table results
SELECT
  CURRENT_DATE() AS prediction_date,
  r.provider_id,
  e.experiment_id,
  e.run_id,
  'random_forest' AS model_name,
  e.experiment_name,
  CAST(r.predicted_class AS FLOAT64) AS fraud_score,
  r.risk_rank,
  CASE
    WHEN r.predicted_class = 1 THEN 'HIGH'
    ELSE 'LOW'
  END AS risk_category,
  CASE
    WHEN r.predicted_class = 1
      THEN 'Random Forest predicted provider as potential fraud'
    ELSE 'Random Forest predicted provider as normal'
  END AS reason,
  CURRENT_TIMESTAMP() AS created_at,
  r.ranked_reasons[SAFE_OFFSET(0)].feature_name AS reason_1,
  r.ranked_reasons[SAFE_OFFSET(1)].feature_name AS reason_2,
  r.ranked_reasons[SAFE_OFFSET(2)].feature_name AS reason_3
FROM rf_ranked r
CROSS JOIN latest_rf_run e;

