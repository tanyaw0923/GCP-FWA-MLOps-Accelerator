-- AUTO-GENERATED from pipeline_config.yaml; do not edit directly.
-- Active model: random_forest
-- Regenerate: python -m src.monitoring.generate_monitoring_sql

-- ============================================================
-- GENERIC MODEL MONITORING
-- FWA MLOps Accelerator
--
-- Shared across Random Forest, XGBoost, or future model types.
-- ============================================================


-- ============================================================
-- 1. DATA QUALITY MONITORING
--
-- Source:
--   provider-level TEST dataset
--
-- Target:
--   fraud_monitoring.model_data_quality
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  dataset_name STRING,

  total_rows INT64,
  unique_providers INT64,
  duplicate_rows INT64,

  null_feature_values INT64,
  invalid_feature_values INT64,

  monitoring_status STRING
);

-- Backward-compatible schema migration for tables created by
-- earlier versions of the demo. Keep this as one metadata update
-- to avoid BigQuery table-update rate limits.
ALTER TABLE
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`
ADD COLUMN IF NOT EXISTS monitoring_timestamp TIMESTAMP,
ADD COLUMN IF NOT EXISTS model_name STRING,
ADD COLUMN IF NOT EXISTS dataset_name STRING,
ADD COLUMN IF NOT EXISTS total_rows INT64,
ADD COLUMN IF NOT EXISTS unique_providers INT64,
ADD COLUMN IF NOT EXISTS duplicate_rows INT64,
ADD COLUMN IF NOT EXISTS null_feature_values INT64,
ADD COLUMN IF NOT EXISTS invalid_feature_values INT64,
ADD COLUMN IF NOT EXISTS monitoring_status STRING;


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`
(
  monitoring_timestamp,
  model_name,
  dataset_name,
  total_rows,
  unique_providers,
  duplicate_rows,
  null_feature_values,
  invalid_feature_values,
  monitoring_status
)

WITH test_data AS (

  SELECT
    *
  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
  WHERE
    dataset_split = 'TEST'

),

dq AS (

  SELECT
    COUNT(*) AS total_rows,

    COUNT(
      DISTINCT provider_id
    ) AS unique_providers,

    COUNT(*)
      - COUNT(
          DISTINCT provider_id
        )
      AS duplicate_rows,

    (COUNTIF(`high_acuity_pct` IS NULL)
      + COUNTIF(`pct_99214` IS NULL)
      + COUNTIF(`pct_99215` IS NULL)
      + COUNTIF(`referral_rate` IS NULL)
      + COUNTIF(`eye_procedure_pct` IS NULL)
      + COUNTIF(`avg_paid_per_claim` IS NULL)
      + COUNTIF(`high_acuity_oe_ratio` IS NULL)
      + COUNTIF(`avg_member_provider_distance` IS NULL)
    ) AS null_feature_values,

    (COUNTIF(`high_acuity_pct` < 0 OR `high_acuity_pct` > 1)
      + COUNTIF(`pct_99214` < 0 OR `pct_99214` > 1)
      + COUNTIF(`pct_99215` < 0 OR `pct_99215` > 1)
      + COUNTIF(`referral_rate` < 0 OR `referral_rate` > 1)
      + COUNTIF(`eye_procedure_pct` < 0 OR `eye_procedure_pct` > 1)
      + COUNTIF(`avg_paid_per_claim` < 0)
      + COUNTIF(`high_acuity_oe_ratio` < 0)
      + COUNTIF(`avg_member_provider_distance` < 0)
    ) AS invalid_feature_values

  FROM
    test_data
)

SELECT
  CURRENT_TIMESTAMP()
    AS monitoring_timestamp,

  'provider_fraud_model'
    AS model_name,

  'TEST'
    AS dataset_name,

  total_rows,
  unique_providers,
  COALESCE(duplicate_rows, 0)
    AS duplicate_rows,
  COALESCE(null_feature_values, 0)
    AS null_feature_values,
  COALESCE(invalid_feature_values, 0)
    AS invalid_feature_values,

  CASE
    WHEN total_rows = 0
      THEN 'ALERT'

    WHEN COALESCE(duplicate_rows, 0) > 0
      THEN 'ALERT'

    WHEN COALESCE(null_feature_values, 0) > 0
      THEN 'ALERT'

    WHEN COALESCE(invalid_feature_values, 0) > 0
      THEN 'ALERT'

    ELSE 'PASS'
  END AS monitoring_status

FROM
  dq;



-- ============================================================
-- 2. FEATURE DRIFT MONITORING
--
-- TRAIN = baseline
-- TEST  = current
--
-- Target:
--   fraud_monitoring.model_feature_drift
--
-- Uses standardized mean shift:
--
--   |current mean - baseline mean|
--   --------------------------------
--            baseline std
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`
(
  monitoring_timestamp TIMESTAMP,
  model_name STRING,
  feature_name STRING,

  baseline_mean FLOAT64,
  current_mean FLOAT64,
  baseline_std FLOAT64,

  standardized_mean_shift FLOAT64,

  monitoring_status STRING
);

-- Backward-compatible schema migration for tables created by
-- earlier versions of the demo. Keep this as one metadata update
-- to avoid BigQuery table-update rate limits.
ALTER TABLE
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`
ADD COLUMN IF NOT EXISTS monitoring_timestamp TIMESTAMP,
ADD COLUMN IF NOT EXISTS model_name STRING,
ADD COLUMN IF NOT EXISTS feature_name STRING,
ADD COLUMN IF NOT EXISTS baseline_mean FLOAT64,
ADD COLUMN IF NOT EXISTS current_mean FLOAT64,
ADD COLUMN IF NOT EXISTS baseline_std FLOAT64,
ADD COLUMN IF NOT EXISTS standardized_mean_shift FLOAT64,
ADD COLUMN IF NOT EXISTS monitoring_status STRING;


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`
(
  monitoring_timestamp,
  model_name,
  feature_name,
  baseline_mean,
  current_mean,
  baseline_std,
  standardized_mean_shift,
  monitoring_status
)

WITH feature_values AS (

  SELECT
    dataset_split,
    feature_name,
    feature_value

  FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

  UNPIVOT (
    feature_value
    FOR feature_name IN (
      `high_acuity_pct`,
      `pct_99214`,
      `pct_99215`,
      `referral_rate`,
      `eye_procedure_pct`,
      `avg_paid_per_claim`,
      `high_acuity_oe_ratio`,
      `avg_member_provider_distance`
    )
  )

),

baseline AS (

  SELECT
    feature_name,

    AVG(
      feature_value
    ) AS baseline_mean,

    STDDEV(
      feature_value
    ) AS baseline_std

  FROM
    feature_values

  WHERE
    dataset_split = 'TRAIN'

  GROUP BY
    feature_name

),

current_data AS (

  SELECT
    feature_name,

    AVG(
      feature_value
    ) AS current_mean

  FROM
    feature_values

  WHERE
    dataset_split = 'TEST'

  GROUP BY
    feature_name

),

drift AS (

  SELECT
    b.feature_name,
    b.baseline_mean,
    c.current_mean,
    b.baseline_std,

    SAFE_DIVIDE(
      ABS(
        c.current_mean
        - b.baseline_mean
      ),
      NULLIF(
        b.baseline_std,
        0
      )
    ) AS standardized_mean_shift

  FROM
    baseline b

  JOIN
    current_data c

  USING (
    feature_name
  )

)

SELECT
  CURRENT_TIMESTAMP()
    AS monitoring_timestamp,

  'provider_fraud_model'
    AS model_name,

  feature_name,

  baseline_mean,
  current_mean,
  baseline_std,

  standardized_mean_shift,

  CASE
    WHEN standardized_mean_shift >= 1.0
      THEN 'ALERT'

    ELSE 'PASS'
  END AS monitoring_status

FROM
  drift;



-- ============================================================
-- 3. PREDICTION-SCORE DRIFT MONITORING
--
-- Curated provider_predictions stores continuous fraud_score,
-- so drift should monitor the score distribution rather than an
-- obsolete binary prediction field.
--
-- Compare the newest scored run with the immediately previous run
-- for the SAME model version.
--
-- Metrics:
--   PSI across ten fixed score buckets
--   mean fraud-score shift
--   p50 / p90 / p95 movement
--
-- Demo interpretation:
--   PSI < 0.10       PASS
--   0.10 <= PSI < .20 WATCH
--   PSI >= 0.20      ALERT
-- ============================================================

CREATE TABLE IF NOT EXISTS
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
(
  check_date TIMESTAMP,
  model_name STRING,
  status STRING
);

-- Backward-compatible schema migration. Keep this as one
-- metadata update to avoid BigQuery table-update rate limits.
ALTER TABLE
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
ADD COLUMN IF NOT EXISTS check_date TIMESTAMP,
ADD COLUMN IF NOT EXISTS monitoring_timestamp TIMESTAMP,
ADD COLUMN IF NOT EXISTS model_name STRING,
ADD COLUMN IF NOT EXISTS model_version STRING,
ADD COLUMN IF NOT EXISTS baseline_run_id STRING,
ADD COLUMN IF NOT EXISTS current_run_id STRING,
ADD COLUMN IF NOT EXISTS baseline_count INT64,
ADD COLUMN IF NOT EXISTS current_count INT64,
ADD COLUMN IF NOT EXISTS baseline_mean_score FLOAT64,
ADD COLUMN IF NOT EXISTS current_mean_score FLOAT64,
ADD COLUMN IF NOT EXISTS mean_score_shift FLOAT64,
ADD COLUMN IF NOT EXISTS baseline_p50 FLOAT64,
ADD COLUMN IF NOT EXISTS current_p50 FLOAT64,
ADD COLUMN IF NOT EXISTS baseline_p90 FLOAT64,
ADD COLUMN IF NOT EXISTS current_p90 FLOAT64,
ADD COLUMN IF NOT EXISTS baseline_p95 FLOAT64,
ADD COLUMN IF NOT EXISTS current_p95 FLOAT64,
ADD COLUMN IF NOT EXISTS population_stability_index FLOAT64,
ADD COLUMN IF NOT EXISTS status STRING,
ADD COLUMN IF NOT EXISTS alert_reason STRING;


INSERT INTO
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`
(
  check_date,
  monitoring_timestamp,
  model_name,
  model_version,
  baseline_run_id,
  current_run_id,
  baseline_count,
  current_count,
  baseline_mean_score,
  current_mean_score,
  mean_score_shift,
  baseline_p50,
  current_p50,
  baseline_p90,
  current_p90,
  baseline_p95,
  current_p95,
  population_stability_index,
  status,
  alert_reason
)

WITH runs AS (
  SELECT
    model_name,
    model_version,
    run_id,
    MAX(created_at) AS run_created_at

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`

  WHERE fraud_score IS NOT NULL

  GROUP BY
    model_name,
    model_version,
    run_id
),

current_run AS (
  SELECT
    model_name,
    model_version,
    run_id,
    run_created_at

  FROM runs

  QUALIFY ROW_NUMBER() OVER (
    ORDER BY run_created_at DESC
  ) = 1
),

baseline_run AS (
  SELECT
    r.model_name,
    r.model_version,
    r.run_id,
    r.run_created_at

  FROM runs r

  JOIN current_run c
    USING(model_version)

  WHERE
    r.run_id != c.run_id

  QUALIFY ROW_NUMBER() OVER (
    ORDER BY r.run_created_at DESC
  ) = 1
),

score_rows AS (
  SELECT
    'CURRENT' AS population,
    p.fraud_score

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

  JOIN current_run r
    ON p.run_id = r.run_id
    AND p.model_version = r.model_version

  WHERE p.fraud_score IS NOT NULL

  UNION ALL

  SELECT
    'BASELINE' AS population,
    p.fraud_score

  FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

  JOIN baseline_run r
    ON p.run_id = r.run_id
    AND p.model_version = r.model_version

  WHERE p.fraud_score IS NOT NULL
),

summary AS (
  SELECT
    population,
    COUNT(*) AS row_count,
    AVG(fraud_score) AS mean_score,
    APPROX_QUANTILES(
      fraud_score,
      100
    )[OFFSET(50)] AS p50,
    APPROX_QUANTILES(
      fraud_score,
      100
    )[OFFSET(90)] AS p90,
    APPROX_QUANTILES(
      fraud_score,
      100
    )[OFFSET(95)] AS p95

  FROM score_rows

  GROUP BY population
),

bucket_counts AS (
  SELECT
    population,

    LEAST(
      CAST(
        FLOOR(fraud_score * 10)
        AS INT64
      ),
      9
    ) AS score_bucket,

    COUNT(*) AS bucket_count

  FROM score_rows

  GROUP BY
    population,
    score_bucket
),

bucket_rates AS (
  SELECT
    b.population,
    b.score_bucket,

    SAFE_DIVIDE(
      b.bucket_count,
      s.row_count
    ) AS bucket_rate

  FROM bucket_counts b

  JOIN summary s
    USING(population)
),

all_buckets AS (
  SELECT score_bucket

  FROM UNNEST(
    GENERATE_ARRAY(0, 9)
  ) AS score_bucket
),

psi AS (
  SELECT
    SUM(
      (
        GREATEST(
          COALESCE(c.bucket_rate, 0),
          0.0001
        )
        -
        GREATEST(
          COALESCE(b.bucket_rate, 0),
          0.0001
        )
      )
      *
      LN(
        SAFE_DIVIDE(
          GREATEST(
            COALESCE(c.bucket_rate, 0),
            0.0001
          ),
          GREATEST(
            COALESCE(b.bucket_rate, 0),
            0.0001
          )
        )
      )
    ) AS population_stability_index

  FROM all_buckets x

  LEFT JOIN bucket_rates b
    ON x.score_bucket = b.score_bucket
    AND b.population = 'BASELINE'

  LEFT JOIN bucket_rates c
    ON x.score_bucket = c.score_bucket
    AND c.population = 'CURRENT'
),

final AS (
  SELECT
    c.model_name,
    c.model_version,
    b.run_id AS baseline_run_id,
    c.run_id AS current_run_id,

    bs.row_count AS baseline_count,
    cs.row_count AS current_count,

    bs.mean_score AS baseline_mean_score,
    cs.mean_score AS current_mean_score,

    ABS(
      cs.mean_score
      - bs.mean_score
    ) AS mean_score_shift,

    bs.p50 AS baseline_p50,
    cs.p50 AS current_p50,
    bs.p90 AS baseline_p90,
    cs.p90 AS current_p90,
    bs.p95 AS baseline_p95,
    cs.p95 AS current_p95,

    p.population_stability_index

  FROM current_run c

  LEFT JOIN baseline_run b
    USING(model_version)

  LEFT JOIN summary bs
    ON bs.population = 'BASELINE'

  LEFT JOIN summary cs
    ON cs.population = 'CURRENT'

  CROSS JOIN psi p
)

SELECT
  CURRENT_TIMESTAMP()
    AS check_date,

  CURRENT_TIMESTAMP()
    AS monitoring_timestamp,

  model_name,
  model_version,
  baseline_run_id,
  current_run_id,
  baseline_count,
  current_count,
  baseline_mean_score,
  current_mean_score,
  mean_score_shift,
  baseline_p50,
  current_p50,
  baseline_p90,
  current_p90,
  baseline_p95,
  current_p95,
  population_stability_index,

  CASE
    WHEN baseline_run_id IS NULL
      THEN 'INSUFFICIENT_DATA'

    WHEN baseline_count = 0
      OR current_count = 0
      THEN 'INSUFFICIENT_DATA'

    WHEN population_stability_index >= 0.20
      THEN 'ALERT'

    WHEN mean_score_shift >= 0.10
      THEN 'ALERT'

    WHEN population_stability_index >= 0.10
      THEN 'WATCH'

    ELSE 'PASS'
  END AS status,

  CASE
    WHEN baseline_run_id IS NULL
      THEN 'No previous scored run exists for the same model version.'

    WHEN population_stability_index >= 0.20
      THEN 'Prediction score PSI exceeded 0.20.'

    WHEN mean_score_shift >= 0.10
      THEN 'Mean fraud score shifted by at least 0.10.'

    WHEN population_stability_index >= 0.10
      THEN 'Prediction score PSI is in the watch range.'

    ELSE 'Prediction score distribution is stable.'
  END AS alert_reason

FROM final;



-- ============================================================
-- 4. QA: LATEST DATA QUALITY RESULT
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_data_quality`

ORDER BY
  monitoring_timestamp DESC

LIMIT 10;



-- ============================================================
-- 5. QA: LATEST FEATURE DRIFT RUN
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`

WHERE
  monitoring_timestamp = (
    SELECT
      MAX(
        monitoring_timestamp
      )
    FROM
      `fwa-mlops-accelerator-demo.fraud_monitoring.model_feature_drift`
  )

ORDER BY
  standardized_mean_shift DESC;



-- ============================================================
-- 6. QA: LATEST PREDICTION DRIFT RESULT
-- ============================================================

SELECT
  *
FROM
  `fwa-mlops-accelerator-demo.fraud_monitoring.model_prediction_drift`

ORDER BY
  check_date DESC

LIMIT 10;
