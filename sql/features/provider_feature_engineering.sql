-- ============================================================
-- FWA Fraud MLOps Accelerator
-- Step 4A: Provider-Level Feature Engineering
--
-- Source:
-- fwa_claims.claims
--
-- Output:
-- fraud_features.provider_eye_features
--
-- Grain:
-- One row per provider per feature snapshot
-- ============================================================


CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

PARTITION BY
  DATE(feature_timestamp)

CLUSTER BY
  provider_id

AS


-- ============================================================
-- 1. Aggregate claim-level data to provider level
-- ============================================================

WITH provider_metrics AS (

    SELECT

        -- ----------------------------------------------------
        -- Entity identifiers
        -- ----------------------------------------------------

        provider_id,

        -- Feature Store / BigQuery point-in-time functions
        -- can use entity_id as the entity identifier.
        provider_id AS entity_id,

        provider_city,
        provider_state,
        provider_specialty,

        dataset_split,


        -- ----------------------------------------------------
        -- Feature timestamp
        -- ----------------------------------------------------

        CASE

            WHEN dataset_split = 'TRAIN'
                THEN TIMESTAMP('2026-06-30 23:59:59+00')

            WHEN dataset_split = 'TEST'
                THEN TIMESTAMP('2026-09-30 23:59:59+00')

        END AS feature_timestamp,


        -- ----------------------------------------------------
        -- Claim volume
        -- ----------------------------------------------------

        COUNT(*) AS total_claims,

        COUNT(
            DISTINCT member_id
        ) AS unique_members,


        -- ----------------------------------------------------
        -- High-acuity E/M
        -- ----------------------------------------------------

        SUM(
            is_high_acuity_em
        ) AS high_acuity_claims,

        AVG(
            is_high_acuity_em
        ) AS high_acuity_pct,


        -- ----------------------------------------------------
        -- CPT 99214
        -- ----------------------------------------------------

        SUM(
            is_99214
        ) AS claims_99214,

        AVG(
            is_99214
        ) AS pct_99214,


        -- ----------------------------------------------------
        -- CPT 99215
        -- ----------------------------------------------------

        SUM(
            is_99215
        ) AS claims_99215,

        AVG(
            is_99215
        ) AS pct_99215,


        -- ----------------------------------------------------
        -- Referral behavior
        -- ----------------------------------------------------

        SUM(
            is_referral
        ) AS referral_count,

        AVG(
            is_referral
        ) AS referral_rate,


        -- ----------------------------------------------------
        -- Eye procedure utilization
        -- ----------------------------------------------------

        SUM(
            is_eye_procedure
        ) AS eye_procedure_claims,

        AVG(
            is_eye_procedure
        ) AS eye_procedure_pct,


        -- ----------------------------------------------------
        -- Financial features
        -- ----------------------------------------------------

        AVG(
            total_billed_amt
        ) AS avg_billed_per_claim,

        AVG(
            total_allowed_amt
        ) AS avg_allowed_per_claim,

        AVG(
            total_paid_amount
        ) AS avg_paid_per_claim,

        SUM(
            total_paid_amount
        ) AS total_paid


    FROM
        `fwa-mlops-accelerator-demo.fwa_claims.claims`


    GROUP BY

        provider_id,
        provider_city,
        provider_state,
        provider_specialty,
        dataset_split
),


-- ============================================================
-- 2. Calculate peer baseline
--
-- Instead of using the overall claim-level mean,
-- use the median provider high-acuity rate.
--
-- Median is more robust to unusually high-utilizing providers.
-- ============================================================

peer_baseline AS (

    SELECT

        dataset_split,

        APPROX_QUANTILES(
            high_acuity_pct,
            100
        )[OFFSET(50)]
            AS peer_high_acuity_rate

    FROM
        provider_metrics

    GROUP BY
        dataset_split
)


-- ============================================================
-- 3. Create final reusable feature table
-- ============================================================

SELECT

    -- --------------------------------------------------------
    -- IDs / metadata
    -- --------------------------------------------------------

    p.entity_id,

    p.provider_id,

    p.provider_city,

    p.provider_state,

    p.provider_specialty,

    p.dataset_split,

    p.feature_timestamp,


    -- --------------------------------------------------------
    -- Volume features
    -- --------------------------------------------------------

    p.total_claims,

    p.unique_members,


    -- --------------------------------------------------------
    -- High-acuity features
    -- --------------------------------------------------------

    p.high_acuity_claims,

    p.high_acuity_pct,


    -- --------------------------------------------------------
    -- CPT features
    -- --------------------------------------------------------

    p.claims_99214,

    p.pct_99214,

    p.claims_99215,

    p.pct_99215,


    -- --------------------------------------------------------
    -- Referral features
    -- --------------------------------------------------------

    p.referral_count,

    p.referral_rate,


    -- --------------------------------------------------------
    -- Eye procedure features
    -- --------------------------------------------------------

    p.eye_procedure_claims,

    p.eye_procedure_pct,


    -- --------------------------------------------------------
    -- Financial features
    -- --------------------------------------------------------

    ROUND(
        p.avg_billed_per_claim,
        2
    ) AS avg_billed_per_claim,

    ROUND(
        p.avg_allowed_per_claim,
        2
    ) AS avg_allowed_per_claim,

    ROUND(
        p.avg_paid_per_claim,
        2
    ) AS avg_paid_per_claim,

    ROUND(
        p.total_paid,
        2
    ) AS total_paid,


    -- --------------------------------------------------------
    -- Peer baseline
    -- --------------------------------------------------------

    b.peer_high_acuity_rate,


    -- --------------------------------------------------------
    -- Expected high-acuity claims
    --
    -- Expected =
    -- provider claim volume × peer high-acuity rate
    -- --------------------------------------------------------

    p.total_claims
        * b.peer_high_acuity_rate
        AS expected_high_acuity_claims,


    -- --------------------------------------------------------
    -- Observed / Expected ratio
    --
    -- > 1 = higher than peer expectation
    -- ~ 1 = close to peer expectation
    -- < 1 = lower than peer expectation
    -- --------------------------------------------------------

    SAFE_DIVIDE(

        p.high_acuity_claims,

        p.total_claims
            * b.peer_high_acuity_rate

    ) AS high_acuity_oe_ratio


FROM
    provider_metrics p

LEFT JOIN
    peer_baseline b

ON
    p.dataset_split = b.dataset_split;
-- ============================================================
-- FWA Fraud MLOps Accelerator
-- Step 4A: Provider-Level Feature Engineering
--
-- Source:
-- fwa_claims.claims
--
-- Output:
-- fraud_features.provider_eye_features
--
-- Grain:
-- One row per provider per feature snapshot
-- ============================================================


CREATE OR REPLACE TABLE
  `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`

PARTITION BY
  DATE(feature_timestamp)

CLUSTER BY
  provider_id

AS


-- ============================================================
-- 1. Aggregate claim-level data to provider level
-- ============================================================

WITH provider_metrics AS (

    SELECT

        -- ----------------------------------------------------
        -- Entity identifiers
        -- ----------------------------------------------------

        provider_id,

        -- Feature Store / BigQuery point-in-time functions
        -- can use entity_id as the entity identifier.
        provider_id AS entity_id,

        provider_city,
        provider_state,
        provider_specialty,

        dataset_split,


        -- ----------------------------------------------------
        -- Feature timestamp
        -- ----------------------------------------------------

        CASE

            WHEN dataset_split = 'TRAIN'
                THEN TIMESTAMP('2026-06-30 23:59:59+00')

            WHEN dataset_split = 'TEST'
                THEN TIMESTAMP('2026-09-30 23:59:59+00')

        END AS feature_timestamp,


        -- ----------------------------------------------------
        -- Claim volume
        -- ----------------------------------------------------

        COUNT(*) AS total_claims,

        COUNT(
            DISTINCT member_id
        ) AS unique_members,


        -- ----------------------------------------------------
        -- High-acuity E/M
        -- ----------------------------------------------------

        SUM(
            is_high_acuity_em
        ) AS high_acuity_claims,

        AVG(
            is_high_acuity_em
        ) AS high_acuity_pct,


        -- ----------------------------------------------------
        -- CPT 99214
        -- ----------------------------------------------------

        SUM(
            is_99214
        ) AS claims_99214,

        AVG(
            is_99214
        ) AS pct_99214,


        -- ----------------------------------------------------
        -- CPT 99215
        -- ----------------------------------------------------

        SUM(
            is_99215
        ) AS claims_99215,

        AVG(
            is_99215
        ) AS pct_99215,


        -- ----------------------------------------------------
        -- Referral behavior
        -- ----------------------------------------------------

        SUM(
            is_referral
        ) AS referral_count,

        AVG(
            is_referral
        ) AS referral_rate,


        -- ----------------------------------------------------
        -- Eye procedure utilization
        -- ----------------------------------------------------

        SUM(
            is_eye_procedure
        ) AS eye_procedure_claims,

        AVG(
            is_eye_procedure
        ) AS eye_procedure_pct,


        -- ----------------------------------------------------
        -- Financial features
        -- ----------------------------------------------------

        AVG(
            total_billed_amt
        ) AS avg_billed_per_claim,

        AVG(
            total_allowed_amt
        ) AS avg_allowed_per_claim,

        AVG(
            total_paid_amount
        ) AS avg_paid_per_claim,

        SUM(
            total_paid_amount
        ) AS total_paid


    FROM
        `fwa-mlops-accelerator-demo.fwa_claims.claims`


    GROUP BY

        provider_id,
        provider_city,
        provider_state,
        provider_specialty,
        dataset_split
),


-- ============================================================
-- 2. Calculate peer baseline
--
-- Instead of using the overall claim-level mean,
-- use the median provider high-acuity rate.
--
-- Median is more robust to unusually high-utilizing providers.
-- ============================================================

peer_baseline AS (

    SELECT

        dataset_split,

        APPROX_QUANTILES(
            high_acuity_pct,
            100
        )[OFFSET(50)]
            AS peer_high_acuity_rate

    FROM
        provider_metrics

    GROUP BY
        dataset_split
)


-- ============================================================
-- 3. Create final reusable feature table
-- ============================================================

SELECT

    -- --------------------------------------------------------
    -- IDs / metadata
    -- --------------------------------------------------------

    p.entity_id,

    p.provider_id,

    p.provider_city,

    p.provider_state,

    p.provider_specialty,

    p.dataset_split,

    p.feature_timestamp,


    -- --------------------------------------------------------
    -- Volume features
    -- --------------------------------------------------------

    p.total_claims,

    p.unique_members,


    -- --------------------------------------------------------
    -- High-acuity features
    -- --------------------------------------------------------

    p.high_acuity_claims,

    p.high_acuity_pct,


    -- --------------------------------------------------------
    -- CPT features
    -- --------------------------------------------------------

    p.claims_99214,

    p.pct_99214,

    p.claims_99215,

    p.pct_99215,


    -- --------------------------------------------------------
    -- Referral features
    -- --------------------------------------------------------

    p.referral_count,

    p.referral_rate,


    -- --------------------------------------------------------
    -- Eye procedure features
    -- --------------------------------------------------------

    p.eye_procedure_claims,

    p.eye_procedure_pct,


    -- --------------------------------------------------------
    -- Financial features
    -- --------------------------------------------------------

    ROUND(
        p.avg_billed_per_claim,
        2
    ) AS avg_billed_per_claim,

    ROUND(
        p.avg_allowed_per_claim,
        2
    ) AS avg_allowed_per_claim,

    ROUND(
        p.avg_paid_per_claim,
        2
    ) AS avg_paid_per_claim,

    ROUND(
        p.total_paid,
        2
    ) AS total_paid,


    -- --------------------------------------------------------
    -- Peer baseline
    -- --------------------------------------------------------

    b.peer_high_acuity_rate,


    -- --------------------------------------------------------
    -- Expected high-acuity claims
    --
    -- Expected =
    -- provider claim volume × peer high-acuity rate
    -- --------------------------------------------------------

    p.total_claims
        * b.peer_high_acuity_rate
        AS expected_high_acuity_claims,


    -- --------------------------------------------------------
    -- Observed / Expected ratio
    --
    -- > 1 = higher than peer expectation
    -- ~ 1 = close to peer expectation
    -- < 1 = lower than peer expectation
    -- --------------------------------------------------------

    SAFE_DIVIDE(

        p.high_acuity_claims,

        p.total_claims
            * b.peer_high_acuity_rate

    ) AS high_acuity_oe_ratio


FROM
    provider_metrics p

LEFT JOIN
    peer_baseline b

ON
    p.dataset_split = b.dataset_split;
