-- ============================================================
-- FWA Fraud MLOps Accelerator
-- Step 4B: Provider Feature Validation
--
-- Source:
-- fwa-mlops-accelerator-demo.fraud_features.provider_eye_features
-- ============================================================


-- ============================================================
-- VALIDATION 1
-- Confirm one provider row per snapshot
-- ============================================================

SELECT
    dataset_split,
    COUNT(*) AS provider_rows,
    COUNT(DISTINCT provider_id) AS providers,
    MIN(feature_timestamp) AS feature_timestamp
FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
GROUP BY
    dataset_split
ORDER BY
    dataset_split;


-- Expected:
--
-- TRAIN
-- approximately 200 provider rows
-- approximately 200 unique providers
-- feature_timestamp = 2026-06-30
--
-- TEST
-- approximately 200 provider rows
-- approximately 200 unique providers
-- feature_timestamp = 2026-09-30



-- ============================================================
-- VALIDATION 2
-- Inspect highest observed / expected providers
-- ============================================================

SELECT
    provider_id,
    dataset_split,
    total_claims,
    unique_members,
    high_acuity_pct,
    pct_99214,
    pct_99215,
    referral_rate,
    eye_procedure_pct,
    avg_paid_per_claim,
    peer_high_acuity_rate,
    high_acuity_oe_ratio
FROM
    `fwa-mlops-accelerator-demo.fraud_features.provider_eye_features`
ORDER BY
    high_acuity_oe_ratio DESC
LIMIT 20;


-- Expected:
--
-- Providers with unusually high:
--   high_acuity_pct
--   pct_99215
--   referral_rate
--   high_acuity_oe_ratio
--
-- should appear near the top.
--
-- This ranking is exploratory only.
-- It is NOT a confirmed fraud label.
