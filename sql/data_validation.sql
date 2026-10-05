-- ============================================================
-- FWA Fraud MLOps Accelerator
-- BigQuery Data Validation
--
-- Source table:
-- fwa-mlops-accelerator-demo.fwa_claims.claims
-- ============================================================


-- ============================================================
-- VALIDATE CLAIM TABLE LOAD
-- ============================================================

SELECT
    COUNT(*) AS total_claims,
    COUNT(DISTINCT provider_id) AS total_providers,
    COUNT(DISTINCT member_id) AS total_members
FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`;


-- Expected:
-- total_claims     = 100000
-- total_providers  = 200
-- total_members    ≈ 20000



-- ============================================================
-- VALIDATE TRAIN / TEST TEMPORAL SPLIT
-- ============================================================

SELECT
    dataset_split,

    MIN(service_date) AS min_service_date,

    MAX(service_date) AS max_service_date,

    COUNT(*) AS claim_count,

    COUNT(DISTINCT provider_id) AS provider_count,

    COUNT(DISTINCT member_id) AS member_count

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY
    dataset_split

ORDER BY
    dataset_split;


-- Expected:
--
-- TRAIN
-- 2026-01-01 through 2026-06-30
--
-- TEST
-- 2026-07-01 through 2026-09-30
--
-- Both periods should contain approximately all 200 providers.



-- ============================================================
-- VALIDATE EYE-RELATED CLAIM COHORT
-- ============================================================

SELECT
    COUNT(*) AS total_claims,

    SUM(is_eye_diagnosis) AS eye_claims,

    ROUND(
        100 * AVG(is_eye_diagnosis),
        2
    ) AS eye_claim_pct

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`;


-- Expected:
-- total_claims    = 100000
-- eye_claims      = 100000
-- eye_claim_pct   = 100.00



-- ============================================================
-- ADDITIONAL VALIDATION:
-- Verify ICD codes are eye-related H-family codes
-- ============================================================

SELECT
    SUBSTR(
        CAST(icd_code AS STRING),
        1,
        1
    ) AS icd_prefix,

    COUNT(*) AS claim_count

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY
    icd_prefix

ORDER BY
    claim_count DESC;


-- Expected:
-- Only prefix "H"



-- ============================================================
-- ADDITIONAL VALIDATION:
-- Provider specialty
-- ============================================================

SELECT
    provider_specialty,

    COUNT(*) AS claim_count,

    COUNT(DISTINCT provider_id) AS provider_count

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY
    provider_specialty

ORDER BY
    claim_count DESC;


-- Expected:
-- Primary Care only



-- ============================================================
-- ADDITIONAL VALIDATION:
-- Texas-only provider population
-- ============================================================

SELECT
    provider_state,

    COUNT(*) AS claim_count,

    COUNT(DISTINCT provider_id) AS provider_count

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY
    provider_state

ORDER BY
    claim_count DESC;


-- Expected:
-- TX only



-- ============================================================
-- ADDITIONAL VALIDATION:
-- Providers by city
-- ============================================================

SELECT
    provider_city,

    COUNT(DISTINCT provider_id) AS provider_count,

    COUNT(*) AS claim_count

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY
    provider_city

ORDER BY
    provider_count DESC;



-- ============================================================
-- ADDITIONAL VALIDATION:
-- Basic null check
-- ============================================================

SELECT
    COUNT(*) AS total_rows,

    COUNTIF(claim_id IS NULL) AS missing_claim_id,

    COUNTIF(provider_id IS NULL) AS missing_provider_id,

    COUNTIF(member_id IS NULL) AS missing_member_id,

    COUNTIF(service_date IS NULL) AS missing_service_date,

    COUNTIF(cpt_code IS NULL) AS missing_cpt_code,

    COUNTIF(icd_code IS NULL) AS missing_icd_code,

    COUNTIF(total_billed_amt IS NULL) AS missing_billed_amount,

    COUNTIF(total_allowed_amt IS NULL) AS missing_allowed_amount,

    COUNTIF(total_paid_amount IS NULL) AS missing_paid_amount

FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`;
