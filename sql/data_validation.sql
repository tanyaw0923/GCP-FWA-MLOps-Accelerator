-- ============================================================
-- CURRENT SYNTHETIC CLAIM DATA VALIDATION
-- ============================================================

-- Volume and cardinality
SELECT
  COUNT(*) AS total_claims,
  COUNT(DISTINCT provider_id) AS total_providers,
  COUNT(DISTINCT member_id) AS total_members,
  MIN(service_date) AS min_service_date,
  MAX(service_date) AS max_service_date

FROM
  `fwa-mlops-accelerator-demo.fwa_claims.claims`;


-- Provider cohort
SELECT
  provider_specialty,
  provider_state,
  COUNT(DISTINCT provider_id) AS providers,
  COUNT(*) AS claims

FROM
  `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY
  provider_specialty,
  provider_state

ORDER BY claims DESC;


-- Eye-diagnosis sanity check.
-- The current generator uses H-family ICD codes for the demo cohort.
SELECT
  SUBSTR(
    CAST(icd_code AS STRING),
    1,
    1
  ) AS icd_prefix,

  COUNT(*) AS claim_count

FROM
  `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY icd_prefix

ORDER BY claim_count DESC;


-- Label prevalence
WITH provider_labels AS (
  SELECT
    provider_id,
    MAX(
      CAST(
        fraud_provider_label
        AS INT64
      )
    ) AS fraud_label

  FROM
    `fwa-mlops-accelerator-demo.fwa_claims.claims`

  GROUP BY provider_id
)

SELECT
  fraud_label,
  COUNT(*) AS providers,
  SAFE_DIVIDE(
    COUNT(*),
    SUM(COUNT(*)) OVER()
  ) AS provider_rate

FROM provider_labels

GROUP BY fraud_label

ORDER BY fraud_label;


-- Null and domain checks
SELECT
  COUNT(*) AS total_rows,

  COUNTIF(claim_id IS NULL) AS missing_claim_id,
  COUNTIF(provider_id IS NULL) AS missing_provider_id,
  COUNTIF(member_id IS NULL) AS missing_member_id,
  COUNTIF(service_date IS NULL) AS missing_service_date,
  COUNTIF(cpt_code IS NULL) AS missing_cpt_code,
  COUNTIF(icd_code IS NULL) AS missing_icd_code,

  COUNTIF(total_billed_amt IS NULL)
    AS missing_billed_amount,

  COUNTIF(total_allowed_amt IS NULL)
    AS missing_allowed_amount,

  COUNTIF(total_paid_amount IS NULL)
    AS missing_paid_amount,

  COUNTIF(
    total_paid_amount < 0
    OR total_allowed_amt < 0
    OR total_billed_amt < 0
  ) AS invalid_negative_amounts,

  COUNTIF(
    fraud_provider_label NOT IN (0, 1)
  ) AS invalid_fraud_labels

FROM
  `fwa-mlops-accelerator-demo.fwa_claims.claims`;


-- Monthly volumes make temporal changes visible.
SELECT
  DATE_TRUNC(
    service_date,
    MONTH
  ) AS service_month,

  COUNT(*) AS claims,
  COUNT(DISTINCT provider_id) AS providers,
  COUNT(DISTINCT member_id) AS members

FROM
  `fwa-mlops-accelerator-demo.fwa_claims.claims`

GROUP BY service_month

ORDER BY service_month;
