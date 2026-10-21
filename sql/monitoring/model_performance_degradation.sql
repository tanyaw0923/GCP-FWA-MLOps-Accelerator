-- ============================================================
-- PRODUCTION MODEL PERFORMANCE DEGRADATION
--
-- Purpose:
-- Store biweekly evaluation of the currently approved
-- production fraud model against completed investigation
-- outcomes.
--
-- This table stores monitoring HISTORY. Each scheduled run
-- appends one row.
-- ============================================================

CREATE TABLE IF NOT EXISTS
`fwa-mlops-accelerator-demo.fraud_monitoring.model_performance_degradation`
(
    evaluation_timestamp TIMESTAMP,
    evaluation_date DATE,

    model_name STRING,
    model_type STRING,

    evaluation_window_start DATE,
    evaluation_window_end DATE,

    evaluated_cases INT64,
    confirmed_fraud_cases INT64,
    confirmed_normal_cases INT64,

    true_positive INT64,
    false_positive INT64,
    true_negative INT64,
    false_negative INT64,

    precision FLOAT64,
    recall FLOAT64,
    f1_score FLOAT64,
    fraud_yield FLOAT64,

    baseline_precision FLOAT64,
    baseline_recall FLOAT64,
    baseline_f1_score FLOAT64,

    precision_change FLOAT64,
    recall_change FLOAT64,
    f1_change FLOAT64,

    monitoring_status STRING,
    alert_reason STRING
);
