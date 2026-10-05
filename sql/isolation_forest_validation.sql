# check experiment history runs
SELECT
    experiment_id,
    run_id,
    experiment_name,
    model_type,
    feature_version,
    training_start,
    training_end,
    status,
    created_at
FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs`
ORDER BY
    created_at DESC
LIMIT 10;


# check experiment metrics
SELECT
    run_id,
    metric_name,
    metric_value
FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.experiment_metrics`
ORDER BY
    created_at DESC,
    metric_name;


# check predictions
SELECT
    provider_id,
    model_name,
    model_version,
    fraud_score,
    risk_rank,
    risk_category,
    reason_1,
    reason_2,
    reason_3
FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
WHERE
    model_name = 'isolation_forest'
ORDER BY
    risk_rank
LIMIT 20;
