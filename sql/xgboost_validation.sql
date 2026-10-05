#XGBoost Experiment Validation


#Experiment metadata
SELECT
    experiment_id,
    run_id,
    experiment_name,
    model_type,
    feature_version,
    dataset_version,
    status,
    created_at
FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.experiment_runs`
WHERE
    model_type = 'xgboost'
ORDER BY
    created_at DESC;


#Experiment metrics

SELECT
    run_id,
    metric_name,
    metric_value
FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.experiment_metrics`
WHERE
    experiment_id = 'xgb_provider_eye_fraud'
ORDER BY
    created_at DESC,
    metric_name;


#Top XGBoost provider predictions

SELECT
    provider_id,
    fraud_score,
    risk_rank,
    risk_category,
    reason_1,
    reason_2,
    reason_3
FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions`
WHERE
    model_name = 'xgboost'
QUALIFY
    run_id = FIRST_VALUE(run_id) OVER (
        ORDER BY created_at DESC
    )
ORDER BY
    risk_rank
LIMIT 20;
