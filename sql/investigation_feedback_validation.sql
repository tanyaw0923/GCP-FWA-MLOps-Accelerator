#Investigator Feedback Validation
#Overall investigation outcomes

SELECT
    investigation_result,
    COUNT(*) AS provider_count,
    ROUND(
        100 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS pct_of_reviewed
FROM
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`
GROUP BY
    investigation_result
ORDER BY
    provider_count DESC;



# Fraud yield and fraud dollars

SELECT
    COUNT(*) AS providers_reviewed,

    COUNTIF(
        confirmed_fraud = TRUE
    ) AS confirmed_fraud,

    COUNTIF(
        confirmed_fraud = FALSE
    ) AS confirmed_normal,

    ROUND(
        100 * AVG(
            CAST(confirmed_fraud AS INT64)
        ),
        2
    ) AS fraud_yield_pct,

    ROUND(
        SUM(fraud_amount),
        2
    ) AS confirmed_fraud_amount

FROM
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results`;



# Compare model rank with investigation result

SELECT
    p.risk_rank,
    p.provider_id,
    p.fraud_score,
    f.investigation_result,
    f.confirmed_fraud,
    f.fraud_amount

FROM
    `fwa-mlops-accelerator-demo.fraud_experiments.provider_predictions` p

JOIN
    `fwa-mlops-accelerator-demo.fraud_feedback.investigation_results` f

USING (
    provider_id,
    model_version
)

WHERE
    p.model_name = 'isolation_forest'

ORDER BY
    p.risk_rank

LIMIT 50;
