from datetime import datetime, timezone

from google.cloud import bigquery

from src.config.load_config import load_config


# ============================================================
# CONSTANTS
# ============================================================

STATE_KEY = "provider_fraud_model"

MIN_COMPLETED_CASES = 20

# Relative degradation thresholds
PRECISION_DEGRADATION_THRESHOLD = 0.10
RECALL_DEGRADATION_THRESHOLD = 0.15
F1_DEGRADATION_THRESHOLD = 0.10


# ============================================================
# LOAD CONFIG
# ============================================================

config = load_config()

PROJECT_ID = config[
    "project_id"
]

PREDICTIONS_TABLE = (
    config[
        "data"
    ][
        "provider_prediction_table"
    ]
)

FEEDBACK_TABLE = (
    config[
        "data"
    ][
        "feedback_table"
    ]
)

EXPERIMENT_METRICS_TABLE = (
    config[
        "data"
    ][
        "experiment_metrics_table"
    ]
)

EXPERIMENT_RUNS_TABLE = (
    config[
        "data"
    ][
        "experiment_runs_table"
    ]
)

PERFORMANCE_TABLE = (
    f"{PROJECT_ID}."
    "fraud_monitoring."
    "model_performance_degradation"
)


# ============================================================
# INITIALIZE
# ============================================================

client = bigquery.Client(
    project=PROJECT_ID
)

evaluation_timestamp = datetime.now(
    timezone.utc
)


# ============================================================
# 1. CREATE MONITORING TABLE IF NEEDED
# ============================================================

create_table_query = f"""
CREATE TABLE IF NOT EXISTS `{PERFORMANCE_TABLE}`
(
    evaluation_timestamp TIMESTAMP,
    evaluation_date DATE,

    model_name STRING,
    model_version STRING,
    run_id STRING,

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
    baseline_f1 FLOAT64,

    precision_change FLOAT64,
    recall_change FLOAT64,
    f1_change FLOAT64,

    monitoring_status STRING,
    alert_reason STRING
)
"""

client.query(
    create_table_query
).result()


# ============================================================
# 2. FIND LATEST MODEL RUN WITH INVESTIGATION RESULTS
#
# Do not rely on approved_model_type for the current demo.
#
# We evaluate whichever prediction run most recently received
# completed investigation outcomes.
# ============================================================

latest_run_query = f"""
SELECT
    f.model_name,
    f.model_version,
    f.run_id,

    MAX(
        f.created_at
    ) AS latest_feedback_timestamp

FROM `{FEEDBACK_TABLE}` f

WHERE
    f.run_id IS NOT NULL

    AND f.model_name IS NOT NULL

    AND f.model_version IS NOT NULL

    AND f.investigation_result IN (
        'CONFIRMED_FRAUD',
        'CONFIRMED_NORMAL'
    )

GROUP BY
    f.model_name,
    f.model_version,
    f.run_id

ORDER BY
    latest_feedback_timestamp DESC

LIMIT 1
"""

latest_run_rows = list(
    client.query(
        latest_run_query
    ).result()
)

if not latest_run_rows:
    raise RuntimeError(
        "No completed investigation results "
        "were found."
    )

latest_run = latest_run_rows[0]

model_name = (
    latest_run.model_name
)

model_version = (
    latest_run.model_version
)

run_id = (
    latest_run.run_id
)


print(
    "\n=============================================="
)

print(
    "PRODUCTION PERFORMANCE MONITORING"
)

print(
    "=============================================="
)

print(
    f"Model: "
    f"{model_name}"
)

print(
    f"Model version: "
    f"{model_version}"
)

print(
    f"Run ID: "
    f"{run_id}"
)


# ============================================================
# 3. JOIN PREDICTIONS TO INVESTIGATION OUTCOMES
#
# Exact lineage:
#
# provider_id
# +
# run_id
# +
# model_version
#
# This prevents feedback from one model/run from being used
# to evaluate another.
# ============================================================

performance_query = f"""
WITH evaluated AS (

    SELECT
        p.provider_id,

        p.fraud_score,

        CASE
            WHEN p.fraud_score >= 0.5
                THEN 1
            ELSE 0
        END AS predicted_fraud,

        CASE
            WHEN f.investigation_result = 'CONFIRMED_FRAUD'
                THEN 1
            ELSE 0
        END AS actual_fraud

    FROM `{PREDICTIONS_TABLE}` p

    INNER JOIN `{FEEDBACK_TABLE}` f

        ON p.provider_id = f.provider_id

        AND p.run_id = f.run_id

        AND p.model_version = f.model_version

    WHERE
        p.model_name = @model_name

        AND p.model_version = @model_version

        AND p.run_id = @run_id

        AND f.model_name = @model_name

        AND f.investigation_result IN (
            'CONFIRMED_FRAUD',
            'CONFIRMED_NORMAL'
        )
)

SELECT
    COUNT(*)
        AS evaluated_cases,

    COUNTIF(
        actual_fraud = 1
    ) AS confirmed_fraud_cases,

    COUNTIF(
        actual_fraud = 0
    ) AS confirmed_normal_cases,

    COUNTIF(
        predicted_fraud = 1
        AND actual_fraud = 1
    ) AS true_positive,

    COUNTIF(
        predicted_fraud = 1
        AND actual_fraud = 0
    ) AS false_positive,

    COUNTIF(
        predicted_fraud = 0
        AND actual_fraud = 0
    ) AS true_negative,

    COUNTIF(
        predicted_fraud = 0
        AND actual_fraud = 1
    ) AS false_negative

FROM evaluated
"""

performance_job_config = (
    bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "model_name",
                "STRING",
                model_name,
            ),

            bigquery.ScalarQueryParameter(
                "model_version",
                "STRING",
                model_version,
            ),

            bigquery.ScalarQueryParameter(
                "run_id",
                "STRING",
                run_id,
            ),
        ]
    )
)

performance_rows = list(
    client.query(
        performance_query,
        job_config=performance_job_config,
    ).result()
)

metrics = performance_rows[0]


# ============================================================
# 4. EXTRACT CONFUSION MATRIX
# ============================================================

evaluated_cases = int(
    metrics.evaluated_cases
)

confirmed_fraud_cases = int(
    metrics.confirmed_fraud_cases
)

confirmed_normal_cases = int(
    metrics.confirmed_normal_cases
)

tp = int(
    metrics.true_positive
)

fp = int(
    metrics.false_positive
)

tn = int(
    metrics.true_negative
)

fn = int(
    metrics.false_negative
)


print(
    f"Completed investigation cases: "
    f"{evaluated_cases}"
)


# ============================================================
# 5. HANDLE INSUFFICIENT DATA
# ============================================================

if evaluated_cases < MIN_COMPLETED_CASES:

    monitoring_status = (
        "INSUFFICIENT_DATA"
    )

    alert_reason = (
        "Not enough completed investigation outcomes"
    )

    precision = None
    recall = None
    f1 = None
    fraud_yield = None

    baseline_precision = None
    baseline_recall = None
    baseline_f1 = None

    precision_change = None
    recall_change = None
    f1_change = None


# ============================================================
# 6. CALCULATE PRODUCTION PERFORMANCE
# ============================================================

else:

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    fraud_yield = (
        confirmed_fraud_cases
        / evaluated_cases
    )


    # ========================================================
    # 7. GET TRAINING BASELINE FOR SAME RUN
    #
    # Because provider_predictions preserves run_id,
    # use the SAME training run rather than merely finding
    # the latest XGBoost/RF metrics.
    # ========================================================

    baseline_query = f"""
    SELECT
        metric_name,
        metric_value

    FROM `{EXPERIMENT_METRICS_TABLE}`

    WHERE
        run_id = @run_id

        AND metric_name IN (
            'precision',
            'recall',
            'f1_score'
        )
    """

    baseline_job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "run_id",
                    "STRING",
                    run_id,
                )
            ]
        )
    )

    baseline_rows = list(
        client.query(
            baseline_query,
            job_config=baseline_job_config,
        ).result()
    )

    baseline_metrics = {
        row.metric_name: float(
            row.metric_value
        )
        for row in baseline_rows
    }

    baseline_precision = (
        baseline_metrics.get(
            "precision"
        )
    )

    baseline_recall = (
        baseline_metrics.get(
            "recall"
        )
    )

    baseline_f1 = (
        baseline_metrics.get(
            "f1_score"
        )
    )


    # ========================================================
    # 8. CALCULATE RELATIVE PERFORMANCE CHANGE
    #
    # Example:
    #
    # baseline F1 = .80
    # production F1 = .72
    #
    # change = (.72 - .80) / .80 = -10%
    # ========================================================

    def relative_change(
        current,
        baseline,
    ):

        if (
            baseline is None
            or baseline == 0
        ):
            return None

        return (
            current
            - baseline
        ) / baseline


    precision_change = (
        relative_change(
            precision,
            baseline_precision,
        )
    )

    recall_change = (
        relative_change(
            recall,
            baseline_recall,
        )
    )

    f1_change = (
        relative_change(
            f1,
            baseline_f1,
        )
    )


    # ========================================================
    # 9. DETERMINE PASS / ALERT
    # ========================================================

    alert_reasons = []

    if (
        precision_change is not None
        and precision_change
        <= -PRECISION_DEGRADATION_THRESHOLD
    ):

        alert_reasons.append(
            "precision_degradation"
        )

    if (
        recall_change is not None
        and recall_change
        <= -RECALL_DEGRADATION_THRESHOLD
    ):

        alert_reasons.append(
            "recall_degradation"
        )

    if (
        f1_change is not None
        and f1_change
        <= -F1_DEGRADATION_THRESHOLD
    ):

        alert_reasons.append(
            "f1_degradation"
        )


    # --------------------------------------------------------
    # If baseline is missing, do not incorrectly PASS.
    # --------------------------------------------------------

    if (
        baseline_precision is None
        or baseline_recall is None
        or baseline_f1 is None
    ):

        monitoring_status = (
            "INSUFFICIENT_BASELINE"
        )

        alert_reason = (
            "Training baseline metrics are missing"
        )

    elif alert_reasons:

        monitoring_status = (
            "ALERT"
        )

        alert_reason = (
            ",".join(
                alert_reasons
            )
        )

    else:

        monitoring_status = (
            "PASS"
        )

        alert_reason = None


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print(
    "\n=============================================="
)

print(
    "PERFORMANCE RESULTS"
)

print(
    "=============================================="
)

print(
    f"TP: {tp}"
)

print(
    f"FP: {fp}"
)

print(
    f"TN: {tn}"
)

print(
    f"FN: {fn}"
)

print()

print(
    f"Precision: "
    f"{precision}"
)

print(
    f"Recall: "
    f"{recall}"
)

print(
    f"F1: "
    f"{f1}"
)

print(
    f"Fraud Yield: "
    f"{fraud_yield}"
)

print(
    "\nTraining baseline:"
)

print(
    f"Precision: "
    f"{baseline_precision}"
)

print(
    f"Recall: "
    f"{baseline_recall}"
)

print(
    f"F1: "
    f"{baseline_f1}"
)

print(
    "\nRelative change:"
)

print(
    f"Precision: "
    f"{precision_change}"
)

print(
    f"Recall: "
    f"{recall_change}"
)

print(
    f"F1: "
    f"{f1_change}"
)

print()

print(
    f"Status: "
    f"{monitoring_status}"
)

print(
    f"Reason: "
    f"{alert_reason}"
)


# ============================================================
# 11. WRITE MONITORING RESULT
# ============================================================

monitoring_record = {
    "evaluation_timestamp": (
        evaluation_timestamp.isoformat()
    ),

    "evaluation_date": (
        evaluation_timestamp
        .date()
        .isoformat()
    ),

    "model_name": (
        model_name
    ),

    "model_version": (
        model_version
    ),

    "run_id": (
        run_id
    ),

    "evaluated_cases": (
        evaluated_cases
    ),

    "confirmed_fraud_cases": (
        confirmed_fraud_cases
    ),

    "confirmed_normal_cases": (
        confirmed_normal_cases
    ),

    "true_positive": (
        tp
    ),

    "false_positive": (
        fp
    ),

    "true_negative": (
        tn
    ),

    "false_negative": (
        fn
    ),

    "precision": (
        precision
    ),

    "recall": (
        recall
    ),

    "f1_score": (
        f1
    ),

    "fraud_yield": (
        fraud_yield
    ),

    "baseline_precision": (
        baseline_precision
    ),

    "baseline_recall": (
        baseline_recall
    ),

    "baseline_f1": (
        baseline_f1
    ),

    "precision_change": (
        precision_change
    ),

    "recall_change": (
        recall_change
    ),

    "f1_change": (
        f1_change
    ),

    "monitoring_status": (
        monitoring_status
    ),

    "alert_reason": (
        alert_reason
    ),
}


errors = client.insert_rows_json(
    PERFORMANCE_TABLE,
    [
        monitoring_record
    ],
)

if errors:
    raise RuntimeError(
        "Failed to write monitoring result: "
        f"{errors}"
    )


# ============================================================
# 12. COMPLETE
# ============================================================

print(
    "\n=============================================="
)

print(
    "PERFORMANCE MONITORING COMPLETE"
)

print(
    "=============================================="
)

print(
    f"Monitoring table: "
    f"{PERFORMANCE_TABLE}"
)
