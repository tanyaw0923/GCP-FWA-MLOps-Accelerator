import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from google.cloud import bigquery

from src.config.load_config import load_config
from src.utils.observability import (
    configure_observability,
    log_event,
)


MIN_COMPLETED_CASES = 20

PRECISION_DEGRADATION_THRESHOLD = 0.10
RECALL_DEGRADATION_THRESHOLD = 0.15
F1_DEGRADATION_THRESHOLD = 0.10


config = load_config()

PROJECT_ID = config["project_id"]

logger = configure_observability(
    project_id=PROJECT_ID,
    service_name="fwa-performance-monitoring",
)

PREDICTIONS_TABLE = (
    config["data"]["provider_prediction_table"]
)

FEEDBACK_TABLE = (
    config["data"]["feedback_table"]
)

EXPERIMENT_METRICS_TABLE = (
    config["data"]["experiment_metrics_table"]
)

PERFORMANCE_TABLE = (
    config["monitoring"][
        "performance_degradation_table"
    ]
)

TRAINING_STATE_TABLE = (
    config["monitoring"][
        "training_state_table"
    ]
)

STATE_KEY = "provider_fraud_model"


client = bigquery.Client(
    project=PROJECT_ID
)

evaluation_timestamp = datetime.now(
    timezone.utc
)


def relative_change(current, baseline):
    if baseline is None or baseline == 0:
        return None

    return (
        current - baseline
    ) / baseline


def point_change(current, baseline):
    if current is None or baseline is None:
        return None

    return current - baseline


def format_percent(value):
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def format_percentage_points(value):
    if value is None:
        return "N/A"

    return f"{value * 100:+.2f} percentage points"


def format_relative_change(value):
    if value is None:
        return "N/A"

    return f"{value:+.2%}"


def format_number(value):
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def format_currency(value):
    if value is None:
        return "$0.00"

    return f"${value:,.2f}"


def send_monitoring_email(
    subject,
    body,
):
    sender = os.environ.get(
        "EMAIL_ADDRESS"
    )

    password = os.environ.get(
        "EMAIL_PASSWORD"
    )

    recipient = os.environ.get(
        "MONITORING_EMAIL"
    )

    if not sender or not password or not recipient:
        print(
            "\nEmail not sent because email "
            "environment variables are not configured."
        )
        return

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30,
        ) as smtp:

            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()

            smtp.login(
                sender,
                password,
            )

            smtp.send_message(
                message
            )

        log_event(
            logger,
            "monitoring_email_sent",
            fields={
                "recipient_configured": True,
                "subject": subject,
            },
        )

        print(
            f"Monitoring email sent to "
            f"{recipient}"
        )

    except Exception as exc:
        log_event(
            logger,
            "monitoring_email_failed",
            severity="ERROR",
            fields={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

        print(
            "\nEmail sending failed."
        )

        print(
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: "
            f"{exc}"
        )

        raise


# Create the monitoring table if it does not exist.
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


# Resolve the model that is actually approved for production.
#
# Monitoring must follow production state, not whichever model happens
# to have the newest investigator feedback. This makes the monitor
# algorithm-agnostic: Random Forest or XGBoost can be production.
client.query(
    f"""
    ALTER TABLE `{TRAINING_STATE_TABLE}`
    ADD COLUMN IF NOT EXISTS approved_model_resource STRING
    """
).result()

production_state_query = f"""
SELECT
    approved_model_type,
    approved_run_id,
    approved_model_resource
FROM `{TRAINING_STATE_TABLE}`
WHERE model_name = @state_key
LIMIT 1
"""

production_state_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter(
            "state_key",
            "STRING",
            STATE_KEY,
        )
    ]
)

production_state_rows = list(
    client.query(
        production_state_query,
        job_config=production_state_config,
    ).result()
)

if not production_state_rows:
    raise RuntimeError(
        "No production training state was found for "
        f"state_key={STATE_KEY}."
    )

production_state = production_state_rows[0]

model_name = production_state.approved_model_type
run_id = production_state.approved_run_id
approved_model_resource = (
    production_state.approved_model_resource
)

if not model_name or not run_id:
    raise RuntimeError(
        "Production model is not fully approved. "
        "approved_model_type and approved_run_id are required."
    )


# Resolve the exact scored model version for the approved run.
#
# New promotions persist approved_model_resource. For older state rows
# that predate that column being populated, fall back to the scored
# prediction/feedback lineage for the approved model and run.
approved_model_id = None

if approved_model_resource:
    approved_model_id = (
        approved_model_resource
        .rstrip("/")
        .split("/")[-1]
    )

expected_model_version = (
    f"{model_name}:{approved_model_id}"
    if approved_model_id
    else None
)

model_version_query = f"""
SELECT
    p.model_version,
    MAX(f.created_at) AS latest_feedback_timestamp
FROM `{PREDICTIONS_TABLE}` p
INNER JOIN `{FEEDBACK_TABLE}` f
    ON p.provider_id = f.provider_id
    AND p.run_id = f.run_id
    AND p.model_version = f.model_version
WHERE
    p.model_name = @model_name
    AND p.run_id = @run_id
    AND f.model_name = @model_name
    AND f.run_id = @run_id
    AND f.investigation_result IN (
        'CONFIRMED_FRAUD',
        'CONFIRMED_NORMAL'
    )
    AND (
        @expected_model_version IS NULL
        OR p.model_version = @expected_model_version
    )
GROUP BY
    p.model_version
ORDER BY
    latest_feedback_timestamp DESC
LIMIT 1
"""

model_version_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter(
            "model_name",
            "STRING",
            model_name,
        ),
        bigquery.ScalarQueryParameter(
            "run_id",
            "STRING",
            run_id,
        ),
        bigquery.ScalarQueryParameter(
            "expected_model_version",
            "STRING",
            expected_model_version,
        ),
    ]
)

model_version_rows = list(
    client.query(
        model_version_query,
        job_config=model_version_config,
    ).result()
)

if not model_version_rows:
    raise RuntimeError(
        "No completed production investigation results were found "
        f"for approved model_type={model_name}, run_id={run_id}"
        + (
            f", model_version={expected_model_version}."
            if expected_model_version
            else "."
        )
    )

model_version = (
    model_version_rows[0].model_version
)


print("=" * 60)
print("PRODUCTION PERFORMANCE MONITORING")
print("=" * 60)

print(
    f"Approved production model: {model_name}"
)

print(
    f"Approved model resource: "
    f"{approved_model_resource or 'legacy state / inferred from lineage'}"
)

print(
    f"Model version: {model_version}"
)

print(
    f"Run ID: {run_id}"
)

log_event(
    logger,
    "performance_monitoring_started",
    fields={
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
    },
)


query_parameters = [
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

job_config = bigquery.QueryJobConfig(
    query_parameters=query_parameters
)


# Join predictions to completed investigator outcomes.
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
    COUNT(*) AS evaluated_cases,

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

performance_rows = list(
    client.query(
        performance_query,
        job_config=job_config,
    ).result()
)

metrics = performance_rows[0]

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


# Build the investigator summary for the email.
investigation_summary_query = f"""
SELECT
    COUNT(*) AS providers_reviewed,

    COUNTIF(
        confirmed_fraud
    ) AS confirmed_fraud,

    COUNTIF(
        NOT confirmed_fraud
    ) AS confirmed_normal,

    SAFE_DIVIDE(
        COUNTIF(confirmed_fraud),
        COUNT(*)
    ) AS fraud_yield,

    SUM(
        CASE
            WHEN confirmed_fraud
                THEN COALESCE(fraud_amount, 0)
            ELSE 0
        END
    ) AS confirmed_fraud_amount

FROM `{FEEDBACK_TABLE}`

WHERE
    model_name = @model_name

    AND model_version = @model_version

    AND run_id = @run_id

    AND investigation_result IN (
        'CONFIRMED_FRAUD',
        'CONFIRMED_NORMAL'
    )
"""

investigation_rows = list(
    client.query(
        investigation_summary_query,
        job_config=job_config,
    ).result()
)

investigation = investigation_rows[0]

providers_reviewed = int(
    investigation.providers_reviewed
)

investigator_confirmed_fraud = int(
    investigation.confirmed_fraud
)

investigator_confirmed_normal = int(
    investigation.confirmed_normal
)

investigator_fraud_yield = (
    float(
        investigation.fraud_yield
    )
    if investigation.fraud_yield is not None
    else 0.0
)

confirmed_fraud_amount = (
    float(
        investigation.confirmed_fraud_amount
    )
    if investigation.confirmed_fraud_amount is not None
    else 0.0
)


print(
    f"Completed investigation cases: "
    f"{evaluated_cases}"
)


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

    precision_point_change = None
    recall_point_change = None
    f1_point_change = None

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

    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:
        f1 = 0.0

    fraud_yield = (
        confirmed_fraud_cases
        / evaluated_cases
    )


    # Pull training metrics from the same experiment run.
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


    # Absolute change in percentage points.
    precision_point_change = point_change(
        precision,
        baseline_precision,
    )

    recall_point_change = point_change(
        recall,
        baseline_recall,
    )

    f1_point_change = point_change(
        f1,
        baseline_f1,
    )


    # Relative change versus the training baseline.
    precision_change = relative_change(
        precision,
        baseline_precision,
    )

    recall_change = relative_change(
        recall,
        baseline_recall,
    )

    f1_change = relative_change(
        f1,
        baseline_f1,
    )


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

        alert_reason = ",".join(
            alert_reasons
        )

    else:
        monitoring_status = (
            "PASS"
        )

        alert_reason = None


print("\n" + "=" * 60)
print("INVESTIGATION SUMMARY")
print("=" * 60)

print(
    f"Providers reviewed: "
    f"{providers_reviewed}"
)

print(
    f"Confirmed fraud: "
    f"{investigator_confirmed_fraud}"
)

print(
    f"Confirmed normal: "
    f"{investigator_confirmed_normal}"
)

print(
    f"Fraud yield: "
    f"{format_percent(investigator_fraud_yield)}"
)

print(
    f"Confirmed fraud amount: "
    f"{format_currency(confirmed_fraud_amount)}"
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Evaluated cases: "
    f"{evaluated_cases}"
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

print(
    f"\nPrecision: "
    f"{format_percent(precision)}"
)

print(
    f"Recall: "
    f"{format_percent(recall)}"
)

print(
    f"F1: "
    f"{format_percent(f1)}"
)


print("\nTraining baseline")

print(
    f"Precision: "
    f"{format_percent(baseline_precision)}"
)

print(
    f"Recall: "
    f"{format_percent(baseline_recall)}"
)

print(
    f"F1: "
    f"{format_percent(baseline_f1)}"
)


print("\nAbsolute change")

print(
    f"Precision: "
    f"{format_percentage_points(precision_point_change)}"
)

print(
    f"Recall: "
    f"{format_percentage_points(recall_point_change)}"
)

print(
    f"F1: "
    f"{format_percentage_points(f1_point_change)}"
)


print("\nRelative change")

print(
    f"Precision: "
    f"{format_relative_change(precision_change)}"
)

print(
    f"Recall: "
    f"{format_relative_change(recall_change)}"
)

print(
    f"F1: "
    f"{format_relative_change(f1_change)}"
)


print(
    f"\nStatus: "
    f"{monitoring_status}"
)

print(
    f"Reason: "
    f"{alert_reason}"
)


# Keep the existing BigQuery schema unchanged.
# precision_change, recall_change, and f1_change continue to
# represent relative changes.
monitoring_record = {
    "evaluation_timestamp": (
        evaluation_timestamp.isoformat()
    ),

    "evaluation_date": (
        evaluation_timestamp
        .date()
        .isoformat()
    ),

    "model_name": model_name,

    "model_version": model_version,

    "run_id": run_id,

    "evaluated_cases": (
        evaluated_cases
    ),

    "confirmed_fraud_cases": (
        confirmed_fraud_cases
    ),

    "confirmed_normal_cases": (
        confirmed_normal_cases
    ),

    "true_positive": tp,

    "false_positive": fp,

    "true_negative": tn,

    "false_negative": fn,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "fraud_yield": fraud_yield,

    "baseline_precision": (
        baseline_precision
    ),

    "baseline_recall": (
        baseline_recall
    ),

    "baseline_f1": baseline_f1,

    "precision_change": (
        precision_change
    ),

    "recall_change": (
        recall_change
    ),

    "f1_change": f1_change,

    "monitoring_status": (
        monitoring_status
    ),

    "alert_reason": alert_reason,
}


errors = client.insert_rows_json(
    PERFORMANCE_TABLE,
    [
        monitoring_record
    ],
)

if errors:
    log_event(
        logger,
        "performance_monitoring_write_failed",
        severity="ERROR",
        fields={
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "errors": str(errors),
        },
    )

    raise RuntimeError(
        "Failed to write monitoring result: "
        f"{errors}"
    )


if monitoring_status == "ALERT":
    action_message = (
        "Performance degradation was detected.\n\n"
        "The daily continuous-training pipeline will read "
        "this monitoring result and determine whether a new "
        "candidate model should be trained."
    )

elif monitoring_status == "PASS":
    action_message = (
        "No performance degradation was detected. "
        "No retraining action is required based on model "
        "performance."
    )

elif monitoring_status == "INSUFFICIENT_DATA":
    action_message = (
        "There are not enough completed investigations to "
        "evaluate production performance reliably."
    )

else:
    action_message = (
        "Training baseline metrics were not available, so "
        "performance degradation could not be evaluated."
    )


subject = (
    "FWA Model Performance Monitoring - "
    f"{monitoring_status}"
)


email_body = f"""
FWA MODEL PERFORMANCE MONITORING

Evaluation time:
{evaluation_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}


INVESTIGATION SUMMARY
------------------------------------------------------------

Model: {model_name}
Model version: {model_version}
Run ID: {run_id}

Providers reviewed: {providers_reviewed}
Confirmed fraud: {investigator_confirmed_fraud}
Confirmed normal: {investigator_confirmed_normal}
Fraud yield: {format_percent(investigator_fraud_yield)}
Confirmed fraud amount: {format_currency(confirmed_fraud_amount)}


MODEL PERFORMANCE
------------------------------------------------------------

Evaluated cases: {evaluated_cases}

True positives: {tp}
False positives: {fp}
True negatives: {tn}
False negatives: {fn}

Precision: {format_percent(precision)}
Recall: {format_percent(recall)}
F1 score: {format_percent(f1)}


TRAINING BASELINE
------------------------------------------------------------

Precision: {format_percent(baseline_precision)}
Recall: {format_percent(baseline_recall)}
F1 score: {format_percent(baseline_f1)}


PERFORMANCE CHANGE
------------------------------------------------------------

Precision
  Baseline: {format_percent(baseline_precision)}
  Current: {format_percent(precision)}
  Absolute change: {format_percentage_points(precision_point_change)}
  Relative change: {format_relative_change(precision_change)}

Recall
  Baseline: {format_percent(baseline_recall)}
  Current: {format_percent(recall)}
  Absolute change: {format_percentage_points(recall_point_change)}
  Relative change: {format_relative_change(recall_change)}

F1
  Baseline: {format_percent(baseline_f1)}
  Current: {format_percent(f1)}
  Absolute change: {format_percentage_points(f1_point_change)}
  Relative change: {format_relative_change(f1_change)}


MONITORING RESULT
------------------------------------------------------------

Status: {monitoring_status}
Reason: {alert_reason or "None"}


NEXT ACTION
------------------------------------------------------------

{action_message}


Monitoring table:
{PERFORMANCE_TABLE}
""".strip()


send_monitoring_email(
    subject=subject,
    body=email_body,
)


print("\n" + "=" * 60)
print("PERFORMANCE MONITORING COMPLETE")
print("=" * 60)

print(
    f"Monitoring table: "
    f"{PERFORMANCE_TABLE}"
)

log_event(
    logger,
    "performance_monitoring_completed",
    severity=(
        "WARNING"
        if monitoring_status == "ALERT"
        else "INFO"
    ),
    fields={
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
        "evaluated_cases": evaluated_cases,
        "confirmed_fraud_cases": (
            confirmed_fraud_cases
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fraud_yield": fraud_yield,
        "monitoring_status": monitoring_status,
        "alert_reason": alert_reason,
        "performance_table": PERFORMANCE_TABLE,
    },
)
