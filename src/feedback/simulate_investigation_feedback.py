import uuid
from datetime import datetime, timezone

import numpy as np
from google.cloud import bigquery


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "fwa-mlops-accelerator-demo"

PREDICTIONS_TABLE = (
    f"{PROJECT_ID}."
    "fraud_experiments.provider_predictions"
)

CLAIMS_TABLE = (
    f"{PROJECT_ID}."
    "fwa_claims.claims"
)

FEEDBACK_TABLE = (
    f"{PROJECT_ID}."
    "fraud_feedback.investigation_results"
)

INVESTIGATION_CAPACITY = 50

RANDOM_SEED = 42


# ============================================================
# INITIALIZE
# ============================================================

client = bigquery.Client(
    project=PROJECT_ID
)

rng = np.random.default_rng(
    RANDOM_SEED
)

created_at = datetime.now(
    timezone.utc
)

investigation_date = (
    created_at.date()
)


# ============================================================
# 1. FIND LATEST PREDICTION RUN
#
# The investigation workflow should follow the newest
# prediction population written into provider_predictions.
#
# No model type is hard-coded here.
# ============================================================

latest_run_query = f"""
SELECT
    model_name,
    model_version,
    run_id,

    MAX(created_at)
        AS latest_prediction_timestamp

FROM `{PREDICTIONS_TABLE}`

WHERE
    run_id IS NOT NULL
    AND model_name IS NOT NULL
    AND model_version IS NOT NULL

GROUP BY
    model_name,
    model_version,
    run_id

ORDER BY
    latest_prediction_timestamp DESC

LIMIT 1
"""

latest_run_rows = list(
    client.query(
        latest_run_query
    ).result()
)

if not latest_run_rows:
    raise RuntimeError(
        "No provider predictions were found."
    )

latest_run = (
    latest_run_rows[0]
)

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
    "INVESTIGATION SOURCE"
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
# 2. SELECT PROVIDERS FOR INVESTIGATION
#
# Select the highest-risk providers from the exact:
#
#   model_name
#   model_version
#   run_id
#
# Exclude providers already investigated for this model/run.
# ============================================================

predictions_query = f"""
SELECT
    p.provider_id,

    p.run_id,
    p.model_name,
    p.model_version,

    p.fraud_score,
    p.risk_rank,
    p.risk_category

FROM `{PREDICTIONS_TABLE}` p

WHERE
    p.model_name = @model_name

    AND p.model_version = @model_version

    AND p.run_id = @run_id

    AND NOT EXISTS (

        SELECT
            1

        FROM `{FEEDBACK_TABLE}` f

        WHERE
            f.provider_id = p.provider_id

            AND f.run_id = p.run_id

            AND f.model_version = p.model_version
    )

ORDER BY
    p.risk_rank

LIMIT @investigation_capacity
"""

predictions_job_config = (
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

            bigquery.ScalarQueryParameter(
                "investigation_capacity",
                "INT64",
                INVESTIGATION_CAPACITY,
            ),
        ]
    )
)

review_df = (
    client.query(
        predictions_query,
        job_config=predictions_job_config,
    )
    .to_dataframe()
)

if review_df.empty:

    print(
        "\nNo new providers need investigation "
        "for this model/run."
    )

    raise SystemExit(
        0
    )


print(
    f"Providers selected for investigation: "
    f"{len(review_df)}"
)


# ============================================================
# 3. LOAD SYNTHETIC GROUND TRUTH
#
# DEMO ONLY
#
# In production, confirmed fraud would come from
# investigator adjudication / SIU / audit outcomes.
# ============================================================

truth_query = f"""
SELECT
    provider_id,

    MAX(
        fraud_provider_label
    ) AS synthetic_true_fraud

FROM `{CLAIMS_TABLE}`

GROUP BY
    provider_id
"""

truth_df = (
    client.query(
        truth_query
    )
    .to_dataframe()
)


# ============================================================
# 4. JOIN INVESTIGATION QUEUE TO GROUND TRUTH
# ============================================================

review_df = review_df.merge(
    truth_df,
    on="provider_id",
    how="left",
)

missing_truth = (
    review_df[
        "synthetic_true_fraud"
    ]
    .isnull()
    .sum()
)

if missing_truth > 0:
    raise RuntimeError(
        "Synthetic ground truth is missing for "
        f"{missing_truth} provider(s)."
    )


# ============================================================
# 5. SIMULATE INVESTIGATION DECISION
#
# Demo:
#
# synthetic_true_fraud = 1
#     -> CONFIRMED_FRAUD
#
# synthetic_true_fraud = 0
#     -> CONFIRMED_NORMAL
# ============================================================

review_df[
    "confirmed_fraud"
] = (
    review_df[
        "synthetic_true_fraud"
    ]
    .astype(bool)
)

review_df[
    "investigation_result"
] = np.where(
    review_df[
        "confirmed_fraud"
    ],
    "CONFIRMED_FRAUD",
    "CONFIRMED_NORMAL",
)


# ============================================================
# 6. SIMULATE FRAUD AMOUNT
#
# Only confirmed fraud gets a simulated dollar amount.
# ============================================================

review_df[
    "fraud_amount"
] = np.where(
    review_df[
        "confirmed_fraud"
    ],

    rng.uniform(
        10_000,
        150_000,
        size=len(
            review_df
        ),
    ),

    0.0,
)

review_df[
    "fraud_amount"
] = (
    review_df[
        "fraud_amount"
    ]
    .round(
        2
    )
)


# ============================================================
# 7. ADD INVESTIGATION METADATA
# ============================================================

review_df[
    "investigation_id"
] = [
    (
        "INV-"
        + uuid.uuid4()
        .hex[:10]
        .upper()
    )

    for _ in range(
        len(
            review_df
        )
    )
]

review_df[
    "investigation_date"
] = investigation_date

review_df[
    "investigator"
] = (
    "synthetic_investigation_team"
)

review_df[
    "created_at"
] = created_at


# ============================================================
# 8. PREPARE OUTPUT
#
# These fields preserve prediction-to-investigation lineage:
#
#   provider_id
#   run_id
#   model_name
#   model_version
# ============================================================

feedback_df = review_df[
    [
        "investigation_id",

        "provider_id",

        "run_id",
        "model_name",
        "model_version",

        "fraud_score",

        "investigation_result",
        "confirmed_fraud",
        "fraud_amount",

        "investigation_date",
        "investigator",
        "created_at",
    ]
].copy()


# ============================================================
# 9. DISPLAY SAMPLE RESULTS
# ============================================================

print(
    "\n=============================================="
)

print(
    "INVESTIGATION RESULTS"
)

print(
    "==============================================\n"
)

print(
    feedback_df[
        [
            "provider_id",
            "model_name",
            "model_version",
            "run_id",
            "fraud_score",
            "investigation_result",
            "fraud_amount",
        ]
    ]
    .sort_values(
        "fraud_score",
        ascending=False,
    )
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 10. SUMMARY
# ============================================================

confirmed_fraud_count = int(
    feedback_df[
        "confirmed_fraud"
    ]
    .sum()
)

confirmed_normal_count = (
    len(
        feedback_df
    )
    - confirmed_fraud_count
)

fraud_yield = (
    confirmed_fraud_count
    / len(
        feedback_df
    )
)

confirmed_fraud_amount = float(
    feedback_df[
        "fraud_amount"
    ]
    .sum()
)


print(
    "\n=============================================="
)

print(
    "INVESTIGATION SUMMARY"
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

print(
    f"Providers reviewed: "
    f"{len(feedback_df)}"
)

print(
    f"Confirmed fraud: "
    f"{confirmed_fraud_count}"
)

print(
    f"Confirmed normal: "
    f"{confirmed_normal_count}"
)

print(
    f"Fraud yield: "
    f"{fraud_yield:.2%}"
)

print(
    "Confirmed fraud amount: "
    f"${confirmed_fraud_amount:,.2f}"
)


# ============================================================
# 11. CONVERT TO BIGQUERY-COMPATIBLE JSON
#
# Avoid load_table_from_dataframe because pyarrow/pandas
# type conversion caused errors in Cloud Shell.
# ============================================================

rows_to_insert = []

for _, row in feedback_df.iterrows():

    record = {
        "investigation_id": str(
            row[
                "investigation_id"
            ]
        ),

        "provider_id": str(
            row[
                "provider_id"
            ]
        ),

        "run_id": str(
            row[
                "run_id"
            ]
        ),

        "model_name": str(
            row[
                "model_name"
            ]
        ),

        "model_version": str(
            row[
                "model_version"
            ]
        ),

        "fraud_score": float(
            row[
                "fraud_score"
            ]
        ),

        "investigation_result": str(
            row[
                "investigation_result"
            ]
        ),

        "confirmed_fraud": bool(
            row[
                "confirmed_fraud"
            ]
        ),

        "fraud_amount": float(
            row[
                "fraud_amount"
            ]
        ),

        "investigation_date": (
            row[
                "investigation_date"
            ]
            .isoformat()
        ),

        "investigator": str(
            row[
                "investigator"
            ]
        ),

        "created_at": (
            row[
                "created_at"
            ]
            .isoformat()
        ),
    }

    rows_to_insert.append(
        record
    )


# ============================================================
# 12. WRITE TO BIGQUERY
# ============================================================

print(
    "\nWriting investigation results "
    "to BigQuery..."
)

errors = client.insert_rows_json(
    FEEDBACK_TABLE,
    rows_to_insert,
)


if errors:
    raise RuntimeError(
        "Failed to write investigation feedback "
        f"to BigQuery: {errors}"
    )


print(
    f"Inserted {len(rows_to_insert)} "
    "investigation records."
)


# ============================================================
# 13. VERIFY WRITE
# ============================================================

verify_query = f"""
SELECT
    COUNT(*) AS investigation_count,

    COUNTIF(
        investigation_result = 'CONFIRMED_FRAUD'
    ) AS confirmed_fraud,

    COUNTIF(
        investigation_result = 'CONFIRMED_NORMAL'
    ) AS confirmed_normal

FROM `{FEEDBACK_TABLE}`

WHERE
    run_id = @run_id
    AND model_version = @model_version
"""

verify_job_config = (
    bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "run_id",
                "STRING",
                run_id,
            ),

            bigquery.ScalarQueryParameter(
                "model_version",
                "STRING",
                model_version,
            ),
        ]
    )
)

verify_rows = list(
    client.query(
        verify_query,
        job_config=verify_job_config,
    ).result()
)

verification = (
    verify_rows[0]
)


print(
    "\n=============================================="
)

print(
    "BIGQUERY WRITE VERIFICATION"
)

print(
    "=============================================="
)

print(
    f"Rows for run: "
    f"{verification.investigation_count}"
)

print(
    f"Confirmed fraud: "
    f"{verification.confirmed_fraud}"
)

print(
    f"Confirmed normal: "
    f"{verification.confirmed_normal}"
)


# ============================================================
# 14. COMPLETE
# ============================================================

print(
    "\n=============================================="
)

print(
    "INVESTIGATOR FEEDBACK SIMULATION COMPLETE"
)

print(
    "=============================================="
)

print(
    f"Feedback table: "
    f"{FEEDBACK_TABLE}"
)
