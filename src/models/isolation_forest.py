import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from google.cloud import bigquery

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "fwa-mlops-accelerator-demo"

FEATURE_TABLE = (
    f"{PROJECT_ID}."
    "fraud_features."
    "provider_eye_features"
)

EXPERIMENT_RUNS_TABLE = (
    f"{PROJECT_ID}."
    "fraud_experiments."
    "experiment_runs"
)

EXPERIMENT_METRICS_TABLE = (
    f"{PROJECT_ID}."
    "fraud_experiments."
    "experiment_metrics"
)

PREDICTIONS_TABLE = (
    f"{PROJECT_ID}."
    "fraud_experiments."
    "provider_predictions"
)


MODEL_NAME = "isolation_forest"

MODEL_VERSION = "isolation_forest_v1"

EXPERIMENT_NAME = (
    "provider_eye_fraud_isolation_forest"
)

FEATURE_VERSION = (
    "provider_eye_features_v1"
)

DATASET_VERSION = (
    "2026_train_test_v1"
)


# ============================================================
# FEATURES USED BY MODEL
#
# Important:
# identifiers and metadata are NOT model inputs.
# ============================================================

FEATURE_COLUMNS = [
    "high_acuity_pct",
    "pct_99214",
    "pct_99215",
    "referral_rate",
    "eye_procedure_pct",
    "avg_paid_per_claim",
    "high_acuity_oe_ratio"
]


# ============================================================
# INITIALIZE BIGQUERY CLIENT
# ============================================================

client = bigquery.Client(
    project=PROJECT_ID
)


# ============================================================
# CREATE EXPERIMENT / RUN IDS
# ============================================================

experiment_id = (
    "if_provider_eye_fraud"
)

run_id = (
    datetime.now(timezone.utc)
    .strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(uuid.uuid4())[:8]
)


created_at = datetime.now(
    timezone.utc
)


print(
    f"\nExperiment ID: {experiment_id}"
)

print(
    f"Run ID: {run_id}"
)


# ============================================================
# 1. LOAD TRAIN AND TEST FEATURE SNAPSHOTS
# ============================================================

query = f"""
SELECT
    provider_id,
    provider_city,
    dataset_split,
    feature_timestamp,

    high_acuity_pct,
    pct_99214,
    pct_99215,
    referral_rate,
    eye_procedure_pct,
    avg_paid_per_claim,
    high_acuity_oe_ratio

FROM
    `{FEATURE_TABLE}`

ORDER BY
    provider_id,
    feature_timestamp
"""


feature_df = (
    client
    .query(query)
    .to_dataframe()
)


print(
    f"\nFeature rows loaded: "
    f"{len(feature_df):,}"
)


train_df = (
    feature_df[
        feature_df[
            "dataset_split"
        ] == "TRAIN"
    ]
    .copy()
)


test_df = (
    feature_df[
        feature_df[
            "dataset_split"
        ] == "TEST"
    ]
    .copy()
)


print(
    f"TRAIN providers: "
    f"{len(train_df):,}"
)

print(
    f"TEST providers: "
    f"{len(test_df):,}"
)


# ============================================================
# 2. VALIDATE FEATURE DATA
# ============================================================

if train_df.empty:
    raise ValueError(
        "TRAIN feature dataset is empty."
    )


if test_df.empty:
    raise ValueError(
        "TEST feature dataset is empty."
    )


missing_train = (
    train_df[
        FEATURE_COLUMNS
    ]
    .isnull()
    .sum()
    .sum()
)


missing_test = (
    test_df[
        FEATURE_COLUMNS
    ]
    .isnull()
    .sum()
    .sum()
)


if missing_train > 0:
    raise ValueError(
        "Missing values detected "
        "in TRAIN features."
    )


if missing_test > 0:
    raise ValueError(
        "Missing values detected "
        "in TEST features."
    )


# ============================================================
# 3. PREPARE TRAIN / TEST MATRICES
# ============================================================

X_train = (
    train_df[
        FEATURE_COLUMNS
    ]
    .astype(float)
)


X_test = (
    test_df[
        FEATURE_COLUMNS
    ]
    .astype(float)
)


# ============================================================
# 4. FIT SCALER ON TRAIN ONLY
#
# This prevents information from the TEST period
# influencing preprocessing.
# ============================================================

scaler = StandardScaler()


X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)


X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ============================================================
# 5. TRAIN ISOLATION FOREST
# ============================================================

model = IsolationForest(

    n_estimators=300,

    contamination=0.10,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train_scaled
)


print(
    "\nIsolation Forest training complete."
)


# ============================================================
# 6. SCORE TEST PROVIDERS
#
# score_samples():
# lower raw score = more anomalous
#
# We negate it so:
# higher risk score = more suspicious
# ============================================================

test_df[
    "fraud_score"
] = (
    -model.score_samples(
        X_test_scaled
    )
)


# Isolation Forest prediction:
#
# -1 = anomaly
#  1 = normal

test_df[
    "suspicious_flag"
] = (
    model.predict(
        X_test_scaled
    )
    == -1
).astype(int)


# ============================================================
# 7. CREATE RISK RANK
# ============================================================

test_df[
    "risk_rank"
] = (
    test_df[
        "fraud_score"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


test_df = (
    test_df
    .sort_values(
        "fraud_score",
        ascending=False
    )
)


# ============================================================
# 8. CREATE RISK CATEGORY
#
# This is operational prioritization,
# NOT a confirmed fraud label.
# ============================================================

def assign_risk_category(row):

    if row["suspicious_flag"] == 1:
        return "HIGH"

    if row["risk_rank"] <= 50:
        return "MEDIUM"

    return "LOW"


test_df[
    "risk_category"
] = (
    test_df.apply(
        assign_risk_category,
        axis=1
    )
)


# ============================================================
# 9. CREATE SIMPLE REASON CODES
#
# These are descriptive reasons that help investigators
# understand why a provider appears unusual.
# ============================================================

train_means = (
    train_df[
        FEATURE_COLUMNS
    ]
    .mean()
)


def create_reason_codes(row):

    deviations = {}

    for feature in FEATURE_COLUMNS:

        baseline = (
            train_means[
                feature
            ]
        )

        if baseline == 0:
            deviation = 0

        else:
            deviation = (
                row[
                    feature
                ]
                /
                baseline
            )

        deviations[
            feature
        ] = deviation


    ranked_features = (
        sorted(
            deviations.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


    reasons = [
        item[0]
        for item in ranked_features[:3]
    ]


    return pd.Series(
        reasons,
        index=[
            "reason_1",
            "reason_2",
            "reason_3"
        ]
    )


test_df[
    [
        "reason_1",
        "reason_2",
        "reason_3"
    ]
] = (
    test_df.apply(
        create_reason_codes,
        axis=1
    )
)


# ============================================================
# 10. DISPLAY TOP 20 TEST PROVIDERS
# ============================================================

print(
    "\n=============================================="
)

print(
    "Top 20 TEST Providers - Isolation Forest"
)

print(
    "==============================================\n"
)


display_columns = [

    "risk_rank",

    "provider_id",

    "provider_city",

    "fraud_score",

    "suspicious_flag",

    "risk_category",

    "high_acuity_pct",

    "pct_99215",

    "referral_rate",

    "high_acuity_oe_ratio",

    "reason_1",

    "reason_2",

    "reason_3"
]


print(
    test_df[
        display_columns
    ]
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. CALCULATE UNSUPERVISED METRICS
#
# We do not yet have investigator-confirmed fraud labels.
# Therefore we should NOT calculate:
#
# precision
# recall
# ROC-AUC
# PR-AUC
#
# Those come later after feedback is available.
# ============================================================

provider_count = (
    len(test_df)
)


flagged_count = (
    int(
        test_df[
            "suspicious_flag"
        ]
        .sum()
    )
)


alert_rate = (
    flagged_count
    /
    provider_count
)


mean_risk_score = (
    float(
        test_df[
            "fraud_score"
        ]
        .mean()
    )
)


p95_risk_score = (
    float(
        test_df[
            "fraud_score"
        ]
        .quantile(
            0.95
        )
    )
)


max_risk_score = (
    float(
        test_df[
            "fraud_score"
        ]
        .max()
    )
)


print(
    "\n=============================================="
)

print(
    "Experiment Metrics"
)

print(
    "=============================================="
)


print(
    f"\nProviders scored: "
    f"{provider_count}"
)


print(
    f"Providers flagged: "
    f"{flagged_count}"
)


print(
    f"Alert rate: "
    f"{alert_rate:.2%}"
)


print(
    f"Mean risk score: "
    f"{mean_risk_score:.4f}"
)


print(
    f"P95 risk score: "
    f"{p95_risk_score:.4f}"
)


# ============================================================
# 12. WRITE EXPERIMENT RUN METADATA
# ============================================================

experiment_run_df = (
    pd.DataFrame(
        [
            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "experiment_name":
                    EXPERIMENT_NAME,

                "owner":
                    "fraud_data_science_team",

                "model_type":
                    MODEL_NAME,

                "feature_version":
                    FEATURE_VERSION,

                "dataset_version":
                    DATASET_VERSION,

                "git_commit":
                    "local-development",

                "training_start":
                    pd.to_datetime(
                        "2026-01-01"
                    ).date(),

                "training_end":
                    pd.to_datetime(
                        "2026-06-30"
                    ).date(),

                "status":
                    "COMPLETED",

                "created_at":
                    created_at
            }
        ]
    )
)


# ============================================================
# 13. WRITE EXPERIMENT METRICS
# ============================================================

metrics_df = (
    pd.DataFrame(
        [
            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "metric_name":
                    "provider_count",

                "metric_value":
                    float(
                        provider_count
                    ),

                "created_at":
                    created_at
            },

            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "metric_name":
                    "flagged_provider_count",

                "metric_value":
                    float(
                        flagged_count
                    ),

                "created_at":
                    created_at
            },

            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "metric_name":
                    "alert_rate",

                "metric_value":
                    float(
                        alert_rate
                    ),

                "created_at":
                    created_at
            },

            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "metric_name":
                    "mean_risk_score",

                "metric_value":
                    mean_risk_score,

                "created_at":
                    created_at
            },

            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "metric_name":
                    "p95_risk_score",

                "metric_value":
                    p95_risk_score,

                "created_at":
                    created_at
            },

            {
                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "metric_name":
                    "max_risk_score",

                "metric_value":
                    max_risk_score,

                "created_at":
                    created_at
            }
        ]
    )
)


# ============================================================
# 14. CREATE PROVIDER PREDICTION OUTPUT
# ============================================================

prediction_date = (
    pd.to_datetime(
        "2026-09-30"
    )
    .date()
)


predictions_df = pd.DataFrame({

    "prediction_date":
        prediction_date,

    "provider_id":
        test_df[
            "provider_id"
        ],

    "experiment_id":
        experiment_id,

    "run_id":
        run_id,

    "model_name":
        MODEL_NAME,

    "model_version":
        MODEL_VERSION,

    "fraud_score":
        test_df[
            "fraud_score"
        ],

    "risk_rank":
        test_df[
            "risk_rank"
        ],

    "risk_category":
        test_df[
            "risk_category"
        ],

    "reason_1":
        test_df[
            "reason_1"
        ],

    "reason_2":
        test_df[
            "reason_2"
        ],

    "reason_3":
        test_df[
            "reason_3"
        ],

    "created_at":
        created_at
})


# ============================================================
# 15. APPEND RESULTS TO BIGQUERY
# ============================================================

print(
    "\nWriting experiment metadata "
    "to BigQuery..."
)


client.load_table_from_dataframe(

    experiment_run_df,

    EXPERIMENT_RUNS_TABLE,

    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND"
    )

).result()


print(
    "Writing experiment metrics "
    "to BigQuery..."
)


client.load_table_from_dataframe(

    metrics_df,

    EXPERIMENT_METRICS_TABLE,

    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND"
    )

).result()


print(
    "Writing provider predictions "
    "to BigQuery..."
)


client.load_table_from_dataframe(

    predictions_df,

    PREDICTIONS_TABLE,

    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND"
    )

).result()


print(
    "\n=============================================="
)

print(
    "Isolation Forest Experiment Complete"
)

print(
    "=============================================="
)


print(
    f"\nRun ID: {run_id}"
)


print(
    f"Predictions written: "
    f"{len(predictions_df):,}"
)
