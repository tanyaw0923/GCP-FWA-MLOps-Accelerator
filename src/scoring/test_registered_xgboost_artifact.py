import xgboost as xgb

from google.cloud import bigquery
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "fwa-mlops-accelerator-demo"

MODEL_PATH = "/tmp/model.bst"

FEATURES = [
    "high_acuity_pct",
    "pct_99214",
    "pct_99215",
    "referral_rate",
    "eye_procedure_pct",
    "avg_paid_per_claim",
    "high_acuity_oe_ratio",
]

CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# LOAD EXACT TEST DATA
# ============================================================

bq = bigquery.Client(
    project=PROJECT_ID
)

feature_select = ",\n    ".join(
    FEATURES
)

query = f"""
SELECT
    provider_id,
    CAST(fraud_label AS INT64) AS fraud_label,
    {feature_select}

FROM
    `{PROJECT_ID}.fraud_features.provider_eye_features`

WHERE
    dataset_split = 'TEST'

ORDER BY
    provider_id
"""

df = (
    bq.query(
        query
    )
    .to_dataframe()
)

if df.empty:
    raise RuntimeError(
        "TEST dataset is empty."
    )

X_test = df[
    FEATURES
]

y_test = df[
    "fraud_label"
]


# ============================================================
# LOAD EXACT REGISTERED model.bst
# ============================================================

booster = xgb.Booster()

booster.load_model(
    MODEL_PATH
)


# ============================================================
# SCORE
# ============================================================

dmatrix = xgb.DMatrix(
    X_test,
    feature_names=FEATURES,
)

probabilities = booster.predict(
    dmatrix
)

predictions = (
    probabilities
    >= CLASSIFICATION_THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

predicted_fraud = int(
    predictions.sum()
)

predicted_normal = (
    len(predictions)
    - predicted_fraud
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n=============================================="
)

print(
    "REGISTERED XGBOOST ARTIFACT TEST"
)

print(
    "=============================================="
)

print(
    f"TEST providers: "
    f"{len(df)}"
)

print(
    f"Actual fraud: "
    f"{int(y_test.sum())}"
)

print(
    f"Predicted fraud: "
    f"{predicted_fraud}"
)

print(
    f"Predicted normal: "
    f"{predicted_normal}"
)

print()

print(
    f"Min probability: "
    f"{probabilities.min():.6f}"
)

print(
    f"Max probability: "
    f"{probabilities.max():.6f}"
)

print(
    f"Mean probability: "
    f"{probabilities.mean():.6f}"
)

print()

print(
    f"Precision: "
    f"{precision:.6f}"
)

print(
    f"Recall: "
    f"{recall:.6f}"
)

print(
    f"F1: "
    f"{f1:.6f}"
)


# ============================================================
# TOP SCORES
# ============================================================

result_df = df[
    [
        "provider_id",
        "fraud_label",
    ]
].copy()

result_df[
    "fraud_probability"
] = probabilities

result_df[
    "prediction"
] = predictions

print(
    "\nTop 20 fraud probabilities:"
)

print(
    result_df
    .sort_values(
        "fraud_probability",
        ascending=False,
    )
    .head(20)
    .to_string(
        index=False
    )
)
