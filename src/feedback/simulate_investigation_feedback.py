import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery


# ============================================================
# Configuration
# ============================================================

PROJECT_ID = "fwa-mlops-accelerator-demo"

PREDICTIONS_TABLE = (
    f"{PROJECT_ID}.fraud_experiments.provider_predictions"
)

CLAIMS_TABLE = (
    f"{PROJECT_ID}.fwa_claims.claims"
)

FEEDBACK_TABLE = (
    f"{PROJECT_ID}.fraud_feedback.investigation_results"
)

MODEL_NAME = "isolation_forest"

# Number of highest-ranked providers investigators review
INVESTIGATION_CAPACITY = 50

RANDOM_SEED = 42


# ============================================================
# Initialize
# ============================================================

client = bigquery.Client(project=PROJECT_ID)
rng = np.random.default_rng(RANDOM_SEED)

created_at = datetime.now(timezone.utc)
investigation_date = pd.to_datetime("2026-10-15").date()


# ============================================================
# 1. Find latest Isolation Forest run
# ============================================================

latest_run_query = f"""
SELECT run_id
FROM `{PREDICTIONS_TABLE}`
WHERE model_name = '{MODEL_NAME}'
GROUP BY run_id
ORDER BY MAX(created_at) DESC
LIMIT 1
"""

latest_run_df = client.query(latest_run_query).to_dataframe()

if latest_run_df.empty:
    raise ValueError("No Isolation Forest predictions found.")

run_id = latest_run_df.iloc[0]["run_id"]

print(f"\nUsing Isolation Forest run: {run_id}")


# ============================================================
# 2. Load highest-risk providers
# ============================================================

predictions_query = f"""
SELECT
    provider_id,
    model_version,
    fraud_score,
    risk_rank,
    risk_category
FROM `{PREDICTIONS_TABLE}`
WHERE
    model_name = '{MODEL_NAME}'
    AND run_id = '{run_id}'
ORDER BY risk_rank
LIMIT {INVESTIGATION_CAPACITY}
"""

review_df = client.query(predictions_query).to_dataframe()

print(f"Providers selected for investigation: {len(review_df)}")


# ============================================================
# 3. Load hidden synthetic ground truth
#
# IMPORTANT:
# This is used ONLY to simulate investigator findings.
# It is never used by the Isolation Forest model.
# ============================================================

truth_query = f"""
SELECT
    provider_id,
    MAX(fraud_provider_label) AS synthetic_true_fraud
FROM `{CLAIMS_TABLE}`
GROUP BY provider_id
"""

truth_df = client.query(truth_query).to_dataframe()


# ============================================================
# 4. Join investigation queue to hidden truth
# ============================================================

review_df = review_df.merge(
    truth_df,
    on="provider_id",
    how="left",
)

if review_df["synthetic_true_fraud"].isnull().any():
    raise ValueError("Missing synthetic truth for some providers.")


# ============================================================
# 5. Simulate investigator decisions
#
# For this demo:
# hidden synthetic fraud = investigator confirms fraud
# hidden synthetic normal = investigator confirms normal
#
# Later we can make this more realistic by adding
# inconclusive / pending investigations.
# ============================================================

review_df["confirmed_fraud"] = (
    review_df["synthetic_true_fraud"]
    .astype(bool)
)

review_df["investigation_result"] = np.where(
    review_df["confirmed_fraud"],
    "CONFIRMED_FRAUD",
    "CONFIRMED_NORMAL",
)


# ============================================================
# 6. Simulate fraud amount
#
# Only confirmed-fraud providers receive a fraud amount.
# This represents an illustrative investigation finding.
# ============================================================

review_df["fraud_amount"] = np.where(
    review_df["confirmed_fraud"],
    rng.uniform(10_000, 150_000, size=len(review_df)),
    0.0,
)

review_df["fraud_amount"] = (
    review_df["fraud_amount"]
    .round(2)
)


# ============================================================
# 7. Create investigation metadata
# ============================================================

review_df["investigation_id"] = [
    f"INV-{uuid.uuid4().hex[:10].upper()}"
    for _ in range(len(review_df))
]

review_df["investigation_date"] = investigation_date
review_df["investigator"] = "synthetic_investigation_team"
review_df["created_at"] = created_at


# ============================================================
# 8. Prepare BigQuery output
# ============================================================

feedback_df = review_df[
    [
        "investigation_id",
        "provider_id",
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
# 9. Display investigation results
# ============================================================

print("\n==============================================")
print("Investigation Results")
print("==============================================\n")

display(
    feedback_df
    .sort_values("fraud_score", ascending=False)
    .head(20)
    .style.hide(axis="index")
)


# ============================================================
# 10. Summary
# ============================================================

confirmed_fraud = int(feedback_df["confirmed_fraud"].sum())
confirmed_normal = len(feedback_df) - confirmed_fraud

fraud_rate = confirmed_fraud / len(feedback_df)

print("\n==============================================")
print("Investigation Summary")
print("==============================================")

print(f"\nProviders reviewed: {len(feedback_df)}")
print(f"Confirmed fraud: {confirmed_fraud}")
print(f"Confirmed normal: {confirmed_normal}")
print(f"Fraud yield: {fraud_rate:.2%}")
print(
    f"Confirmed fraud amount: "
    f"${feedback_df['fraud_amount'].sum():,.2f}"
)


# ============================================================
# 11. Write results to BigQuery
# ============================================================

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND"
)

print("\nWriting investigation results to BigQuery...")

client.load_table_from_dataframe(
    feedback_df,
    FEEDBACK_TABLE,
    job_config=job_config,
).result()

print("\n==============================================")
print("Investigator Feedback Simulation Complete")
print("==============================================")
