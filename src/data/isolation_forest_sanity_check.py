import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "synthetic_claims.csv"


# ============================================================
# 1. LOAD CLAIM DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# 2. AGGREGATE CLAIMS TO PROVIDER LEVEL
# ============================================================

provider_df = (
    df
    .groupby(
        [
            "provider_id",
            "provider_city"
        ]
    )
    .agg(
        total_claims=(
            "claim_id",
            "count"
        ),

        unique_members=(
            "member_id",
            "nunique"
        ),

        high_acuity_pct=(
            "is_high_acuity_em",
            "mean"
        ),

        pct_99214=(
            "is_99214",
            "mean"
        ),

        pct_99215=(
            "is_99215",
            "mean"
        ),

        referral_rate=(
            "is_referral",
            "mean"
        ),

        eye_procedure_pct=(
            "is_eye_procedure",
            "mean"
        ),

        avg_paid_per_claim=(
            "total_paid_amount",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 3. FEATURES USED BY ISOLATION FOREST
# ============================================================

feature_columns = [
    "high_acuity_pct",
    "pct_99214",
    "pct_99215",
    "referral_rate",
    "eye_procedure_pct",
    "avg_paid_per_claim"
]


X = provider_df[
    feature_columns
]


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 5. TRAIN ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=300,

    # Synthetic demo assumes roughly
    # 10% anomalous providers
    contamination=0.10,

    random_state=42
)


model.fit(
    X_scaled
)


# ============================================================
# 6. CREATE SUSPICIOUS RISK SCORE
#
# sklearn score_samples:
# lower = more anomalous
#
# Negate the score so:
# higher score = more suspicious
# ============================================================

provider_df[
    "suspicious_risk_score"
] = (
    -model.score_samples(
        X_scaled
    )
)


# ============================================================
# 7. CREATE SUSPICIOUS FLAG
#
# Isolation Forest prediction:
# -1 = anomaly
#  1 = normal
# ============================================================

provider_df[
    "suspicious_flag"
] = (
    model.predict(
        X_scaled
    )
    == -1
).astype(int)


# ============================================================
# 8. CREATE RISK RANK
# ============================================================

provider_df[
    "risk_rank"
] = (
    provider_df[
        "suspicious_risk_score"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


provider_df = (
    provider_df
    .sort_values(
        "suspicious_risk_score",
        ascending=False
    )
)


# ============================================================
# 9. DISPLAY TOP 20 SUSPICIOUS PROVIDERS
# ============================================================

print("\n==============================================")
print("Top 20 Providers - Isolation Forest")
print("==============================================\n")


display_columns = [
    "risk_rank",
    "provider_id",
    "provider_city",

    "total_claims",
    "unique_members",

    "high_acuity_pct",
    "pct_99214",
    "pct_99215",

    "referral_rate",
    "eye_procedure_pct",

    "avg_paid_per_claim",

    "suspicious_risk_score",
    "suspicious_flag"
]


print(
    provider_df[
        display_columns
    ]
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("\n==============================================")
print("Isolation Forest Summary")
print("==============================================")


print(
    f"\nProviders scored: "
    f"{len(provider_df):,}"
)


print(
    f"Providers flagged as suspicious: "
    f"{provider_df['suspicious_flag'].sum():,}"
)


print(
    "\nSuspicious risk score distribution:"
)


print(
    provider_df[
        "suspicious_risk_score"
    ]
    .describe()
    .round(4)
    .to_string()
)


# ============================================================
# 11. FEATURE DISTRIBUTION
# ============================================================

print("\n==============================================")
print("Provider Feature Distribution")
print("==============================================")


print(
    provider_df[
        feature_columns
    ]
    .describe()
    .round(4)
    .to_string()
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    "isolation_forest_sanity_results.csv"
)


provider_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================================")
print("Sanity Check Complete")
print("==============================================")


print(
    f"\nSaved results to: "
    f"{OUTPUT_FILE}"
)
