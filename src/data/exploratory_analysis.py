import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "synthetic_claims.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print("\n==============================================")
print("Exploratory Analysis - Synthetic Claims")
print("==============================================")


print(
    f"\nTotal claims: {len(df):,}"
)

print(
    f"Total providers: "
    f"{df['provider_id'].nunique():,}"
)

print(
    f"Total members: "
    f"{df['member_id'].nunique():,}"
)


# ============================================================
# 2. SHOW TOP 10 CLAIM ROWS
# ============================================================

print("\n==============================================")
print("Top 10 Rows of Synthetic Claim Data")
print("==============================================")


display_columns = [
    "claim_id",
    "claim_date",

    "provider_id",
    "provider_specialty",
    "provider_city",
    "provider_state",

    "member_id",
    "service_date",

    "cpt_code",
    "icd_code",

    "total_billed_amt",
    "total_allowed_amt",
    "total_paid_amount",

    "is_eye_diagnosis",
    "is_high_acuity_em",
    "is_99214",
    "is_99215",
    "is_eye_procedure",

    "is_referral"
]


display_columns = [
    col
    for col in display_columns
    if col in df.columns
]


print(
    df[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 3. BASIC DATA QUALITY CHECKS
# ============================================================

print("\n==============================================")
print("Basic Data Quality Checks")
print("==============================================")


# ------------------------------------------------------------
# Duplicate claim IDs
# ------------------------------------------------------------

duplicate_claims = (
    df["claim_id"]
    .duplicated()
    .sum()
)


print(
    f"\nDuplicate claim IDs: "
    f"{duplicate_claims:,}"
)


# ------------------------------------------------------------
# Missing value counts
# ------------------------------------------------------------

print("\nMissing Value Counts:")


missing_count = (
    df
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


missing_count_nonzero = (
    missing_count[
        missing_count > 0
    ]
)


if len(missing_count_nonzero) == 0:

    print(
        "No missing values found."
    )

else:

    print(
        missing_count_nonzero
        .to_string()
    )


# ------------------------------------------------------------
# Missing value percentages
# ------------------------------------------------------------

print("\nMissing Value Percentage:")


missing_pct = (
    df
    .isnull()
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)


missing_pct_nonzero = (
    missing_pct[
        missing_pct > 0
    ]
)


if len(missing_pct_nonzero) == 0:

    print(
        "No missing values found."
    )

else:

    print(
        missing_pct_nonzero
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )


# ------------------------------------------------------------
# Total rows / columns
# ------------------------------------------------------------

print(
    f"\nDataset shape: "
    f"{df.shape[0]:,} rows x "
    f"{df.shape[1]:,} columns"
)


# ============================================================
# 4. VERIFY PRIMARY CARE ONLY
# ============================================================

print("\n==============================================")
print("Provider Specialty Validation")
print("==============================================")


specialty_counts = (
    df[
        "provider_specialty"
    ]
    .value_counts()
)


print(
    specialty_counts
    .to_string()
)


unique_specialties = (
    df[
        "provider_specialty"
    ]
    .nunique()
)


if (
    unique_specialties == 1
    and
    df[
        "provider_specialty"
    ]
    .iloc[0]
    == "Primary Care"
):

    print(
        "\nPASS: All providers are Primary Care."
    )

else:

    print(
        "\nWARNING: Dataset contains "
        "non-Primary-Care providers."
    )


# ============================================================
# 5. VERIFY TEXAS-ONLY POPULATION
# ============================================================

print("\n==============================================")
print("Texas Geography Validation")
print("==============================================")


print(
    "\nProvider states:"
)


state_counts = (
    df[
        "provider_state"
    ]
    .value_counts()
)


print(
    state_counts
    .to_string()
)


print(
    "\nProviders by Texas city:"
)


provider_city_summary = (
    df[
        [
            "provider_id",
            "provider_city"
        ]
    ]
    .drop_duplicates()
    ["provider_city"]
    .value_counts()
)


print(
    provider_city_summary
    .to_string()
)


if (
    df[
        "provider_state"
    ]
    .nunique() == 1
    and
    df[
        "provider_state"
    ]
    .iloc[0] == "TX"
):

    print(
        "\nPASS: All providers are located in Texas."
    )

else:

    print(
        "\nWARNING: Dataset contains providers "
        "outside Texas."
    )


# ============================================================
# 6. VERIFY ALL CLAIMS ARE EYE RELATED
# ============================================================

print("\n==============================================")
print("Eye Diagnosis Validation")
print("==============================================")


eye_rate = (
    df[
        "is_eye_diagnosis"
    ]
    .mean()
)


print(
    f"\nEye-related claim rate: "
    f"{eye_rate:.2%}"
)


print(
    "\nICD code distribution:"
)


icd_distribution = (
    df[
        "icd_code"
    ]
    .value_counts()
)


print(
    icd_distribution
    .to_string()
)


print(
    "\nICD first-character distribution:"
)


icd_prefix_distribution = (
    df[
        "icd_code"
    ]
    .astype(str)
    .str[0]
    .value_counts()
)


print(
    icd_prefix_distribution
    .to_string()
)


if eye_rate == 1.0:

    print(
        "\nPASS: All claims are eye related."
    )

else:

    print(
        "\nWARNING: Some claims are not marked "
        "as eye related."
    )


# ============================================================
# 7. CPT CODE DISTRIBUTION
# ============================================================

print("\n==============================================")
print("CPT Code Distribution")
print("==============================================")


cpt_distribution = (
    df[
        "cpt_code"
    ]
    .value_counts()
    .to_frame(
        name="claim_count"
    )
)


cpt_distribution[
    "claim_pct"
] = (
    cpt_distribution[
        "claim_count"
    ]
    /
    len(df)
    * 100
)


print(
    cpt_distribution
    .round(2)
    .to_string()
)


# ============================================================
# 8. OVERALL CLAIM-LEVEL BEHAVIOR
# ============================================================

print("\n==============================================")
print("Overall Claim-Level Metrics")
print("==============================================")


overall_metrics = pd.DataFrame({

    "metric": [
        "High Acuity Rate",
        "99214 Rate",
        "99215 Rate",
        "Referral Rate",
        "Eye Procedure Rate"
    ],

    "value_pct": [
        df[
            "is_high_acuity_em"
        ].mean() * 100,

        df[
            "is_99214"
        ].mean() * 100,

        df[
            "is_99215"
        ].mean() * 100,

        df[
            "is_referral"
        ].mean() * 100,

        df[
            "is_eye_procedure"
        ].mean() * 100
    ]
})


print(
    overall_metrics
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 9. TEMPORAL TRAIN / TEST VALIDATION
# ============================================================

print("\n==============================================")
print("Train / Test Distribution")
print("==============================================")


split_summary = (
    df
    .groupby(
        "dataset_split"
    )
    .agg(

        claims=(
            "claim_id",
            "count"
        ),

        providers=(
            "provider_id",
            "nunique"
        ),

        members=(
            "member_id",
            "nunique"
        ),

        min_service_date=(
            "service_date",
            "min"
        ),

        max_service_date=(
            "service_date",
            "max"
        )
    )
)


print(
    split_summary
    .to_string()
)


# ============================================================
# 10. BUILD TEMPORARY PROVIDER-LEVEL SUMMARY
#
# Exploratory only.
# Step 4 will create the production feature table.
# ============================================================

print("\n==============================================")
print("Provider-Level Exploratory Summary")
print("==============================================")


provider_summary = (
    df
    .groupby(
        [
            "provider_id",
            "provider_city",
            "provider_state"
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

        high_acuity_claims=(
            "is_high_acuity_em",
            "sum"
        ),

        high_acuity_pct=(
            "is_high_acuity_em",
            "mean"
        ),

        claims_99214=(
            "is_99214",
            "sum"
        ),

        pct_99214=(
            "is_99214",
            "mean"
        ),

        claims_99215=(
            "is_99215",
            "sum"
        ),

        pct_99215=(
            "is_99215",
            "mean"
        ),

        referral_count=(
            "is_referral",
            "sum"
        ),

        referral_rate=(
            "is_referral",
            "mean"
        ),

        eye_procedure_claims=(
            "is_eye_procedure",
            "sum"
        ),

        eye_procedure_pct=(
            "is_eye_procedure",
            "mean"
        ),

        avg_billed_per_claim=(
            "total_billed_amt",
            "mean"
        ),

        avg_allowed_per_claim=(
            "total_allowed_amt",
            "mean"
        ),

        avg_paid_per_claim=(
            "total_paid_amount",
            "mean"
        ),

        total_paid=(
            "total_paid_amount",
            "sum"
        )
    )
    .reset_index()
)


print(
    f"\nProvider-level rows: "
    f"{len(provider_summary):,}"
)


# ============================================================
# 11. TOP 20 PROVIDERS BY HIGH-ACUITY RATE
#
# This is only an exploratory ranking.
# It is NOT a fraud label.
# ============================================================

print("\n==============================================")
print("Top 20 Providers by High-Acuity Percentage")
print("==============================================")


top_high_acuity = (
    provider_summary
    .sort_values(
        "high_acuity_pct",
        ascending=False
    )
    .head(20)
    .copy()
)


percentage_columns = [
    "high_acuity_pct",
    "pct_99214",
    "pct_99215",
    "referral_rate",
    "eye_procedure_pct"
]


top_high_acuity[
    percentage_columns
] = (
    top_high_acuity[
        percentage_columns
    ]
    * 100
)


print(
    top_high_acuity[
        [
            "provider_id",
            "provider_city",

            "total_claims",
            "unique_members",

            "high_acuity_claims",
            "high_acuity_pct",

            "claims_99214",
            "pct_99214",

            "claims_99215",
            "pct_99215",

            "referral_rate",
            "eye_procedure_pct",

            "avg_paid_per_claim"
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. PROVIDER FEATURE DISTRIBUTION
#
# These distributions help us understand what the
# Isolation Forest will see later.
# ============================================================

print("\n==============================================")
print("Provider Feature Distribution")
print("==============================================")


provider_features = [
    "total_claims",
    "unique_members",

    "high_acuity_pct",
    "pct_99214",
    "pct_99215",

    "referral_rate",
    "eye_procedure_pct",

    "avg_billed_per_claim",
    "avg_allowed_per_claim",
    "avg_paid_per_claim",

    "total_paid"
]


print(
    provider_summary[
        provider_features
    ]
    .describe()
    .round(4)
    .to_string()
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("Exploratory Analysis Complete")
print("==============================================")
