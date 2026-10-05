import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# 1. CONFIGURATION
# ============================================================

np.random.seed(42)

N_CLAIMS = 100_000
N_PROVIDERS = 200
N_MEMBERS = 20_000
N_SUSPICIOUS_PROVIDERS = 20

OUTPUT_FILE = "synthetic_claims.csv"

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 9, 30)

TRAIN_END_DATE = datetime(2026, 6, 30)


# ============================================================
# 2. TEXAS GEOGRAPHY
# ============================================================

texas_cities = [
    "Houston",
    "Dallas",
    "Austin",
    "San Antonio",
    "Fort Worth",
    "El Paso"
]

city_probabilities = [
    0.25,
    0.20,
    0.15,
    0.18,
    0.15,
    0.07
]


# ============================================================
# 3. CREATE PRIMARY CARE PROVIDERS
# ============================================================

providers = pd.DataFrame({
    "provider_id": [
        f"P{str(i).zfill(5)}"
        for i in range(1, N_PROVIDERS + 1)
    ],

    "provider_name": [
        f"Primary_Care_Provider_{i}"
        for i in range(1, N_PROVIDERS + 1)
    ],

    "provider_specialty": "Primary Care",

    "provider_state": "TX",

    "provider_city": np.random.choice(
        texas_cities,
        size=N_PROVIDERS,
        p=city_probabilities
    )
})


# ============================================================
# 4. SELECT SYNTHETIC SUSPICIOUS PROVIDERS
# ============================================================

suspicious_provider_ids = set(
    np.random.choice(
        providers["provider_id"],
        size=N_SUSPICIOUS_PROVIDERS,
        replace=False
    )
)

providers["fraud_provider_label"] = (
    providers["provider_id"]
    .isin(suspicious_provider_ids)
    .astype(int)
)


# ============================================================
# 5. PROVIDER BEHAVIOR PARAMETERS
#
# These are used only to generate synthetic behavior.
# They will be dropped before saving the final claim file.
# ============================================================

providers["high_acuity_rate_param"] = 0.0
providers["referral_rate_param"] = 0.0
providers["eye_procedure_rate_param"] = 0.0
providers["pct_99215_given_high_acuity_param"] = 0.0


for idx, row in providers.iterrows():

    suspicious = (
        row["provider_id"]
        in suspicious_provider_ids
    )

    if suspicious:

        # Main anomaly signal
        providers.at[
            idx,
            "high_acuity_rate_param"
        ] = np.random.uniform(
            0.65,
            0.80
        )

        # Moderately higher referral rate
        providers.at[
            idx,
            "referral_rate_param"
        ] = np.random.uniform(
            0.38,
            0.45
        )

        # Similar eye procedure utilization
        providers.at[
            idx,
            "eye_procedure_rate_param"
        ] = np.random.uniform(
            0.08,
            0.12
        )

        # More 99215 usage within high-acuity claims
        providers.at[
            idx,
            "pct_99215_given_high_acuity_param"
        ] = np.random.uniform(
            0.35,
            0.50
        )

    else:

        providers.at[
            idx,
            "high_acuity_rate_param"
        ] = np.random.uniform(
            0.18,
            0.25
        )

        providers.at[
            idx,
            "referral_rate_param"
        ] = np.random.uniform(
            0.27,
            0.33
        )

        providers.at[
            idx,
            "eye_procedure_rate_param"
        ] = np.random.uniform(
            0.08,
            0.12
        )

        providers.at[
            idx,
            "pct_99215_given_high_acuity_param"
        ] = np.random.uniform(
            0.08,
            0.15
        )


# ============================================================
# 6. MEMBERS
# ============================================================

members = [
    f"M{str(i).zfill(6)}"
    for i in range(1, N_MEMBERS + 1)
]


# ============================================================
# 7. ALL DIAGNOSES ARE EYE / ADNEXA RELATED
# ============================================================

eye_icd_codes = [
    "H10.9",
    "H25.9",
    "H40.9",
    "H35.30",
    "H53.8",
    "H57.9"
]


# ============================================================
# 8. CPT CODES
# ============================================================

routine_em_codes = [
    "99211",
    "99212",
    "99213"
]

routine_em_probs = [
    0.10,
    0.20,
    0.70
]

high_acuity_em_codes = [
    "99214",
    "99215"
]

eye_procedure_codes = [
    "92002",
    "92004",
    "92012",
    "92014",
    "92083",
    "92133",
    "92134",
    "92250"
]


# ============================================================
# 9. PLACE OF SERVICE
# ============================================================

place_of_service_codes = [
    "11",
    "21",
    "22",
    "23"
]


# ============================================================
# 10. CLAIM VOLUME
#
# Uniform provider weights so suspicious providers
# are not detectable merely by claim volume.
# ============================================================

provider_choices = (
    providers["provider_id"]
    .values
)

provider_weights = np.ones(
    N_PROVIDERS
)

provider_weights = (
    provider_weights
    / provider_weights.sum()
)

chosen_providers = np.random.choice(
    provider_choices,
    size=N_CLAIMS,
    p=provider_weights
)


# ============================================================
# 11. BASE CLAIM DATA
# ============================================================

date_range_days = (
    END_DATE - START_DATE
).days + 1


df = pd.DataFrame({

    "claim_id": [
        f"C{str(i).zfill(8)}"
        for i in range(1, N_CLAIMS + 1)
    ],

    "provider_id": chosen_providers,

    "member_id": np.random.choice(
        members,
        N_CLAIMS
    ),

    "service_date": [
        START_DATE + timedelta(days=int(x))
        for x in np.random.randint(
            0,
            date_range_days,
            N_CLAIMS
        )
    ],

    "place_of_service": np.random.choice(
        place_of_service_codes,
        N_CLAIMS
    )
})


# ============================================================
# 12. ADD PROVIDER PROFILE
# ============================================================

df = df.merge(
    providers,
    on="provider_id",
    how="left"
)

df["service_city"] = (
    df["provider_city"]
)

df["service_state"] = (
    df["provider_state"]
)


# ============================================================
# 13. GENERATE ICD + CPT
# ============================================================

def generate_claim_codes(row):

    # Every claim is eye/adnexa related
    icd_code = np.random.choice(
        eye_icd_codes
    )

    high_acuity_rate = (
        row["high_acuity_rate_param"]
    )

    eye_procedure_rate = (
        row["eye_procedure_rate_param"]
    )

    pct_99215 = (
        row[
            "pct_99215_given_high_acuity_param"
        ]
    )

    r = np.random.rand()

    # Eye procedure
    if r < eye_procedure_rate:

        cpt_code = np.random.choice(
            eye_procedure_codes
        )

    # High-acuity E/M
    elif r < (
        eye_procedure_rate
        + high_acuity_rate
    ):

        cpt_code = np.random.choice(
            high_acuity_em_codes,
            p=[
                1 - pct_99215,
                pct_99215
            ]
        )

    # Routine E/M
    else:

        cpt_code = np.random.choice(
            routine_em_codes,
            p=routine_em_probs
        )

    return cpt_code, icd_code


codes = df.apply(
    generate_claim_codes,
    axis=1
)

df["cpt_code"] = [
    x[0]
    for x in codes
]

df["icd_code"] = [
    x[1]
    for x in codes
]

df["diagnoses_code"] = (
    df["icd_code"]
)


# ============================================================
# 14. CLAIM-LEVEL FLAGS
# ============================================================

df["is_eye_diagnosis"] = 1


df["is_high_acuity_em"] = (
    df["cpt_code"]
    .isin(high_acuity_em_codes)
    .astype(int)
)


df["is_99214"] = (
    df["cpt_code"]
    .eq("99214")
    .astype(int)
)


df["is_99215"] = (
    df["cpt_code"]
    .eq("99215")
    .astype(int)
)


df["is_eye_procedure"] = (
    df["cpt_code"]
    .isin(eye_procedure_codes)
    .astype(int)
)


# ============================================================
# 15. REFERRAL BEHAVIOR
# ============================================================

df["is_referral"] = (
    np.random.rand(
        len(df)
    )
    <
    df["referral_rate_param"]
).astype(int)


df["referral_specialty"] = (
    np.where(
        df["is_referral"] == 1,
        "Ophthalmology",
        None
    )
)


# ============================================================
# 16. FINANCIAL AMOUNTS
#
# Amount depends on CPT, not directly on fraud label.
# ============================================================

base_billed = (
    np.random.gamma(
        shape=2.5,
        scale=100,
        size=len(df)
    )
    + 40
)

df["total_billed_amt"] = (
    base_billed
)


df.loc[
    df["cpt_code"] == "99214",
    "total_billed_amt"
] *= 1.30


df.loc[
    df["cpt_code"] == "99215",
    "total_billed_amt"
] *= 1.55


df.loc[
    df["is_eye_procedure"] == 1,
    "total_billed_amt"
] *= 1.40


df["total_allowed_amt"] = (
    df["total_billed_amt"]
    *
    np.random.uniform(
        0.60,
        0.90,
        len(df)
    )
)


df["total_paid_amount"] = (
    df["total_allowed_amt"]
    *
    np.random.uniform(
        0.85,
        1.00,
        len(df)
    )
)


# ============================================================
# 17. CLAIM DATE + TEMPORAL TRAIN / TEST
# ============================================================

df["service_date"] = (
    pd.to_datetime(
        df["service_date"]
    )
)


df["claim_date"] = (
    df["service_date"]
    +
    pd.to_timedelta(
        np.random.randint(
            0,
            15,
            len(df)
        ),
        unit="D"
    )
)


df["dataset_split"] = (
    np.where(
        df["service_date"]
        <= pd.Timestamp(
            TRAIN_END_DATE
        ),
        "TRAIN",
        "TEST"
    )
)


# ============================================================
# 18. ROUND FINANCIAL FIELDS
# ============================================================

for col in [
    "total_billed_amt",
    "total_allowed_amt",
    "total_paid_amount"
]:

    df[col] = (
        df[col]
        .round(2)
    )


# ============================================================
# 19. DROP INTERNAL GENERATION PARAMETERS
# ============================================================

df = df.drop(
    columns=[
        "high_acuity_rate_param",
        "referral_rate_param",
        "eye_procedure_rate_param",
        "pct_99215_given_high_acuity_param"
    ]
)


# ============================================================
# 20. FINAL COLUMN ORDER
# ============================================================

df = df[
    [
        "claim_id",
        "claim_date",

        "provider_id",
        "provider_name",
        "provider_specialty",
        "provider_city",
        "provider_state",

        "member_id",

        "service_date",
        "service_city",
        "service_state",

        "cpt_code",
        "icd_code",
        "diagnoses_code",

        "total_billed_amt",
        "total_allowed_amt",
        "total_paid_amount",

        "place_of_service",

        "is_eye_diagnosis",
        "is_high_acuity_em",
        "is_99214",
        "is_99215",
        "is_eye_procedure",

        "is_referral",
        "referral_specialty",

        # Evaluation only.
        # NEVER use this as an Isolation Forest feature.
        "fraud_provider_label",

        "dataset_split"
    ]
]


# ============================================================
# 21. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 22. VALIDATION OUTPUT
# ============================================================

print(
    "\n=============================================="
)

print(
    "Synthetic Texas Eye-Related PCP Claims"
)

print(
    "=============================================="
)

print(
    f"\nClaims: {len(df):,}"
)

print(
    f"Providers: "
    f"{df['provider_id'].nunique():,}"
)

print(
    f"Members: "
    f"{df['member_id'].nunique():,}"
)

print(
    f"Suspicious providers: "
    f"{df.loc[df['fraud_provider_label'] == 1, 'provider_id'].nunique():,}"
)

print(
    "\nState:"
)

print(
    df["provider_state"]
    .value_counts()
)

print(
    "\nTrain/Test:"
)

print(
    df["dataset_split"]
    .value_counts()
)

print(
    "\nDate range:"
)

print(
    df["service_date"].min().date(),
    "to",
    df["service_date"].max().date()
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

