import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config.load_config import load_config


# ============================================================
# 1. CONFIGURATION
# ============================================================

config = load_config()

np.random.seed(42)

N_CLAIMS = 100_000
N_PROVIDERS = 200
N_MEMBERS = 20_000
N_SUSPICIOUS_PROVIDERS = 20

OUTPUT_FILE = "synthetic_claims.csv"

CURRENT_DATA_MONTH = str(
    config["data"]["current_data_month"]
)

TRAINING_MONTHS = int(
    config["data"]["rolling_window"]["training_months"]
)

TESTING_MONTHS = int(
    config["data"]["rolling_window"]["testing_months"]
)


# ============================================================
# 2. DATE HELPERS
# ============================================================

def add_months(date_value, months):
    month_index = (
        date_value.year * 12
        + date_value.month
        - 1
        + months
    )

    year = month_index // 12
    month = month_index % 12 + 1

    return datetime(
        year,
        month,
        1,
    )


def get_month_end(date_value):
    last_day = calendar.monthrange(
        date_value.year,
        date_value.month,
    )[1]

    return datetime(
        date_value.year,
        date_value.month,
        last_day,
    )


def calculate_raw_data_window(
    current_data_month,
    training_months,
    testing_months,
):
    current_month = datetime.strptime(
        current_data_month,
        "%Y-%m-%d",
    )

    total_months = (
        training_months
        + testing_months
    )

    start_date = add_months(
        current_month,
        -(total_months - 1),
    )

    end_date = get_month_end(
        current_month
    )

    return (
        start_date,
        end_date,
    )


START_DATE, END_DATE = (
    calculate_raw_data_window(
        CURRENT_DATA_MONTH,
        TRAINING_MONTHS,
        TESTING_MONTHS,
    )
)


# ============================================================
# 3. TEXAS GEOGRAPHY
# ============================================================

texas_cities = [
    "Houston",
    "Dallas",
    "Austin",
    "San Antonio",
    "Fort Worth",
    "El Paso",
]

city_probabilities = [
    0.25,
    0.20,
    0.15,
    0.18,
    0.15,
    0.07,
]


# ============================================================
# 4. CREATE PRIMARY CARE PROVIDERS
# ============================================================

providers = pd.DataFrame({
    "provider_id": [
        f"P{str(i).zfill(5)}"
        for i in range(
            1,
            N_PROVIDERS + 1,
        )
    ],

    "provider_name": [
        f"Primary_Care_Provider_{i}"
        for i in range(
            1,
            N_PROVIDERS + 1,
        )
    ],

    "provider_specialty": "Primary Care",

    "provider_state": "TX",

    "provider_city": np.random.choice(
        texas_cities,
        size=N_PROVIDERS,
        p=city_probabilities,
    ),
})


# ============================================================
# 5. SELECT SYNTHETIC SUSPICIOUS PROVIDERS
# ============================================================

suspicious_provider_ids = set(
    np.random.choice(
        providers["provider_id"],
        size=N_SUSPICIOUS_PROVIDERS,
        replace=False,
    )
)

providers["fraud_provider_label"] = (
    providers["provider_id"]
    .isin(suspicious_provider_ids)
    .astype(int)
)


# ============================================================
# 6. PROVIDER BEHAVIOR PARAMETERS
#
# These fields exist only to generate synthetic behavior.
# They are removed before the final claim file is saved.
# ============================================================

providers["high_acuity_rate_param"] = 0.0
providers["referral_rate_param"] = 0.0
providers["eye_procedure_rate_param"] = 0.0
providers["pct_99215_given_high_acuity_param"] = 0.0
providers["member_provider_distance_mean_param"] = 0.0


for idx, row in providers.iterrows():
    suspicious = (
        row["provider_id"]
        in suspicious_provider_ids
    )

    if suspicious:
        providers.at[
            idx,
            "high_acuity_rate_param",
        ] = np.random.uniform(
            0.65,
            0.80,
        )

        providers.at[
            idx,
            "referral_rate_param",
        ] = np.random.uniform(
            0.38,
            0.45,
        )

        providers.at[
            idx,
            "eye_procedure_rate_param",
        ] = np.random.uniform(
            0.08,
            0.12,
        )

        providers.at[
            idx,
            "pct_99215_given_high_acuity_param",
        ] = np.random.uniform(
            0.35,
            0.50,
        )

        providers.at[
            idx,
            "member_provider_distance_mean_param",
        ] = np.random.uniform(
            40,
            70,
        )

    else:
        providers.at[
            idx,
            "high_acuity_rate_param",
        ] = np.random.uniform(
            0.18,
            0.25,
        )

        providers.at[
            idx,
            "referral_rate_param",
        ] = np.random.uniform(
            0.27,
            0.33,
        )

        providers.at[
            idx,
            "eye_procedure_rate_param",
        ] = np.random.uniform(
            0.08,
            0.12,
        )

        providers.at[
            idx,
            "pct_99215_given_high_acuity_param",
        ] = np.random.uniform(
            0.08,
            0.15,
        )

        providers.at[
            idx,
            "member_provider_distance_mean_param",
        ] = np.random.uniform(
            10,
            30,
        )


# ============================================================
# 7. MEMBERS
# ============================================================

members = [
    f"M{str(i).zfill(6)}"
    for i in range(
        1,
        N_MEMBERS + 1,
    )
]


# ============================================================
# 8. EYE-RELATED DIAGNOSES
# ============================================================

eye_icd_codes = [
    "H10.9",
    "H25.9",
    "H40.9",
    "H35.30",
    "H53.8",
    "H57.9",
]


# ============================================================
# 9. CPT CODES
# ============================================================

routine_em_codes = [
    "99211",
    "99212",
    "99213",
]

routine_em_probs = [
    0.10,
    0.20,
    0.70,
]

high_acuity_em_codes = [
    "99214",
    "99215",
]

eye_procedure_codes = [
    "92002",
    "92004",
    "92012",
    "92014",
    "92083",
    "92133",
    "92134",
    "92250",
]


# ============================================================
# 10. PLACE OF SERVICE
# ============================================================

place_of_service_codes = [
    "11",
    "21",
    "22",
    "23",
]


# ============================================================
# 11. CLAIM VOLUME
#
# Uniform provider probability prevents fraud labels from being
# identifiable purely from claim volume.
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
    p=provider_weights,
)


# ============================================================
# 12. BASE CLAIM DATA
# ============================================================

date_range_days = (
    END_DATE - START_DATE
).days + 1

df = pd.DataFrame({
    "claim_id": [
        f"C{str(i).zfill(8)}"
        for i in range(
            1,
            N_CLAIMS + 1,
        )
    ],

    "provider_id": chosen_providers,

    "member_id": np.random.choice(
        members,
        N_CLAIMS,
    ),

    "service_date": [
        START_DATE
        + timedelta(days=int(x))
        for x in np.random.randint(
            0,
            date_range_days,
            N_CLAIMS,
        )
    ],

    "place_of_service": np.random.choice(
        place_of_service_codes,
        N_CLAIMS,
    ),
})


# ============================================================
# 13. ADD PROVIDER PROFILE
# ============================================================

df = df.merge(
    providers,
    on="provider_id",
    how="left",
)

df["service_city"] = (
    df["provider_city"]
)

df["service_state"] = (
    df["provider_state"]
)


# ============================================================
# 14. GENERATE ICD + CPT
# ============================================================

def generate_claim_codes(row):
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

    random_value = np.random.rand()

    if random_value < eye_procedure_rate:
        cpt_code = np.random.choice(
            eye_procedure_codes
        )

    elif random_value < (
        eye_procedure_rate
        + high_acuity_rate
    ):
        cpt_code = np.random.choice(
            high_acuity_em_codes,
            p=[
                1 - pct_99215,
                pct_99215,
            ],
        )

    else:
        cpt_code = np.random.choice(
            routine_em_codes,
            p=routine_em_probs,
        )

    return (
        cpt_code,
        icd_code,
    )


codes = df.apply(
    generate_claim_codes,
    axis=1,
)

df["cpt_code"] = [
    value[0]
    for value in codes
]

df["icd_code"] = [
    value[1]
    for value in codes
]

df["diagnoses_code"] = (
    df["icd_code"]
)


# ============================================================
# 15. CLAIM-LEVEL FLAGS
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
# 16. REFERRAL BEHAVIOR
# ============================================================

df["is_referral"] = (
    np.random.rand(
        len(df)
    )
    <
    df["referral_rate_param"]
).astype(int)

df["referral_specialty"] = np.where(
    df["is_referral"] == 1,
    "Ophthalmology",
    None,
)


# ============================================================
# 17. MEMBER-PROVIDER DISTANCE
#
# Use Case 2 feature.
#
# Normal providers:
#   typical average distance around 10-30 miles
#
# Suspicious providers:
#   typical average distance around 40-70 miles
#
# Gamma noise creates realistic overlap between the groups.
# ============================================================

distance_shape = 2.0

distance_scale = (
    df["member_provider_distance_mean_param"]
    / distance_shape
)

member_provider_distance = np.random.gamma(
    shape=distance_shape,
    scale=distance_scale,
)

member_provider_distance += np.random.normal(
    loc=0,
    scale=5,
    size=len(df),
)

df["member_provider_distance"] = (
    np.clip(
        member_provider_distance,
        0.5,
        250,
    )
    .round(1)
)


# ============================================================
# 18. FINANCIAL AMOUNTS
#
# Amount depends on CPT rather than directly on fraud label.
# ============================================================

base_billed = (
    np.random.gamma(
        shape=2.5,
        scale=100,
        size=len(df),
    )
    + 40
)

df["total_billed_amt"] = (
    base_billed
)

df.loc[
    df["cpt_code"] == "99214",
    "total_billed_amt",
] *= 1.30

df.loc[
    df["cpt_code"] == "99215",
    "total_billed_amt",
] *= 1.55

df.loc[
    df["is_eye_procedure"] == 1,
    "total_billed_amt",
] *= 1.40

df["total_allowed_amt"] = (
    df["total_billed_amt"]
    * np.random.uniform(
        0.60,
        0.90,
        len(df),
    )
)

df["total_paid_amount"] = (
    df["total_allowed_amt"]
    * np.random.uniform(
        0.85,
        1.00,
        len(df),
    )
)


# ============================================================
# 19. CLAIM DATE
#
# Claims may arrive up to 14 days after service.
# Cap claim_date at the configured raw-data end date.
# ============================================================

df["service_date"] = pd.to_datetime(
    df["service_date"]
)

claim_delay_days = np.random.randint(
    0,
    15,
    len(df),
)

df["claim_date"] = (
    df["service_date"]
    + pd.to_timedelta(
        claim_delay_days,
        unit="D",
    )
)

df["claim_date"] = df[
    "claim_date"
].clip(
    upper=pd.Timestamp(
        END_DATE
    )
)


# ============================================================
# 20. ROUND FINANCIAL FIELDS
# ============================================================

for column in [
    "total_billed_amt",
    "total_allowed_amt",
    "total_paid_amount",
]:
    df[column] = (
        df[column]
        .round(2)
    )


# ============================================================
# 21. DROP INTERNAL GENERATION PARAMETERS
# ============================================================

df = df.drop(
    columns=[
        "high_acuity_rate_param",
        "referral_rate_param",
        "eye_procedure_rate_param",
        "pct_99215_given_high_acuity_param",
        "member_provider_distance_mean_param",
    ]
)


# ============================================================
# 22. FINAL COLUMN ORDER
#
# No dataset_split here.
# TRAIN / TEST is assigned later by the rolling-window
# feature-engineering component.
# ============================================================

final_columns = [
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

    "member_provider_distance",

    # Demo-only ground truth.
    # Do not use this as a model feature.
    "fraud_provider_label",
]

df = df[
    final_columns
]


# ============================================================
# 23. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 24. VALIDATION OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Synthetic Texas Eye-Related PCP Claims"
)

print(
    "=" * 70
)

print(
    f"\nCurrent data month: "
    f"{CURRENT_DATA_MONTH}"
)

print(
    f"Raw data window: "
    f"{START_DATE.date()} "
    f"to {END_DATE.date()}"
)

print(
    f"\nClaims: "
    f"{len(df):,}"
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
    "\nMember-provider distance by fraud label:"
)

print(
    df.groupby(
        "fraud_provider_label"
    )["member_provider_distance"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .round(2)
)

print(
    "\nMonthly claim volume:"
)

monthly_claim_volume = (
    df.assign(
        service_month=(
            df["service_date"]
            .dt.to_period("M")
            .astype(str)
        )
    )
    .groupby(
        "service_month"
    )
    .size()
)

print(
    monthly_claim_volume
)

print(
    "\nFinal columns:"
)

for column in df.columns:
    print(
        f"  - {column}"
    )

print(
    f"\nSaved to: "
    f"{OUTPUT_FILE}"
)
