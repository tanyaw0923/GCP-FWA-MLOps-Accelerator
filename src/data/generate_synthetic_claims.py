from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# 1. SYNTHETIC DATA SETTINGS
#
# Demo data volume:
#   Jan 1 - Sep 30, 2026: 100,000 claims
#   Oct 1 - Oct 31, 2026: 12,500 claims (100,000 / 8)
#
# Total: 112,500 claims
# ============================================================

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)

BASE_N_CLAIMS = 100_000
OCTOBER_N_CLAIMS = BASE_N_CLAIMS // 8
N_CLAIMS = BASE_N_CLAIMS + OCTOBER_N_CLAIMS

N_PROVIDERS = 200
N_MEMBERS = 20_000

# 10% confirmed/synthetic fraud providers
N_FRAUD_PROVIDERS = 20

OUTPUT_FILE = (
    "synthetic_claims.csv"
)


# ============================================================
# 2. FIXED DEMO DATE WINDOWS
# ============================================================

START_DATE = datetime(
    2026,
    1,
    1,
)

SEPTEMBER_END_DATE = datetime(
    2026,
    9,
    30,
)

OCTOBER_START_DATE = datetime(
    2026,
    10,
    1,
)

END_DATE = datetime(
    2026,
    10,
    31,
)


# ============================================================
# 3. PROVIDERS
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


providers = pd.DataFrame(
    {
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

        "provider_specialty": (
            "Primary Care"
        ),

        "provider_state": (
            "TX"
        ),

        "provider_city": (
            rng.choice(
                texas_cities,
                size=N_PROVIDERS,
                p=city_probabilities,
            )
        ),
    }
)


# ============================================================
# 5. SYNTHETIC FRAUD LABELS
#
# Fraud label != perfect behavior.
#
# This is important.
#
# Real FWA providers do not always look obviously fraudulent,
# and some legitimate providers can have unusual utilization.
# ============================================================

fraud_provider_ids = set(
    rng.choice(
        providers[
            "provider_id"
        ],
        size=N_FRAUD_PROVIDERS,
        replace=False,
    )
)


providers[
    "fraud_provider_label"
] = (
    providers[
        "provider_id"
    ]
    .isin(
        fraud_provider_ids
    )
    .astype(int)
)


# ============================================================
# 6. PROVIDER LATENT RISK
#
# Instead of assigning completely separate fraud/normal
# feature ranges, create overlapping latent risk.
#
# Normal providers:
#     generally lower risk
#
# Fraud providers:
#     generally higher risk
#
# But substantial overlap is intentionally retained.
# ============================================================

providers[
    "latent_risk"
] = 0.0


fraud_mask = (
    providers[
        "fraud_provider_label"
    ]
    == 1
)


normal_mask = (
    providers[
        "fraud_provider_label"
    ]
    == 0
)


# Normal providers usually lower risk,
# but some naturally appear unusual.
providers.loc[
    normal_mask,
    "latent_risk",
] = rng.beta(
    2.2,
    4.0,
    size=normal_mask.sum(),
)


# Fraud providers generally higher risk,
# but not all are obvious.
providers.loc[
    fraud_mask,
    "latent_risk",
] = rng.beta(
    4.0,
    2.3,
    size=fraud_mask.sum(),
)


# ============================================================
# 7. HARD NEGATIVES + SUBTLE FRAUD
#
# These are particularly important for preventing perfect
# model performance.
#
# Hard negatives:
# legitimate providers with suspicious-looking behavior.
#
# Subtle fraud:
# fraudulent providers whose behavior looks closer to normal.
# ============================================================

normal_indices = (
    providers.index[
        normal_mask
    ]
    .to_numpy()
)


fraud_indices = (
    providers.index[
        fraud_mask
    ]
    .to_numpy()
)


# About 10% of normal providers become hard negatives.
hard_negative_indices = (
    rng.choice(
        normal_indices,
        size=18,
        replace=False,
    )
)


providers.loc[
    hard_negative_indices,
    "latent_risk",
] = rng.uniform(
    0.55,
    0.85,
    size=len(
        hard_negative_indices
    ),
)


# About 25% of fraud providers are deliberately subtle.
subtle_fraud_indices = (
    rng.choice(
        fraud_indices,
        size=5,
        replace=False,
    )
)


providers.loc[
    subtle_fraud_indices,
    "latent_risk",
] = rng.uniform(
    0.20,
    0.50,
    size=len(
        subtle_fraud_indices
    ),
)


# Keep risk valid.
providers[
    "latent_risk"
] = (
    providers[
        "latent_risk"
    ]
    .clip(
        0.02,
        0.98,
    )
)


# ============================================================
# 8. PROVIDER BEHAVIOR PARAMETERS
#
# Features are related to risk but contain independent noise.
#
# No single feature should perfectly identify fraud.
# ============================================================

provider_count = len(
    providers
)


risk = (
    providers[
        "latent_risk"
    ]
    .to_numpy()
)


# Higher risk tends to mean more high-acuity claims.
providers[
    "high_acuity_rate_param"
] = np.clip(
    0.16
    + 0.40 * risk
    + rng.normal(
        0,
        0.07,
        provider_count,
    ),
    0.08,
    0.75,
)


# Referral behavior provides another moderate signal.
providers[
    "referral_rate_param"
] = np.clip(
    0.20
    + 0.20 * risk
    + rng.normal(
        0,
        0.055,
        provider_count,
    ),
    0.08,
    0.60,
)


# 99215 is correlated with risk but overlaps strongly.
providers[
    "pct_99215_given_high_acuity_param"
] = np.clip(
    0.08
    + 0.28 * risk
    + rng.normal(
        0,
        0.06,
        provider_count,
    ),
    0.03,
    0.60,
)


# Eye procedures are intentionally weak predictors.
providers[
    "eye_procedure_rate_param"
] = np.clip(
    0.08
    + 0.04 * risk
    + rng.normal(
        0,
        0.025,
        provider_count,
    ),
    0.03,
    0.20,
)


# ============================================================
# 9. USE CASE 2 FEATURE
#
# member_provider_distance
#
# Higher-risk providers tend to have somewhat greater distance,
# but there is substantial overlap.
#
# Normal provider average:
# roughly 10–40 miles
#
# Higher-risk provider average:
# roughly 20–55 miles
#
# This should help the model without becoming a perfect rule.
# ============================================================

providers[
    "member_provider_distance_mean_param"
] = np.clip(
    12
    + 34 * risk
    + rng.normal(
        0,
        9,
        provider_count,
    ),
    5,
    65,
)


# ============================================================
# 10. PAYMENT BEHAVIOR
#
# Payment amount is informative but noisy.
# ============================================================

providers[
    "paid_amount_mean_param"
] = np.clip(
    110
    + 75 * risk
    + rng.normal(
        0,
        30,
        provider_count,
    ),
    60,
    300,
)


# ============================================================
# 11. MEMBERS
# ============================================================

members = [
    f"M{str(i).zfill(6)}"
    for i in range(
        1,
        N_MEMBERS + 1,
    )
]


# ============================================================
# 12. CODES
# ============================================================

eye_icd_codes = [
    "H10.9",
    "H25.9",
    "H40.9",
    "H35.30",
    "H53.8",
    "H57.9",
]


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


place_of_service_codes = [
    "11",
    "21",
    "22",
    "23",
]


# ============================================================
# 13. CLAIM VOLUME
#
# Keep volumes roughly similar across providers so claim count
# itself does not become a fraud shortcut.
# ============================================================

provider_choices = (
    providers[
        "provider_id"
    ]
    .to_numpy()
)


provider_weights = (
    rng.uniform(
        0.85,
        1.15,
        N_PROVIDERS,
    )
)


provider_weights = (
    provider_weights
    / provider_weights.sum()
)


chosen_providers = (
    rng.choice(
        provider_choices,
        size=N_CLAIMS,
        p=provider_weights,
    )
)


# ============================================================
# 14. SERVICE DATES
#
# Keep the demo volumes explicit:
#   - exactly 100,000 claims from Jan 1 through Sep 30
#   - exactly 12,500 claims from Oct 1 through Oct 31
#
# Dates are sampled uniformly within each period, then shuffled
# so claim IDs are not ordered by month.
# ============================================================

base_date_range_days = (
    SEPTEMBER_END_DATE
    - START_DATE
).days + 1


base_service_date_offsets = (
    rng.integers(
        0,
        base_date_range_days,
        size=BASE_N_CLAIMS,
    )
)


base_service_dates = (
    pd.to_datetime(
        START_DATE
    )
    + pd.to_timedelta(
        base_service_date_offsets,
        unit="D",
    )
)


october_date_range_days = (
    END_DATE
    - OCTOBER_START_DATE
).days + 1


october_service_date_offsets = (
    rng.integers(
        0,
        october_date_range_days,
        size=OCTOBER_N_CLAIMS,
    )
)


october_service_dates = (
    pd.to_datetime(
        OCTOBER_START_DATE
    )
    + pd.to_timedelta(
        october_service_date_offsets,
        unit="D",
    )
)


service_dates = np.concatenate(
    [
        base_service_dates.to_numpy(),
        october_service_dates.to_numpy(),
    ]
)


service_dates = (
    pd.to_datetime(
        rng.permutation(
            service_dates
        )
    )
)


# ============================================================
# 15. BASE CLAIM DATA
# ============================================================

df = pd.DataFrame(
    {
        "claim_id": [
            f"C{str(i).zfill(8)}"
            for i in range(
                1,
                N_CLAIMS + 1,
            )
        ],

        "provider_id": (
            chosen_providers
        ),

        "member_id": (
            rng.choice(
                members,
                size=N_CLAIMS,
            )
        ),

        "service_date": (
            service_dates
        ),
    }
)


# ============================================================
# 16. JOIN PROVIDER CHARACTERISTICS
# ============================================================

provider_columns = [
    "provider_id",
    "provider_name",
    "provider_specialty",
    "provider_state",
    "provider_city",
    "fraud_provider_label",
    "latent_risk",
    "high_acuity_rate_param",
    "referral_rate_param",
    "eye_procedure_rate_param",
    "pct_99215_given_high_acuity_param",
    "member_provider_distance_mean_param",
    "paid_amount_mean_param",
]


df = df.merge(
    providers[
        provider_columns
    ],
    on="provider_id",
    how="left",
)


# ============================================================
# 17. TEMPORAL VARIATION
#
# Provider behavior should not be identical every month.
#
# This prevents TRAIN and TEST from looking like exact copies
# of the same provider behavior.
# ============================================================

df[
    "service_month"
] = (
    df[
        "service_date"
    ]
    .dt.month
)


month_noise_lookup = {
    month: rng.normal(
        0,
        0.035,
    )
    for month in sorted(
        df[
            "service_month"
        ].unique()
    )
}


df[
    "month_noise"
] = (
    df[
        "service_month"
    ]
    .map(
        month_noise_lookup
    )
)


# ============================================================
# 18. HIGH-ACUITY E/M BEHAVIOR
# ============================================================

high_acuity_probability = np.clip(
    df[
        "high_acuity_rate_param"
    ]
    + df[
        "month_noise"
    ]
    + rng.normal(
        0,
        0.03,
        N_CLAIMS,
    ),
    0.03,
    0.90,
)


df[
    "is_high_acuity_em"
] = (
    rng.random(
        N_CLAIMS
    )
    < high_acuity_probability
).astype(int)


# ============================================================
# 19. 99214 / 99215
# ============================================================

pct_99215_probability = np.clip(
    df[
        "pct_99215_given_high_acuity_param"
    ]
    + rng.normal(
        0,
        0.04,
        N_CLAIMS,
    ),
    0.03,
    0.75,
)


is_99215 = (
    (
        df[
            "is_high_acuity_em"
        ]
        == 1
    )
    &
    (
        rng.random(
            N_CLAIMS
        )
        < pct_99215_probability
    )
)


df[
    "is_99215"
] = (
    is_99215.astype(int)
)


df[
    "is_99214"
] = (
    (
        df[
            "is_high_acuity_em"
        ]
        == 1
    )
    &
    (
        df[
            "is_99215"
        ]
        == 0
    )
).astype(int)


# ============================================================
# 20. REFERRAL
# ============================================================

referral_probability = np.clip(
    df[
        "referral_rate_param"
    ]
    + rng.normal(
        0,
        0.04,
        N_CLAIMS,
    ),
    0.03,
    0.80,
)


df[
    "is_referral"
] = (
    rng.random(
        N_CLAIMS
    )
    < referral_probability
).astype(int)


# ============================================================
# 21. EYE PROCEDURE
# ============================================================

eye_probability = np.clip(
    df[
        "eye_procedure_rate_param"
    ]
    + rng.normal(
        0,
        0.025,
        N_CLAIMS,
    ),
    0.02,
    0.40,
)


df[
    "is_eye_procedure"
] = (
    rng.random(
        N_CLAIMS
    )
    < eye_probability
).astype(int)


# ============================================================
# 22. CPT CODE
# ============================================================

cpt_codes = []


for row in df.itertuples():

    if row.is_eye_procedure == 1:

        cpt_code = (
            rng.choice(
                eye_procedure_codes
            )
        )

    elif row.is_99215 == 1:

        cpt_code = (
            "99215"
        )

    elif row.is_99214 == 1:

        cpt_code = (
            "99214"
        )

    else:

        cpt_code = (
            rng.choice(
                routine_em_codes,
                p=routine_em_probs,
            )
        )

    cpt_codes.append(
        cpt_code
    )


df[
    "cpt_code"
] = cpt_codes


# ============================================================
# 23. DIAGNOSIS
# ============================================================

df[
    "icd_code"
] = (
    rng.choice(
        eye_icd_codes,
        size=N_CLAIMS,
    )
)


df[
    "diagnoses_code"
] = (
    df[
        "icd_code"
    ]
)


# ============================================================
# 24. MEMBER / PROVIDER DISTANCE
#
# Gamma distribution creates realistic right skew.
#
# Claim-level noise ensures providers with similar underlying
# means still have overlapping distributions.
# ============================================================

distance_mean = (
    df[
        "member_provider_distance_mean_param"
    ]
    .to_numpy()
)


distance_shape = 2.0


distance_scale = (
    distance_mean
    / distance_shape
)


member_provider_distance = (
    rng.gamma(
        shape=distance_shape,
        scale=distance_scale,
    )
)


# Additional measurement/geography noise
member_provider_distance += (
    rng.normal(
        0,
        7,
        N_CLAIMS,
    )
)


df[
    "member_provider_distance"
] = np.clip(
    member_provider_distance,
    0.5,
    180,
).round(1)


# ============================================================
# 25. PAYMENT AMOUNTS
#
# Paid amounts correlate weakly with risk and utilization.
# There is intentionally substantial variance.
# ============================================================

base_paid = (
    df[
        "paid_amount_mean_param"
    ]
    .to_numpy()
)


acuity_adjustment = (
    df[
        "is_high_acuity_em"
    ]
    .to_numpy()
    * 55
)


procedure_adjustment = (
    df[
        "is_eye_procedure"
    ]
    .to_numpy()
    * 90
)


paid_amount = (
    base_paid
    + acuity_adjustment
    + procedure_adjustment
    + rng.normal(
        0,
        55,
        N_CLAIMS,
    )
)


df[
    "total_paid_amount"
] = np.clip(
    paid_amount,
    20,
    700,
).round(2)


allowed_multiplier = (
    rng.uniform(
        1.05,
        1.30,
        N_CLAIMS,
    )
)


df[
    "total_allowed_amt"
] = (
    df[
        "total_paid_amount"
    ]
    * allowed_multiplier
).round(2)


billed_multiplier = (
    rng.uniform(
        1.10,
        1.50,
        N_CLAIMS,
    )
)


df[
    "total_billed_amt"
] = (
    df[
        "total_allowed_amt"
    ]
    * billed_multiplier
).round(2)


# ============================================================
# 26. PLACE OF SERVICE
# ============================================================

df[
    "place_of_service"
] = (
    rng.choice(
        place_of_service_codes,
        size=N_CLAIMS,
        p=[
            0.72,
            0.08,
            0.15,
            0.05,
        ],
    )
)


# ============================================================
# 27. CLAIM DATE
# ============================================================

claim_delay_days = (
    rng.integers(
        1,
        15,
        N_CLAIMS,
    )
)


df[
    "claim_date"
] = (
    df[
        "service_date"
    ]
    + pd.to_timedelta(
        claim_delay_days,
        unit="D",
    )
)


# ============================================================
# 28. REMOVE GENERATION-ONLY FIELDS
#
# These parameters create synthetic behavior but must not be
# available to the ML model.
#
# Otherwise they could introduce target leakage.
# ============================================================

generation_only_columns = [
    "latent_risk",
    "high_acuity_rate_param",
    "referral_rate_param",
    "eye_procedure_rate_param",
    "pct_99215_given_high_acuity_param",
    "member_provider_distance_mean_param",
    "paid_amount_mean_param",
    "service_month",
    "month_noise",
]


df = df.drop(
    columns=(
        generation_only_columns
    )
)


# ============================================================
# 29. FINAL COLUMN ORDER
# ============================================================

final_columns = [
    "claim_id",
    "claim_date",
    "provider_id",
    "provider_name",
    "provider_specialty",
    "member_id",
    "service_date",
    "provider_city",
    "provider_state",
    "cpt_code",
    "icd_code",
    "diagnoses_code",
    "total_billed_amt",
    "total_allowed_amt",
    "total_paid_amount",
    "place_of_service",
    "is_high_acuity_em",
    "is_99214",
    "is_99215",
    "is_referral",
    "is_eye_procedure",
    "member_provider_distance",
    "fraud_provider_label",
]


df = df[
    final_columns
]


# ============================================================
# 30. SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 31. DATASET SUMMARY
# ============================================================

print(
    "=" * 70
)

print(
    "Synthetic FWA Claims Generated"
)

print(
    "=" * 70
)


print(
    f"Rows: "
    f"{len(df):,}"
)


print(
    f"Jan-Sep claims: "
    f"{(
        (df['service_date'] >= pd.Timestamp('2026-01-01'))
        & (df['service_date'] <= pd.Timestamp('2026-09-30'))
    ).sum():,}"
)


print(
    f"October claims: "
    f"{(
        (df['service_date'] >= pd.Timestamp('2026-10-01'))
        & (df['service_date'] <= pd.Timestamp('2026-10-31'))
    ).sum():,}"
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
    f"Fraud providers: "
    f"{providers['fraud_provider_label'].sum()}"
)


print(
    f"Normal providers: "
    f"{N_PROVIDERS - providers['fraud_provider_label'].sum()}"
)


print(
    f"Raw data window: "
    f"{START_DATE.date()} "
    f"to "
    f"{END_DATE.date()}"
)


print(
    f"Output: "
    f"{OUTPUT_FILE}"
)


# ============================================================
# 32. PROVIDER-LEVEL SANITY CHECK
#
# This lets us verify that fraud providers are statistically
# different but NOT perfectly separable.
# ============================================================

provider_summary = (
    df.groupby(
        [
            "provider_id",
            "fraud_provider_label",
        ]
    )
    .agg(
        high_acuity_pct=(
            "is_high_acuity_em",
            "mean",
        ),
        pct_99215=(
            "is_99215",
            "mean",
        ),
        referral_rate=(
            "is_referral",
            "mean",
        ),
        eye_procedure_pct=(
            "is_eye_procedure",
            "mean",
        ),
        avg_paid_per_claim=(
            "total_paid_amount",
            "mean",
        ),
        avg_member_provider_distance=(
            "member_provider_distance",
            "mean",
        ),
    )
    .reset_index()
)


print(
    "\nProvider-level feature averages:"
)


print(
    provider_summary.groupby(
        "fraud_provider_label"
    )[
        [
            "high_acuity_pct",
            "pct_99215",
            "referral_rate",
            "eye_procedure_pct",
            "avg_paid_per_claim",
            "avg_member_provider_distance",
        ]
    ]
    .mean()
    .round(3)
)


print(
    "\nProvider-level feature ranges:"
)


for feature in [
    "high_acuity_pct",
    "pct_99215",
    "referral_rate",
    "avg_paid_per_claim",
    "avg_member_provider_distance",
]:

    print(
        f"\n{feature}"
    )

    for label in [
        0,
        1,
    ]:

        subset = (
            provider_summary[
                provider_summary[
                    "fraud_provider_label"
                ]
                == label
            ][
                feature
            ]
        )

        print(
            f"  label={label}: "
            f"min={subset.min():.3f}, "
            f"mean={subset.mean():.3f}, "
            f"max={subset.max():.3f}"
        )
