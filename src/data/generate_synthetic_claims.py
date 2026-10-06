```python
import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config.load_config import load_config


# ============================================================
# 1. LOAD PIPELINE CONFIGURATION
#
# The synthetic data generator uses the same YAML configuration
# as the Vertex AI pipeline.
#
# Example:
#
# data:
#   current_data_month: "2026-10-01"
#
#   rolling_window:
#     training_months: 6
#     testing_months: 3
#
# No TRAIN / TEST dates are hard coded in this script.
# ============================================================

config = load_config()
DATA_CONFIG = config["data"]

CURRENT_DATA_MONTH = str(DATA_CONFIG["current_data_month"])
ROLLING_WINDOW_CONFIG = DATA_CONFIG["rolling_window"]

TRAINING_MONTHS = int(
    ROLLING_WINDOW_CONFIG["training_months"]
)

TESTING_MONTHS = int(
    ROLLING_WINDOW_CONFIG["testing_months"]
)


# ============================================================
# 2. SYNTHETIC DATA CONFIGURATION
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_CLAIMS = 100_000
N_PROVIDERS = 200
N_MEMBERS = 20_000

# 10% of providers are synthetically suspicious.
N_SUSPICIOUS_PROVIDERS = 20

OUTPUT_FILE = "synthetic_claims.csv"


# ============================================================
# 3. DATE UTILITIES
# ============================================================

def add_months(year: int, month: int, offset: int):
    """
    Move a year/month pair forward or backward.

    Example:
    add_months(2026, 10, -8)
    returns:
    (2026, 2)
    """
    month_index = year * 12 + month - 1 + offset

    new_year = month_index // 12
    new_month = month_index % 12 + 1

    return new_year, new_month


def get_month_end(year: int, month: int):
    """
    Return the final datetime of a month.
    """
    final_day = calendar.monthrange(
        year,
        month,
    )[1]

    return datetime(
        year,
        month,
        final_day,
    )


def calculate_raw_data_window():
    """
    Calculate the minimum raw-data window required by the
    configured rolling training/testing strategy.

    Example:

    current_data_month = 2026-09-01
    training_months = 6
    testing_months = 3

    Raw claims:
        2026-01-01 -> 2026-09-30

    Pipeline later creates:

        TRAIN:
        2026-01-01 -> 2026-06-30

        TEST:
        2026-07-01 -> 2026-09-30

    If current_data_month changes to:

        2026-10-01

    Raw claims become:

        2026-02-01 -> 2026-10-31

    Pipeline later creates:

        TRAIN:
        2026-02-01 -> 2026-07-31

        TEST:
        2026-08-01 -> 2026-10-31
    """
    current_month = datetime.strptime(
        CURRENT_DATA_MONTH,
        "%Y-%m-%d",
    )

    total_months = TRAINING_MONTHS + TESTING_MONTHS

    start_year, start_month = add_months(
        current_month.year,
        current_month.month,
        -(total_months - 1),
    )

    start_date = datetime(
        start_year,
        start_month,
        1,
    )

    end_date = get_month_end(
        current_month.year,
        current_month.month,
    )

    return start_date, end_date


START_DATE, END_DATE = calculate_raw_data_window()


# ============================================================
# 4. TEXAS GEOGRAPHY
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
# 5. CREATE PRIMARY CARE PROVIDERS
# ============================================================

providers = pd.DataFrame(
    {
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
            p=city_probabilities,
        ),
    }
)


# ============================================================
# 6. SELECT SYNTHETIC SUSPICIOUS PROVIDERS
#
# This is synthetic ground truth used for the demo.
#
# Use Case 1:
# The label allows Random Forest to train on the rolling
# TRAIN dataset and evaluate on the rolling TEST dataset.
#
# In a real FWA system, labels would come from confirmed
# investigation outcomes rather than synthetic truth.
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
# 7. PROVIDER BEHAVIOR PARAMETERS
#
# These parameters exist ONLY to generate synthetic patterns.
#
# They are removed from the final claim-level data.
# ============================================================

providers["high_acuity_rate_param"] = 0.0
providers["referral_rate_param"] = 0.0
providers["eye_procedure_rate_param"] = 0.0
providers["pct_99215_given_high_acuity_param"] = 0.0

# ------------------------------------------------------------
# USE CASE 2
#
# New claim-level field:
#
# member_provider_distance
#
# Suspicious providers tend to attract members from farther
# away.
#
# We intentionally retain overlap between normal and suspicious
# providers so the feature is informative but does NOT become
# a perfect fraud identifier.
# ------------------------------------------------------------

providers["member_provider_distance_mean_param"] = 0.0

for idx, row in providers.iterrows():
    provider_id = row["provider_id"]
    suspicious = provider_id in suspicious_provider_ids

    if suspicious:
        # ----------------------------------------------------
        # Strongest existing anomaly:
        # high use of higher-acuity E/M services
        # ----------------------------------------------------
        providers.at[
            idx,
            "high_acuity_rate_param",
        ] = np.random.uniform(
            0.65,
            0.80,
        )

        # ----------------------------------------------------
        # Moderately elevated referral behavior
        # ----------------------------------------------------
        providers.at[
            idx,
            "referral_rate_param",
        ] = np.random.uniform(
            0.38,
            0.45,
        )

        # ----------------------------------------------------
        # Similar eye-procedure utilization
        # ----------------------------------------------------
        providers.at[
            idx,
            "eye_procedure_rate_param",
        ] = np.random.uniform(
            0.08,
            0.12,
        )

        # ----------------------------------------------------
        # Higher 99215 utilization
        # ----------------------------------------------------
        providers.at[
            idx,
            "pct_99215_given_high_acuity_param",
        ] = np.random.uniform(
            0.35,
            0.50,
        )

        # ----------------------------------------------------
        # USE CASE 2 FEATURE
        #
        # Suspicious providers tend to serve members who live
        # farther away.
        #
        # Approximate provider-level mean distance:
        # 40–70 miles.
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Normal provider/member distance.
        #
        # Approximate provider-level mean:
        # 10–30 miles.
        #
        # There will still be long-distance normal claims.
        # ----------------------------------------------------
        providers.at[
            idx,
            "member_provider_distance_mean_param",
        ] = np.random.uniform(
            10,
            30,
        )


# ============================================================
# 8. CREATE MEMBERS
# ============================================================

members = [
    f"M{str(i).zfill(6)}"
    for i in range(1, N_MEMBERS + 1)
]


# ============================================================
# 9. DIAGNOSIS CODES
#
# All claims remain eye/adnexa related for the synthetic demo.
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
# 10. CPT CODES
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
# 11. PLACE OF SERVICE
# ============================================================

place_of_service_codes = [
    "11",
    "21",
    "22",
    "23",
]


# ============================================================
# 12. CLAIM VOLUME
#
# Provider volume is intentionally approximately uniform.
#
# Fraud should be detected from behavioral patterns rather
# than simply from claim volume.
# ============================================================

provider_choices = providers["provider_id"].values

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
# 13. GENERATE SERVICE DATES
#
# There is intentionally NO TRAIN / TEST assignment here.
#
# The Vertex pipeline dynamically determines dataset_split
# from the rolling window.
# ============================================================

date_range_days = (
    END_DATE
    - START_DATE
).days + 1

service_date_offsets = np.random.randint(
    0,
    date_range_days,
    N_CLAIMS,
)


# ============================================================
# 14. BASE CLAIM DATA
# ============================================================

df = pd.DataFrame(
    {
        "claim_id": [
            f"C{str(i).zfill(8)}"
            for i in range(1, N_CLAIMS + 1)
        ],
        "provider_id": chosen_providers,
        "member_id": np.random.choice(
            members,
            size=N_CLAIMS,
        ),
        "service_date": [
           
