import os

import numpy as np
import xgboost as xgb

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/model/model.bst",
)

FEATURES = [
    "high_acuity_pct",
    "pct_99214",
    "pct_99215",
    "referral_rate",
    "eye_procedure_pct",
    "avg_paid_per_claim",
    "high_acuity_oe_ratio",
]


# ============================================================
# LOAD MODEL ON STARTUP
# ============================================================

booster = xgb.Booster()

booster.load_model(
    MODEL_PATH
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()


class PredictionRequest(BaseModel):
    instances: list


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):
    instances = request.instances

    if not instances:
        return {
            "predictions": []
        }

    rows = []

    for instance in instances:

        # --------------------------------------------
        # Support dictionary-style input:
        #
        # {
        #   "high_acuity_pct": 0.2,
        #   ...
        # }
        # --------------------------------------------

        if isinstance(
            instance,
            dict,
        ):

            row = [
                float(
                    instance[
                        feature
                    ]
                )
                for feature in FEATURES
            ]

        # --------------------------------------------
        # Also support positional input:
        #
        # [0.2, 0.1, ...]
        # --------------------------------------------

        elif isinstance(
            instance,
            list,
        ):

            if len(instance) != len(FEATURES):
                raise ValueError(
                    "Input feature count does not "
                    "match expected feature count."
                )

            row = [
                float(value)
                for value in instance
            ]

        else:

            raise ValueError(
                "Each instance must be either "
                "a dictionary or list."
            )

        rows.append(
            row
        )

    # ========================================================
    # CONVERT TO DMATRIX
    # ========================================================

    matrix = np.asarray(
        rows,
        dtype=float,
    )

    dmatrix = xgb.DMatrix(
        matrix,
        feature_names=FEATURES,
    )

    # ========================================================
    # PREDICT
    # ========================================================

    probabilities = (
        booster.predict(
            dmatrix
        )
    )

    predictions = [
        float(value)
        for value in probabilities
    ]

    return {
        "predictions":
            predictions
    }
