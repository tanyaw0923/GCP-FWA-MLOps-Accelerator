import os

import numpy as np
import xgboost as xgb

from fastapi import FastAPI, HTTPException
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


# provider_id is metadata.
# It must NOT be sent into XGBoost.
METADATA_FIELDS = [
    "provider_id",
]


# ============================================================
# LOAD MODEL
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

    instances = (
        request.instances
    )


    if not instances:

        return {
            "predictions": []
        }


    feature_rows = []

    provider_ids = []


    for instance in instances:

        # ====================================================
        # CASE 1:
        # Vertex sends a dictionary
        #
        # {
        #   "provider_id": "P001",
        #   "high_acuity_pct": 0.31,
        #   ...
        # }
        #
        # In this case provider_id is simply ignored for the
        # model matrix.
        # ====================================================

        if isinstance(
            instance,
            dict,
        ):

            provider_id = (
                instance.get(
                    "provider_id"
                )
            )


            missing_features = [
                feature
                for feature in FEATURES
                if feature not in instance
            ]


            if missing_features:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing model features: "
                        f"{missing_features}"
                    ),
                )


            row = [
                float(
                    instance[
                        feature
                    ]
                )
                for feature in FEATURES
            ]


        # ====================================================
        # CASE 2:
        # Vertex sends positional BigQuery values
        #
        # Expected production layout:
        #
        # [
        #   provider_id,
        #   high_acuity_pct,
        #   pct_99214,
        #   pct_99215,
        #   referral_rate,
        #   eye_procedure_pct,
        #   avg_paid_per_claim,
        #   high_acuity_oe_ratio
        # ]
        #
        # That is:
        #
        #   1 metadata field
        #   +
        #   7 model features
        #   =
        #   8 values
        # ====================================================

        elif isinstance(
            instance,
            list,
        ):

            expected_features_only = (
                len(
                    FEATURES
                )
            )

            expected_with_metadata = (
                len(
                    FEATURES
                )
                + len(
                    METADATA_FIELDS
                )
            )


            # --------------------------------------------
            # Feature-only input
            #
            # Used by the consistency test table.
            # --------------------------------------------

            if (
                len(instance)
                ==
                expected_features_only
            ):

                provider_id = None

                feature_values = (
                    instance
                )


            # --------------------------------------------
            # Production input
            #
            # First field = provider_id
            # Remaining fields = model features
            # --------------------------------------------

            elif (
                len(instance)
                ==
                expected_with_metadata
            ):

                provider_id = (
                    instance[0]
                )

                feature_values = (
                    instance[1:]
                )


            else:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unexpected input field count. "
                        f"Received {len(instance)} fields. "
                        f"Expected "
                        f"{expected_features_only} "
                        "feature-only fields or "
                        f"{expected_with_metadata} "
                        "fields including provider_id."
                    ),
                )


            row = [
                float(value)
                for value in feature_values
            ]


        else:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Each instance must be "
                    "a dictionary or list."
                ),
            )


        provider_ids.append(
            provider_id
        )

        feature_rows.append(
            row
        )


    # ========================================================
    # CREATE NUMERIC MODEL MATRIX
    # ========================================================

    matrix = np.asarray(
        feature_rows,
        dtype=np.float32,
    )


    if (
        matrix.shape[1]
        != len(FEATURES)
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Final model matrix has "
                f"{matrix.shape[1]} features; "
                f"expected {len(FEATURES)}."
            ),
        )


    # ========================================================
    # CONVERT TO XGBOOST DMATRIX
    # ========================================================

    dmatrix = xgb.DMatrix(
        matrix,
        feature_names=FEATURES,
    )


    # ========================================================
    # PREDICT FRAUD PROBABILITY
    # ========================================================

    probabilities = (
        booster.predict(
            dmatrix
        )
    )


    # ========================================================
    # RETURN
    #
    # Keep the prediction itself simple because Vertex writes
    # this field into the raw BigQuery prediction table.
    #
    # provider_id is already carried through the BigQuery
    # instance fields by Vertex.
    # ========================================================

    predictions = [
        float(
            probability
        )
        for probability in probabilities
    ]


    return {
        "predictions":
            predictions
    }
