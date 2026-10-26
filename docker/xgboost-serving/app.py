import json
import os
from urllib.parse import urlparse

import numpy as np
import xgboost as xgb
from google.cloud import storage

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/model/model.bst",
)

AIP_STORAGE_URI = os.environ.get(
    "AIP_STORAGE_URI",
    "",
)


FEATURE_COLUMNS_JSON = os.environ.get(
    "FEATURE_COLUMNS_JSON",
    "",
)

if not FEATURE_COLUMNS_JSON:
    raise RuntimeError(
        "FEATURE_COLUMNS_JSON is required. "
        "The training pipeline must pass the exact "
        "model feature contract into the serving container."
    )

try:
    FEATURES = json.loads(
        FEATURE_COLUMNS_JSON
    )
except json.JSONDecodeError as exc:
    raise RuntimeError(
        "FEATURE_COLUMNS_JSON is not valid JSON."
    ) from exc

if (
    not isinstance(FEATURES, list)
    or not FEATURES
    or not all(
        isinstance(feature, str)
        and feature
        for feature in FEATURES
    )
):
    raise RuntimeError(
        "FEATURE_COLUMNS_JSON must be a non-empty "
        "JSON array of feature names."
    )


# provider_id is removed upstream by Vertex BatchPredictionJob
# instance_config.key_field and is returned separately as "key".
# The model container therefore receives model features only.


# ============================================================
# LOAD MODEL
#
# For Vertex custom prediction containers, AIP_STORAGE_URI
# points to the model artifact directory copied by Vertex.
# Download model.bst at startup when it is not baked into the
# image. Local development can still use MODEL_PATH directly.
# ============================================================

def ensure_model_file():

    if os.path.exists(
        MODEL_PATH
    ):
        return

    if not AIP_STORAGE_URI:
        raise RuntimeError(
            "model.bst was not found locally and "
            "AIP_STORAGE_URI is empty."
        )

    parsed = urlparse(
        AIP_STORAGE_URI
    )

    if parsed.scheme != "gs":
        raise RuntimeError(
            "AIP_STORAGE_URI must be a gs:// URI."
        )

    bucket_name = (
        parsed.netloc
    )

    prefix = (
        parsed.path
        .lstrip("/")
        .rstrip("/")
    )

    blob_name = (
        f"{prefix}/model.bst"
        if prefix
        else "model.bst"
    )

    os.makedirs(
        os.path.dirname(
            MODEL_PATH
        ),
        exist_ok=True,
    )

    client = storage.Client()

    bucket = client.bucket(
        bucket_name
    )

    blob = bucket.blob(
        blob_name
    )

    if not blob.exists():
        raise RuntimeError(
            "model.bst was not found under "
            f"{AIP_STORAGE_URI}"
        )

    blob.download_to_filename(
        MODEL_PATH
    )


ensure_model_file()

booster = xgb.Booster()

booster.load_model(
    MODEL_PATH
)

booster_feature_names = (
    booster.feature_names
)

if (
    booster_feature_names
    and booster_feature_names != FEATURES
):
    raise RuntimeError(
        "XGBoost artifact feature names do not match "
        "the pipeline serving contract. "
        f"Artifact={booster_feature_names}, "
        f"Pipeline={FEATURES}"
    )

if (
    booster.num_features()
    != len(FEATURES)
):
    raise RuntimeError(
        f"XGBoost artifact expects "
        f"{booster.num_features()} features, "
        f"but pipeline serving contract contains "
        f"{len(FEATURES)}: {FEATURES}"
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
        #   model-specific numeric features
        #   =
        #   1 metadata field + len(FEATURES)
        # ====================================================

        elif isinstance(
            instance,
            list,
        ):

            expected_features = (
                len(
                    FEATURES
                )
            )

            if (
                len(instance)
                !=
                expected_features
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unexpected input field count. "
                        f"Received {len(instance)} fields; "
                        f"expected exactly {expected_features} "
                        "model features. provider_id must be "
                        "removed upstream using Vertex "
                        "instance_config.key_field."
                    ),
                )

            row = [
                float(value)
                for value in instance
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
