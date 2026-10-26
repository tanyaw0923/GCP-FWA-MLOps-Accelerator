import os
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/model/model.joblib",
)

AIP_STORAGE_URI = os.environ.get(
    "AIP_STORAGE_URI",
    "",
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

METADATA_FIELDS = [
    "provider_id",
]


def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return

    if not AIP_STORAGE_URI:
        raise RuntimeError(
            "model.joblib was not found locally and "
            "AIP_STORAGE_URI is empty."
        )

    parsed = urlparse(
        AIP_STORAGE_URI
    )

    if parsed.scheme != "gs":
        raise RuntimeError(
            "AIP_STORAGE_URI must be a gs:// URI."
        )

    bucket_name = parsed.netloc

    prefix = (
        parsed.path
        .lstrip("/")
        .rstrip("/")
    )

    blob_name = (
        f"{prefix}/model.joblib"
        if prefix
        else "model.joblib"
    )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    client = storage.Client()

    blob = (
        client
        .bucket(bucket_name)
        .blob(blob_name)
    )

    if not blob.exists():
        raise RuntimeError(
            "model.joblib was not found under "
            f"{AIP_STORAGE_URI}"
        )

    blob.download_to_filename(
        MODEL_PATH
    )


ensure_model_file()

model = joblib.load(
    MODEL_PATH
)

app = FastAPI()


class PredictionRequest(BaseModel):
    instances: list


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
    request: PredictionRequest,
):
    instances = request.instances

    if not instances:
        return {
            "predictions": []
        }

    feature_rows = []

    for instance in instances:

        if isinstance(instance, dict):
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

            feature_values = [
                float(instance[feature])
                for feature in FEATURES
            ]

        elif isinstance(instance, list):

            expected_features_only = len(
                FEATURES
            )

            expected_with_metadata = (
                len(FEATURES)
                + len(METADATA_FIELDS)
            )

            if (
                len(instance)
                == expected_features_only
            ):
                feature_values = instance

            elif (
                len(instance)
                == expected_with_metadata
            ):
                feature_values = (
                    instance[1:]
                )

            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unexpected input field count. "
                        f"Received {len(instance)} fields. "
                        f"Expected {expected_features_only} "
                        "feature-only fields or "
                        f"{expected_with_metadata} fields "
                        "including provider_id."
                    ),
                )

            feature_values = [
                float(value)
                for value in feature_values
            ]

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Each instance must be a "
                    "dictionary or list."
                ),
            )

        feature_rows.append(
            feature_values
        )

    matrix = np.asarray(
        feature_rows,
        dtype=np.float64,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(FEATURES)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Final model matrix has shape "
                f"{matrix.shape}; expected "
                f"(N, {len(FEATURES)})."
            ),
        )

    probabilities = (
        model.predict_proba(
            matrix
        )[:, 1]
    )

    return {
        "predictions": [
            float(probability)
            for probability in probabilities
        ]
    }
