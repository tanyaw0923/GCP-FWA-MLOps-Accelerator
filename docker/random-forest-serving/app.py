import json
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

if (
    hasattr(model, "n_features_in_")
    and model.n_features_in_ != len(FEATURES)
):
    raise RuntimeError(
        f"Random Forest artifact expects "
        f"{model.n_features_in_} features, "
        f"but pipeline serving contract contains "
        f"{len(FEATURES)}: {FEATURES}"
    )

if hasattr(model, "feature_names_in_"):
    artifact_feature_names = [
        str(feature)
        for feature in model.feature_names_in_
    ]

    if artifact_feature_names != FEATURES:
        raise RuntimeError(
            "Random Forest artifact feature names/order "
            "do not match the pipeline serving contract. "
            f"Artifact={artifact_feature_names}, "
            f"Pipeline={FEATURES}"
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

            expected_features = len(
                FEATURES
            )

            if (
                len(instance)
                != expected_features
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

            feature_values = [
                float(value)
                for value in instance
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
