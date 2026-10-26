import argparse
import os
import tempfile
from urllib.parse import urlparse

import joblib
import numpy as np
from google.cloud import bigquery, storage

from src.config.load_config import load_config


def download_gcs_file(uri, destination):
    parsed = urlparse(uri)

    if parsed.scheme != "gs":
        raise ValueError(
            "Artifact URI must be gs://... when --model-path "
            "is not supplied."
        )

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    if not blob_name:
        raise ValueError(
            "Artifact URI must point to model.joblib."
        )

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)

    if not blob.exists():
        raise RuntimeError(
            f"Model artifact does not exist: {uri}"
        )

    blob.download_to_filename(destination)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast offline Random Forest scoring smoke test. "
            "Validates the saved model artifact against one real "
            "BigQuery batch-input row without starting a Vertex "
            "BatchPredictionJob."
        )
    )

    parser.add_argument(
        "--provider-id",
        default=None,
        help=(
            "Optional provider_id to test, for example P00175. "
            "If omitted, the first batch-input row is used."
        ),
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional local model.joblib path. If omitted, the "
            "artifact is downloaded from --artifact-uri."
        ),
    )

    parser.add_argument(
        "--artifact-uri",
        default=None,
        help=(
            "Optional gs://.../model.joblib URI. If omitted, the "
            "Random Forest candidate artifact from pipeline config "
            "is used."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    project_id = config["project_id"]
    bucket_name = config["bucket"]

    rf_config = config["model"]["random_forest"]

    features = rf_config.get(
        "features",
        config["features"],
    )

    input_table = (
        config["data"][
            "random_forest_batch_input_table"
        ]
    )

    if args.artifact_uri:
        artifact_uri = args.artifact_uri
    else:
        candidate_path = (
            config["promotion"]["random_forest"][
                "candidate_artifact_path"
            ]
            .rstrip("/")
        )

        artifact_uri = (
            f"gs://{bucket_name}/"
            f"{candidate_path}/model.joblib"
        )

    print("=" * 72)
    print("RANDOM FOREST OFFLINE SCORING SMOKE TEST")
    print("=" * 72)
    print(f"Project: {project_id}")
    print(f"Input table: {input_table}")
    print(f"Feature count from config: {len(features)}")
    print("Feature order:")

    for i, feature in enumerate(features, 1):
        print(f"  {i:02d}. {feature}")

    # --------------------------------------------------------
    # Load one real scoring row.
    # --------------------------------------------------------

    selected_columns = ",\n        ".join(
        ["provider_id"] + features
    )

    if args.provider_id:
        where_clause = (
            "WHERE provider_id = @provider_id"
        )

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "provider_id",
                    "STRING",
                    args.provider_id,
                )
            ]
        )

    else:
        where_clause = ""
        job_config = None

    query = f"""
    SELECT
        {selected_columns}
    FROM `{input_table}`
    {where_clause}
    LIMIT 1
    """

    bq = bigquery.Client(
        project=project_id
    )

    rows = list(
        bq.query(
            query,
            job_config=job_config,
        ).result()
    )

    if not rows:
        raise RuntimeError(
            "No matching batch-input row was found."
        )

    row = rows[0]

    provider_id = str(
        row["provider_id"]
    )

    print("\nSelected provider:")
    print(f"  provider_id = {provider_id}")

    print("\nRaw model feature values:")

    feature_values = []

    for feature in features:
        value = row[feature]

        print(
            f"  {feature} = "
            f"{value!r} "
            f"({type(value).__name__})"
        )

        try:
            feature_values.append(
                float(value)
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "A configured model feature could not be converted "
                f"to float: feature={feature}, value={value!r}"
            ) from exc

    matrix = np.asarray(
        [feature_values],
        dtype=np.float64,
    )

    print("\nNumeric matrix passed to predict_proba():")
    print(matrix)
    print(f"Matrix shape: {matrix.shape}")

    # --------------------------------------------------------
    # Load exact model artifact.
    # --------------------------------------------------------

    cleanup_temp_file = False

    if args.model_path:
        model_path = args.model_path

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Local model file does not exist: {model_path}"
            )

        print(
            f"\nLoading local model artifact: "
            f"{model_path}"
        )

    else:
        handle = tempfile.NamedTemporaryFile(
            suffix=".joblib",
            delete=False,
        )
        model_path = handle.name
        handle.close()
        cleanup_temp_file = True

        print(
            f"\nDownloading model artifact: "
            f"{artifact_uri}"
        )

        download_gcs_file(
            artifact_uri,
            model_path,
        )

    try:
        model = joblib.load(
            model_path
        )

        model_feature_count = getattr(
            model,
            "n_features_in_",
            None,
        )

        print(
            f"Model n_features_in_: "
            f"{model_feature_count}"
        )

        if (
            model_feature_count is not None
            and model_feature_count != len(features)
        ):
            raise RuntimeError(
                "Feature-count mismatch before prediction: "
                f"artifact expects {model_feature_count}, "
                f"config supplies {len(features)}."
            )

        artifact_feature_names = (
            list(
                getattr(
                    model,
                    "feature_names_in_",
                    [],
                )
            )
        )

        if artifact_feature_names:
            artifact_feature_names = [
                str(name)
                for name in artifact_feature_names
            ]

            print(
                "Model feature_names_in_: "
                f"{artifact_feature_names}"
            )

            if artifact_feature_names != features:
                raise RuntimeError(
                    "Feature names/order mismatch before prediction. "
                    f"Artifact={artifact_feature_names}; "
                    f"config={features}"
                )

        probabilities = (
            model.predict_proba(
                matrix
            )[:, 1]
        )

        print("\nOFFLINE PREDICTION SUCCEEDED")
        print(
            f"Provider: {provider_id}"
        )
        print(
            f"Fraud probability: "
            f"{float(probabilities[0]):.8f}"
        )

        # ----------------------------------------------------
        # Diagnostic demonstration:
        # prove why provider_id must never enter the model matrix.
        # ----------------------------------------------------

        raw_bigquery_style_instance = (
            [provider_id]
            + feature_values
        )

        print("\nContract diagnostic:")
        print(
            "  Raw BigQuery-style row length: "
            f"{len(raw_bigquery_style_instance)}"
        )
        print(
            "  Model-only feature length: "
            f"{len(feature_values)}"
        )
        print(
            "  provider_id is metadata and must be removed "
            "before predict_proba()."
        )

        try:
            np.asarray(
                [raw_bigquery_style_instance],
                dtype=np.float64,
            )

        except ValueError as exc:
            print(
                "  Expected failure if provider_id leaks into "
                f"numeric conversion: {exc}"
            )

        print("\nRESULT: LOCAL MODEL + FEATURE CONTRACT ARE VALID.")
        print(
            "If Vertex batch prediction still raises "
            "'could not convert string to float: P00...', "
            "the remaining problem is in the deployed serving "
            "container / Vertex instance transformation, not "
            "the Random Forest artifact itself."
        )

    finally:
        if cleanup_temp_file:
            try:
                os.remove(
                    model_path
                )
            except OSError:
                pass


if __name__ == "__main__":
    main()
