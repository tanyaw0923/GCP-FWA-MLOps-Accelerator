import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import bigquery, storage

from src.config.load_config import load_config


def download_gcs_file(uri, destination):
    parsed = urlparse(uri)

    if parsed.scheme != "gs":
        raise ValueError(
            "Artifact URI must be a gs:// URI."
        )

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

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
            "Offline smoke test for the actual Random Forest "
            "custom serving app without starting a Vertex batch job."
        )
    )

    parser.add_argument(
        "--provider-id",
        default="P00175",
    )

    parser.add_argument(
        "--artifact-uri",
        default=None,
    )

    args = parser.parse_args()

    config = load_config()

    project_id = config["project_id"]
    bucket_name = config["bucket"]

    features = (
        config["model"]["random_forest"]
        .get(
            "features",
            config["features"],
        )
    )

    input_table = (
        config["data"][
            "feature_table"
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

    selected_columns = ",\n        ".join(
        ["provider_id"] + features
    )

    query = f"""
    SELECT
        {selected_columns}
    FROM `{input_table}`
    WHERE dataset_split = 'TEST'
      AND provider_id = @provider_id
    LIMIT 1
    """

    bq = bigquery.Client(
        project=project_id
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

    rows = list(
        bq.query(
            query,
            job_config=job_config,
        ).result()
    )

    if not rows:
        raise RuntimeError(
            f"Provider not found: {args.provider_id}"
        )

    row = rows[0]

    positional_instance = [
        float(row[feature])
        for feature in features
    ]

    dict_instance = {
        feature: float(row[feature])
        for feature in features
    }

    handle = tempfile.NamedTemporaryFile(
        suffix=".joblib",
        delete=False,
    )

    model_path = handle.name
    handle.close()

    try:
        download_gcs_file(
            artifact_uri,
            model_path,
        )

        os.environ["MODEL_PATH"] = model_path
        os.environ["FEATURE_COLUMNS_JSON"] = (
            json.dumps(features)
        )

        app_path = (
            Path(__file__)
            .resolve()
            .parents[2]
            / "docker"
            / "random-forest-serving"
            / "app.py"
        )

        spec = importlib.util.spec_from_file_location(
            "rf_serving_app",
            app_path,
        )

        module = importlib.util.module_from_spec(
            spec
        )

        spec.loader.exec_module(
            module
        )

        print("=" * 72)
        print("RANDOM FOREST SERVING APP OFFLINE TEST")
        print("=" * 72)
        print(
            f"Provider: {args.provider_id}"
        )
        print(
            f"Feature count: {len(features)}"
        )
        print(
            f"Features: {features}"
        )
        print(
            f"Artifact: {artifact_uri}"
        )

        positional_request = (
            module.PredictionRequest(
                instances=[
                    positional_instance
                ]
            )
        )

        positional_result = module.predict(
            positional_request
        )

        print(
            "\nPositional array request:"
        )
        print(
            positional_instance
        )
        print(
            "Prediction response:"
        )
        print(
            positional_result
        )

        dict_request = (
            module.PredictionRequest(
                instances=[
                    dict_instance
                ]
            )
        )

        dict_result = module.predict(
            dict_request
        )

        print(
            "\nDictionary request:"
        )
        print(
            dict_instance
        )
        print(
            "Prediction response:"
        )
        print(
            dict_result
        )

        print(
            "\nSERVING APP OFFLINE TEST: PASS"
        )

        print(
            "\nThe custom app accepts both expected request "
            "shapes when provider_id is excluded."
        )

        print(
            "If Vertex still sends provider_id into the model, "
            "the remaining issue is the registered Vertex model "
            "or BatchPredictionJob instance_config, not app.py."
        )

    finally:
        try:
            os.remove(
                model_path
            )
        except OSError:
            pass


if __name__ == "__main__":
    main()
