"""Opt-in, isolated GCP smoke test: RF training -> registry -> Vertex batch -> lineage.

Run against a DEDICATED disposable GCP project. No project-specific production
configuration is imported. Creates a unique dataset, bucket, model and batch job,
then attempts to delete every created resource in a finally block.

This is an integration smoke test, not a full Vertex AI Pipelines execution.
"""
import argparse
import io
import json
import re
import sys
import time
import uuid

 
FEATURES = ["high_acuity_pct", "referral_rate"]
SAMPLES = [
    {"provider_id": "SMOKE001", "high_acuity_pct": 0.12, "referral_rate": 0.03},
    {"provider_id": "SMOKE002", "high_acuity_pct": 0.79, "referral_rate": 0.38},
    {"provider_id": "SMOKE003", "high_acuity_pct": 0.44, "referral_rate": 0.17},
]
PRODUCTION_PROJECT = "fwa-mlops-accelerator-demo"
REGION_RE = re.compile(r"^[a-z]+-[a-z]+\d+$")
PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
def validate_args(args):
    if not PROJECT_RE.fullmatch(args.project):
        raise ValueError("Invalid GCP project ID")
    if args.project == PRODUCTION_PROJECT:
        raise ValueError("Refusing to run against the demo/production project")
    if not REGION_RE.fullmatch(args.region):
        raise ValueError("Invalid GCP region")
    expected_image_prefix = f"{args.region}-docker.pkg.dev/{args.project}/"
    if not args.serving_image.startswith(expected_image_prefix):
        raise ValueError(
            "Serving image must be in the explicitly selected smoke-test project "
            "and region (not the demo/production Artifact Registry)"
        )
    if not args.acknowledge_cost:
        raise ValueError("Pass --acknowledge-cost to explicitly opt in to cloud charges")
    if args.timeout_minutes < 5 or args.timeout_minutes > 120:
        raise ValueError("Timeout must be between 5 and 120 minutes")


def create_smoke_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # Fixed, synthetic inputs so feature count/order can be checked reliably.
    x = pd.DataFrame(
        [[0.05, 0.01], [0.20, 0.05], [0.68, 0.32], [0.95, 0.44]],
        columns=FEATURES,
    )
    y = [0, 0, 1, 1]
    model = RandomForestClassifier(n_estimators=12, max_depth=2, random_state=42)
    model.fit(x, y)
    assert list(model.feature_names_in_) == FEATURES
    return model


def verify_prediction_rows(rows):
    expected = {record["provider_id"] for record in SAMPLES}
    if len(rows) != len(SAMPLES):
        raise AssertionError(f"Expected {len(SAMPLES)} predictions; got {len(rows)}")
    observed = set()
    for row in rows:
        provider_id = row.get("provider_id") or row.get("key")
        if not provider_id:
            raise AssertionError("Vertex output lost provider lineage")
        if provider_id in observed:
            raise AssertionError(f"Duplicate prediction for provider {provider_id}")
        observed.add(provider_id)
        value = row.get("prediction")
        if value is None and "predictions" in row:
            value = row["predictions"]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = float(value)
        if isinstance(value, dict):
            value = value.get("prediction", value.get("score"))
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssertionError(f"Unexpected prediction value: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"Prediction is not a probability: {value!r}")
    if observed != expected:
        raise AssertionError(f"Provider lineage mismatch: {observed} vs {expected}")


def run(args):
    import joblib
    from google.cloud import aiplatform, aiplatform_v1, bigquery, storage

    terminal = {
        aiplatform_v1.JobState.JOB_STATE_SUCCEEDED,
        aiplatform_v1.JobState.JOB_STATE_FAILED,
        aiplatform_v1.JobState.JOB_STATE_CANCELLED,
        aiplatform_v1.JobState.JOB_STATE_PAUSED,
    }
    validate_args(args)  # Must complete BEFORE any cloud resource is created.
    suffix = uuid.uuid4().hex[:12]
    dataset_id = f"fwa_smoke_{suffix}"
    bucket_name = f"{args.project[:35]}-fwa-smoke-{suffix}"
    dataset_full = f"{args.project}.{dataset_id}"
    table_id = f"{dataset_full}.batch_input"
    bucket = None
    model = None
    job_name = None
    bq = bigquery.Client(project=args.project)
    gcs = storage.Client(project=args.project)
    aiplatform.init(project=args.project, location=args.region)
    api_endpoint = f"{args.region}-aiplatform.googleapis.com"
    jobs = aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )
    try:
        print(f"[smoke] Creating isolated resources in project {args.project}")
        # BigQuery dataset location must be assigned at creation time.
        # The US multi-region supports Vertex jobs within US regions.
        dataset = bigquery.Dataset(dataset_full)
        dataset.location = "US" if args.region.startswith("us-") else args.region
        bq.create_dataset(dataset, exists_ok=False)

        bucket = gcs.create_bucket(bucket_name, location=args.region)
        model_bytes = io.BytesIO()
        joblib.dump(create_smoke_model(), model_bytes)
        blob = bucket.blob(f"{suffix}/model.joblib")
        blob.upload_from_string(model_bytes.getvalue(), content_type="application/octet-stream")
        artifact_uri = f"gs://{bucket_name}/{suffix}"

        # Provider IDs are lineage keys, NEVER input features.
        schema = [bigquery.SchemaField("provider_id", "STRING", mode="REQUIRED")]
        schema += [bigquery.SchemaField(f, "FLOAT", mode="REQUIRED") for f in FEATURES]
        bq.create_table(bigquery.Table(table_id, schema=schema))
        errors = bq.insert_rows_json(table_id, SAMPLES)
        if errors:
            raise RuntimeError(f"Failed to insert synthetic providers: {errors}")

        print("[smoke] Registering synthetic Random Forest model")
        model = aiplatform.Model.upload(
            display_name=f"fwa-smoke-{suffix}",
            artifact_uri=artifact_uri,
            serving_container_image_uri=args.serving_image,
            serving_container_environment_variables={
                "FEATURE_COLUMNS_JSON": json.dumps(FEATURES),
            },
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            sync=True,
        )

        job_spec = {
            "display_name": f"fwa-smoke-batch-{suffix}",
            "model": model.resource_name,
            "input_config": {
                "instances_format": "bigquery",
                "bigquery_source": {"input_uri": f"bq://{table_id}"},
            },
            "instance_config": {
                "instance_type": "object",
                "excluded_fields": ["provider_id"],
            },
            "output_config": {
                "predictions_format": "bigquery",
                "bigquery_destination": {"output_uri": f"bq://{dataset_full}"},
            },
            "dedicated_resources": {
                "machine_spec": {"machine_type": args.machine_type},
                "starting_replica_count": 1,
                "max_replica_count": 1,
            },
        }
        print("[smoke] Submitting Vertex AI BatchPredictionJob")
        job = jobs.create_batch_prediction_job(
            parent=f"projects/{args.project}/locations/{args.region}",
            batch_prediction_job=job_spec,
        )
        job_name = job.name
        deadline = time.monotonic() + args.timeout_minutes * 60
        while job.state not in terminal:
            if time.monotonic() > deadline:
                jobs.cancel_batch_prediction_job(name=job_name)
                raise TimeoutError(f"Batch job exceeded {args.timeout_minutes} minutes")
            time.sleep(30)
            job = jobs.get_batch_prediction_job(name=job_name)
            print(f"[smoke] Batch job: {job.state.name}")
        if job.state != aiplatform_v1.JobState.JOB_STATE_SUCCEEDED:
            raise RuntimeError(f"Batch prediction failed: {job.state.name} {job.error}")

        output_name = job.output_info.bigquery_output_table
        output_dataset = job.output_info.bigquery_output_dataset.removeprefix("bq://")
        if not output_name or output_dataset != dataset_full:
            raise AssertionError(
                f"Unexpected Vertex output location: {output_dataset}.{output_name}"
            )
        # Vertex output_info.bigquery_output_table is normally just the table
        # name (e.g. predictions_20261102), not a fully qualified ID.
        output_table = (
            output_name.removeprefix("bq://")
            if "." in output_name
            else f"{output_dataset}.{output_name}"
        )
        if not output_table.startswith(dataset_full + "."):
            raise AssertionError(f"Prediction output escaped isolated dataset: {output_table}")
        rows = [
            dict(row.items()) for row in bq.query(
                f"SELECT * FROM `{output_table}`",
                location="US" if args.region.startswith("us-") else args.region,
            ).result()
        ]
        verify_prediction_rows(rows)
        print(f"[smoke] PASS: model registered, {len(rows)} batch predictions and lineage verified")
    finally:
        if not args.keep_resources:
            if job_name:
                try:
                    jobs.delete_batch_prediction_job(name=job_name)
                except Exception as exc:
                    print(f"[cleanup] Could not delete batch job {job_name}: {exc}", file=sys.stderr)
            if model:
                try:
                    model.delete()
                except Exception as exc:
                    print(f"[cleanup] Could not delete model {model.resource_name}: {exc}", file=sys.stderr)
            try:
                bq.delete_dataset(dataset_full, delete_contents=True, not_found_ok=True)
            except Exception as exc:
                print(f"[cleanup] Could not delete dataset {dataset_full}: {exc}", file=sys.stderr)
            if bucket:
                try:
                    bucket.delete(force=True)
                except Exception as exc:
                    print(f"[cleanup] Could not delete bucket {bucket_name}: {exc}", file=sys.stderr)
        else:
            print(f"[smoke] Keeping resources (explicit opt-in): {dataset_full} {bucket_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Dedicated non-production GCP project ID")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--serving-image", required=True, help="Prebuilt RF serving image in smoke project")
    parser.add_argument("--machine-type", default="n1-standard-2")
    parser.add_argument("--timeout-minutes", type=int, default=40)
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--keep-resources", action="store_true", help="Debug only; leaves billable resources")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
