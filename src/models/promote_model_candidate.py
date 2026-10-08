import argparse
from datetime import datetime, timezone

from google.cloud import bigquery
from google.cloud import storage

from src.config.load_config import load_config


def get_pending_candidate(
    bq_client,
    training_state_table,
):
    query = f"""
    SELECT
      model_name,
      last_training_data_month,
      candidate_status,
      candidate_data_month,
      candidate_model_resource
    FROM `{training_state_table}`
    WHERE model_name = 'random_forest'
    LIMIT 1
    """

    rows = list(
        bq_client.query(query).result()
    )

    if not rows:
        raise RuntimeError(
            "No Random Forest training state found."
        )

    candidate = rows[0]

    if candidate.candidate_status != "PENDING_APPROVAL":
        raise RuntimeError(
            "No candidate is waiting for approval. "
            f"Current status: {candidate.candidate_status}"
        )

    if candidate.candidate_data_month is None:
        raise RuntimeError(
            "Pending candidate does not have a candidate_data_month."
        )

    if not candidate.candidate_model_resource:
        raise RuntimeError(
            "Pending candidate does not have a "
            "candidate_model_resource."
        )

    return candidate


def copy_candidate_to_production(
    storage_client,
    bucket_name,
    candidate_artifact_path,
    production_artifact_path,
):
    candidate_blob_name = (
        f"{candidate_artifact_path.rstrip('/')}/"
        "model.joblib"
    )

    production_blob_name = (
        f"{production_artifact_path.rstrip('/')}/"
        "model.joblib"
    )

    bucket = storage_client.bucket(
        bucket_name
    )

    candidate_blob = bucket.blob(
        candidate_blob_name
    )

    if not candidate_blob.exists():
        raise RuntimeError(
            "Candidate model artifact does not exist:\n"
            f"gs://{bucket_name}/{candidate_blob_name}"
        )

    bucket.copy_blob(
        candidate_blob,
        bucket,
        production_blob_name,
    )

    production_blob = bucket.blob(
        production_blob_name
    )

    if not production_blob.exists():
        raise RuntimeError(
            "Production artifact copy could not be verified."
        )

    return (
        f"gs://{bucket_name}/"
        f"{production_blob_name}"
    )


def approve_candidate(
    bq_client,
    training_state_table,
    candidate_data_month,
):
    query = f"""
    UPDATE `{training_state_table}`
    SET
      last_training_data_month = @candidate_data_month,
      last_training_timestamp = CURRENT_TIMESTAMP(),
      candidate_status = 'APPROVED'
    WHERE model_name = 'random_forest'
      AND candidate_status = 'PENDING_APPROVAL'
      AND candidate_data_month = @candidate_data_month
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "candidate_data_month",
                "DATE",
                candidate_data_month,
            )
        ]
    )

    job = bq_client.query(
        query,
        job_config=job_config,
    )

    job.result()

    if job.num_dml_affected_rows != 1:
        raise RuntimeError(
            "Training state was not updated. "
            "Expected exactly one pending candidate."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manually approve and promote the "
            "Random Forest candidate model."
        )
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Approve without interactive confirmation.",
    )

    args = parser.parse_args()

    config = load_config()

    project_id = config["project_id"]
    bucket_name = config["bucket"]

    training_state_table = (
        config["monitoring"][
            "training_state_table"
        ]
    )

    candidate_artifact_path = (
        config["promotion"][
            "candidate_artifact_path"
        ]
    )

    production_artifact_path = (
        config["promotion"][
            "production_artifact_path"
        ]
    )

    bq_client = bigquery.Client(
        project=project_id
    )

    storage_client = storage.Client(
        project=project_id
    )

    candidate = get_pending_candidate(
        bq_client=bq_client,
        training_state_table=training_state_table,
    )

    print("=" * 70)
    print("Random Forest Candidate Promotion")
    print("=" * 70)

    print(
        f"Current approved data month: "
        f"{candidate.last_training_data_month}"
    )

    print(
        f"Candidate data month: "
        f"{candidate.candidate_data_month}"
    )

    print(
        f"Candidate status: "
        f"{candidate.candidate_status}"
    )

    print(
        f"Candidate model resource: "
        f"{candidate.candidate_model_resource}"
    )

    candidate_uri = (
        f"gs://{bucket_name}/"
        f"{candidate_artifact_path.rstrip('/')}/"
        "model.joblib"
    )

    production_uri = (
        f"gs://{bucket_name}/"
        f"{production_artifact_path.rstrip('/')}/"
        "model.joblib"
    )

    print(
        f"\nCandidate artifact:\n"
        f"  {candidate_uri}"
    )

    print(
        f"\nProduction artifact:\n"
        f"  {production_uri}"
    )

    print("\n" + "-" * 70)

    if not args.yes:
        approval = input(
            "Promote this candidate to production? [y/N]: "
        ).strip().lower()

        if approval not in {
            "y",
            "yes",
        }:
            print(
                "\nPromotion cancelled. "
                "Production remains unchanged."
            )
            return

    print(
        "\nPromoting candidate artifact..."
    )

    production_model_uri = (
        copy_candidate_to_production(
            storage_client=storage_client,
            bucket_name=bucket_name,
            candidate_artifact_path=(
                candidate_artifact_path
            ),
            production_artifact_path=(
                production_artifact_path
            ),
        )
    )

    print(
        "Candidate artifact promoted successfully."
    )

    print(
        "\nUpdating approved training state..."
    )

    approve_candidate(
        bq_client=bq_client,
        training_state_table=(
            training_state_table
        ),
        candidate_data_month=(
            candidate.candidate_data_month
        ),
    )

    promoted_at = (
        datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    print("\n" + "=" * 70)
    print("CANDIDATE APPROVED")
    print("=" * 70)

    print(
        f"Approved data month: "
        f"{candidate.candidate_data_month}"
    )

    print(
        "Status: APPROVED"
    )

    print(
        f"Production model artifact: "
        f"{production_model_uri}"
    )

    print(
        f"Vertex candidate resource: "
        f"{candidate.candidate_model_resource}"
    )

    print(
        f"Promoted at: "
        f"{promoted_at}"
    )

    print(
        "\nProduction is now ready to use "
        "the newly approved model."
    )


if __name__ == "__main__":
    main()

