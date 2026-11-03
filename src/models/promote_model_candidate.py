import argparse
from datetime import datetime, timezone

from google.cloud import bigquery
from google.cloud import storage

from src.config.load_config import load_config


# ============================================================
# MODEL ARTIFACT FORMAT
# ============================================================

MODEL_FILENAMES = {
    "random_forest": "model.joblib",
    "xgboost": "model.bst",
}


# ============================================================
# 1. GET PENDING CANDIDATE
# ============================================================

def get_pending_candidate(
    bq_client,
    training_state_table,
    state_key,
):
    query = f"""
    SELECT
      model_name,
      last_training_data_month,
      candidate_status,
      candidate_data_month,
      candidate_model_resource,
      candidate_feature_signature,
      candidate_model_signature,
      candidate_model_type,
      candidate_run_id,
      approved_run_id,
      approved_feature_signature,
      approved_model_signature,
      approved_model_type
    FROM `{training_state_table}`
    WHERE model_name = @state_key
    LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "state_key",
                "STRING",
                state_key,
            )
        ]
    )

    rows = list(
        bq_client.query(
            query,
            job_config=job_config,
        ).result()
    )

    if not rows:
        raise RuntimeError(
            "No training state found for "
            f"state_key={state_key}"
        )

    candidate = rows[0]

    if candidate.candidate_status != "PENDING_APPROVAL":
        raise RuntimeError(
            "No candidate is waiting for approval. "
            f"Current status: {candidate.candidate_status}"
        )

    if candidate.candidate_data_month is None:
        raise RuntimeError(
            "Pending candidate does not have "
            "candidate_data_month."
        )

    if not candidate.candidate_model_type:
        raise RuntimeError(
            "Pending candidate does not have "
            "candidate_model_type."
        )

    if not candidate.candidate_model_signature:
        raise RuntimeError(
            "Pending candidate does not have "
            "candidate_model_signature."
        )

    if not candidate.candidate_feature_signature:
        raise RuntimeError(
            "Pending candidate does not have "
            "candidate_feature_signature."
        )

    if not candidate.candidate_model_resource:
        raise RuntimeError(
            "Pending candidate does not have "
            "candidate_model_resource."
        )

    if not candidate.candidate_run_id:
        raise RuntimeError(
            "Pending candidate does not have "
            "candidate_run_id."
        )

    return candidate


# ============================================================
# 2. RESOLVE MODEL-SPECIFIC ARTIFACT CONFIG
# ============================================================

def get_model_artifact_config(
    config,
    model_type,
):
    if model_type not in MODEL_FILENAMES:
        raise ValueError(
            f"Unsupported candidate model type: {model_type}"
        )

    promotion_config = config["promotion"]

    if model_type not in promotion_config:
        raise ValueError(
            "No promotion configuration found for "
            f"model_type={model_type}"
        )

    model_promotion_config = (
        promotion_config[
            model_type
        ]
    )

    candidate_artifact_path = (
        model_promotion_config[
            "candidate_artifact_path"
        ]
    )

    production_artifact_path = (
        model_promotion_config[
            "production_artifact_path"
        ]
    )

    model_filename = (
        MODEL_FILENAMES[
            model_type
        ]
    )

    return {
        "candidate_artifact_path":
            candidate_artifact_path,
        "production_artifact_path":
            production_artifact_path,
        "model_filename":
            model_filename,
    }


# ============================================================
# 3. COPY CANDIDATE ARTIFACT TO PRODUCTION
# ============================================================

def copy_candidate_to_production(
    storage_client,
    bucket_name,
    candidate_artifact_path,
    production_artifact_path,
    model_filename,
):
    candidate_blob_name = (
        f"{candidate_artifact_path.rstrip('/')}/"
        f"{model_filename}"
    )

    production_blob_name = (
        f"{production_artifact_path.rstrip('/')}/"
        f"{model_filename}"
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

    print(
        "\nCopying candidate artifact..."
    )

    print(
        f"Source: "
        f"gs://{bucket_name}/{candidate_blob_name}"
    )

    print(
        f"Target: "
        f"gs://{bucket_name}/{production_blob_name}"
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
            "Production artifact copy could not "
            "be verified."
        )

    return (
        f"gs://{bucket_name}/"
        f"{production_blob_name}"
    )


# ============================================================
# 4. UPDATE APPROVED MODEL STATE
#
# Important:
#
# Candidate values become approved values only AFTER
# the production artifact has been copied successfully.
# ============================================================

def approve_candidate(
    bq_client,
    training_state_table,
    state_key,
    candidate_data_month,
):
    # Preserve the exact Vertex model resource that was approved.
    # Older training-state tables may not have this column yet.
    bq_client.query(
        f"""
        ALTER TABLE `{training_state_table}`
        ADD COLUMN IF NOT EXISTS approved_model_resource STRING
        """
    ).result()

    query = f"""
    UPDATE `{training_state_table}`

    SET
      last_training_data_month =
        candidate_data_month,

      last_training_timestamp =
        CURRENT_TIMESTAMP(),

      approved_feature_signature =
        candidate_feature_signature,

      approved_model_signature =
        candidate_model_signature,

      approved_model_type =
        candidate_model_type,

      approved_model_resource =
        candidate_model_resource,

      approved_run_id =
        candidate_run_id,

      candidate_status =
        'APPROVED'

    WHERE
      model_name = @state_key

      AND candidate_status =
        'PENDING_APPROVAL'

      AND candidate_data_month =
        @candidate_data_month
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "state_key",
                "STRING",
                state_key,
            ),
            bigquery.ScalarQueryParameter(
                "candidate_data_month",
                "DATE",
                candidate_data_month,
            ),
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


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manually approve and promote a "
            "fraud model candidate."
        )
    )

    parser.add_argument(
        "--state-key",
        default="provider_fraud_model",
        help=(
            "Stable training-state key for the "
            "fraud ML product."
        ),
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Approve without interactive "
            "confirmation."
        ),
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

    pending_status = (
        config["promotion"][
            "candidate_status"
        ][
            "pending"
        ]
    )

    approved_status = (
        config["promotion"][
            "candidate_status"
        ][
            "approved"
        ]
    )

    if pending_status != "PENDING_APPROVAL":
        raise ValueError(
            "This script currently expects "
            "pending status PENDING_APPROVAL."
        )

    if approved_status != "APPROVED":
        raise ValueError(
            "This script currently expects "
            "approved status APPROVED."
        )

    # --------------------------------------------------------
    # Clients
    # --------------------------------------------------------

    bq_client = bigquery.Client(
        project=project_id
    )

    storage_client = storage.Client(
        project=project_id
    )

    # --------------------------------------------------------
    # Read candidate
    # --------------------------------------------------------

    candidate = get_pending_candidate(
        bq_client=bq_client,
        training_state_table=(
            training_state_table
        ),
        state_key=args.state_key,
    )

    model_type = (
        candidate.candidate_model_type
    )

    # --------------------------------------------------------
    # Resolve candidate / production artifact config
    # --------------------------------------------------------

    artifact_config = (
        get_model_artifact_config(
            config=config,
            model_type=model_type,
        )
    )

    candidate_artifact_path = (
        artifact_config[
            "candidate_artifact_path"
        ]
    )

    production_artifact_path = (
        artifact_config[
            "production_artifact_path"
        ]
    )

    model_filename = (
        artifact_config[
            "model_filename"
        ]
    )

    candidate_uri = (
        f"gs://{bucket_name}/"
        f"{candidate_artifact_path.rstrip('/')}/"
        f"{model_filename}"
    )

    production_uri = (
        f"gs://{bucket_name}/"
        f"{production_artifact_path.rstrip('/')}/"
        f"{model_filename}"
    )

    # ========================================================
    # 6. DISPLAY APPROVAL SUMMARY
    # ========================================================

    print("=" * 70)
    print("FWA MODEL CANDIDATE PROMOTION")
    print("=" * 70)

    print(
        f"State key: "
        f"{args.state_key}"
    )

    print(
        f"Current approved model: "
        f"{candidate.approved_model_type}"
    )

    print(
        f"Candidate model: "
        f"{candidate.candidate_model_type}"
    )

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
        f"Current approved run ID: "
        f"{candidate.approved_run_id}"
    )

    print(
        f"Candidate run ID: "
        f"{candidate.candidate_run_id}"
    )

    print(
        "\nApproved feature signature: "
        f"{candidate.approved_feature_signature}"
    )

    print(
        "Candidate feature signature: "
        f"{candidate.candidate_feature_signature}"
    )

    print(
        "\nApproved model signature: "
        f"{candidate.approved_model_signature}"
    )

    print(
        "Candidate model signature: "
        f"{candidate.candidate_model_signature}"
    )

    print(
        "\nVertex candidate resource:"
    )

    print(
        f"  {candidate.candidate_model_resource}"
    )

    print(
        "\nCandidate artifact:"
    )

    print(
        f"  {candidate_uri}"
    )

    print(
        "\nProduction artifact:"
    )

    print(
        f"  {production_uri}"
    )

    # ========================================================
    # 7. MANUAL APPROVAL
    # ========================================================

    print("\n" + "-" * 70)

    if not args.yes:
        approval = input(
            "Promote this candidate to production? "
            "[y/N]: "
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

    # ========================================================
    # 8. COPY MODEL ARTIFACT
    # ========================================================

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
            model_filename=(
                model_filename
            ),
        )
    )

    print(
        "\nCandidate artifact "
        "promoted successfully."
    )

    # ========================================================
    # 9. UPDATE APPROVED STATE
    # ========================================================

    print(
        "\nUpdating approved "
        "training state..."
    )

    approve_candidate(
        bq_client=bq_client,
        training_state_table=(
            training_state_table
        ),
        state_key=args.state_key,
        candidate_data_month=(
            candidate.candidate_data_month
        ),
    )

    promoted_at = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    # ========================================================
    # 10. FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("CANDIDATE APPROVED")
    print("=" * 70)

    print(
        f"Approved model type: "
        f"{model_type}"
    )

    print(
        f"Approved data month: "
        f"{candidate.candidate_data_month}"
    )

    print(
        f"Approved run ID: "
        f"{candidate.candidate_run_id}"
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
        "\nProduction can now use "
        "the newly approved candidate."
    )


if __name__ == "__main__":
    main()
