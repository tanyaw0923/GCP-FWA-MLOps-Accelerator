import argparse
import json
import time
from datetime import datetime, timezone

from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import bigquery
from google.cloud import storage

from src.config.load_config import load_config


# ============================================================
# 1. COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Run Vertex AI batch prediction for the "
        "provider-level fraud model."
    )
)

parser.add_argument(
    "--model-id",
    required=True,
    help=(
        "Vertex AI Model Registry model ID. "
        "Example: 977871007127699456"
    ),
)

parser.add_argument(
    "--run-id",
    required=False,
    default=None,
    help=(
        "Optional experiment run ID. "
        "If omitted, the latest successful run "
        "for the inferred model type is used."
    ),
)

parser.add_argument(
    "--machine-type",
    required=False,
    default="n1-standard-8",
    help=(
        "Vertex batch prediction machine type. "
        "Default: n1-standard-8"
    ),
)

args = parser.parse_args()


# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]
BUCKET = config["bucket"]
DATA_CONFIG = config["data"]

MODEL_ID = args.model_id
RUN_ID = args.run_id
MACHINE_TYPE = args.machine_type

EXPERIMENT_RUNS_TABLE = (
    DATA_CONFIG["experiment_runs_table"]
)

PROVIDER_PREDICTION_TABLE = (
    DATA_CONFIG["provider_prediction_table"]
)


# ============================================================
# 3. INITIALIZE CLIENTS
# ============================================================

bq = bigquery.Client(
    project=PROJECT_ID
)

storage_client = storage.Client(
    project=PROJECT_ID
)

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)


# ============================================================
# 4. LOAD EXACT VERTEX MODEL
# ============================================================

model = aiplatform.Model(
    model_name=MODEL_ID
)

print(
    "\n=============================================="
)
print(
    "VERTEX BATCH FRAUD SCORING"
)
print(
    "=============================================="
)
print(
    f"Model ID: {MODEL_ID}"
)
print(
    f"Model resource: {model.resource_name}"
)
print(
    f"Display name: {model.display_name}"
)
print(
    f"Machine type: {MACHINE_TYPE}"
)


# ============================================================
# 5. INFER MODEL TYPE
# ============================================================

display_name_lower = (
    model.display_name.lower()
)

if (
    "xgboost" in display_name_lower
    or "xgb" in display_name_lower
):
    MODEL_TYPE = "xgboost"

elif (
    "random_forest" in display_name_lower
    or "random-forest" in display_name_lower
    or "random forest" in display_name_lower
    or "sklearn" in display_name_lower
):
    MODEL_TYPE = "random_forest"

else:
    raise ValueError(
        "Unable to infer model type from "
        f"Vertex display name: {model.display_name}"
    )

print(
    f"Model type: {MODEL_TYPE}"
)


# ============================================================
# 6. SELECT MODEL-SPECIFIC BATCH INPUT TABLE
#
# The input table contains:
#
#   provider_id
#   + seven numeric model features
#
# provider_id is metadata. Vertex key_field removes it before
# sending the remaining numeric values to the model.
# ============================================================

if MODEL_TYPE == "xgboost":
    INPUT_TABLE = (
        DATA_CONFIG["xgboost_batch_input_table"]
    )

elif MODEL_TYPE == "random_forest":
    INPUT_TABLE = (
        DATA_CONFIG["random_forest_batch_input_table"]
    )

else:
    raise ValueError(
        f"Unsupported model type: {MODEL_TYPE}"
    )

INPUT_URI = (
    f"bq://{INPUT_TABLE}"
)

print(
    f"Input table: {INPUT_TABLE}"
)


# ============================================================
# 7. VALIDATE INPUT TABLE
# ============================================================

input_table = bq.get_table(
    INPUT_TABLE
)

if input_table.num_rows == 0:
    raise RuntimeError(
        "Batch input table is empty."
    )

input_columns = [
    field.name
    for field in input_table.schema
]

if "provider_id" not in input_columns:
    raise RuntimeError(
        "Batch input table must contain provider_id."
    )

duplicate_query = f"""
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT provider_id) AS unique_providers,
    COUNTIF(provider_id IS NULL) AS missing_provider_id

FROM
    `{INPUT_TABLE}`
"""

input_validation = next(
    iter(
        bq.query(
            duplicate_query
        ).result()
    )
)

if int(
    input_validation[
        "missing_provider_id"
    ]
) > 0:
    raise RuntimeError(
        "Batch input contains null provider_id values."
    )

if (
    int(
        input_validation["total_rows"]
    )
    != int(
        input_validation["unique_providers"]
    )
):
    raise RuntimeError(
        "provider_id must be unique in the batch input."
    )

print(
    f"Input rows: {input_table.num_rows}"
)
print(
    "Provider lineage validation: PASS"
)


# ============================================================
# 8. RESOLVE EXPERIMENT RUN
# ============================================================

if RUN_ID is None:

    run_query = f"""
    SELECT
        experiment_id,
        run_id

    FROM
        `{EXPERIMENT_RUNS_TABLE}`

    WHERE
        model_type = @model_type
        AND status = 'SUCCEEDED'

    ORDER BY
        created_at DESC

    LIMIT 1
    """

    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "model_type",
                    "STRING",
                    MODEL_TYPE,
                )
            ]
        )
    )

    run_rows = list(
        bq.query(
            run_query,
            job_config=job_config,
        ).result()
    )

    if not run_rows:
        raise RuntimeError(
            "No successful experiment run found "
            f"for model_type={MODEL_TYPE}."
        )

    EXPERIMENT_ID = (
        run_rows[0]["experiment_id"]
    )

    RUN_ID = (
        run_rows[0]["run_id"]
    )

else:

    run_query = f"""
    SELECT
        experiment_id,
        run_id

    FROM
        `{EXPERIMENT_RUNS_TABLE}`

    WHERE
        run_id = @run_id

    LIMIT 1
    """

    job_config = (
        bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "run_id",
                    "STRING",
                    RUN_ID,
                )
            ]
        )
    )

    run_rows = list(
        bq.query(
            run_query,
            job_config=job_config,
        ).result()
    )

    if not run_rows:
        raise RuntimeError(
            f"Experiment run not found: {RUN_ID}"
        )

    EXPERIMENT_ID = (
        run_rows[0]["experiment_id"]
    )

print(
    f"Experiment ID: {EXPERIMENT_ID}"
)
print(
    f"Run ID: {RUN_ID}"
)


# ============================================================
# 9. MODEL VERSION / GCS OUTPUT
# ============================================================

MODEL_VERSION = (
    f"{MODEL_TYPE}:{MODEL_ID}"
)

BATCH_TIMESTAMP = (
    datetime.now(
        timezone.utc
    )
    .strftime("%Y%m%dT%H%M%SZ")
)

OUTPUT_PREFIX = (
    "batch_predictions/"
    f"{MODEL_TYPE}/"
    f"{RUN_ID}/"
    f"{BATCH_TIMESTAMP}/"
)

OUTPUT_URI = (
    f"gs://{BUCKET}/"
    f"{OUTPUT_PREFIX}"
)

print(
    f"Model version: {MODEL_VERSION}"
)
print(
    f"Raw prediction output: {OUTPUT_URI}"
)


# ============================================================
# 10. RUN VERTEX BATCH PREDICTION
#
# Why GCS JSONL output?
#
# Vertex does not support key_field with BigQuery prediction
# output for this batch path. JSONL output DOES support it.
#
# key_field="provider_id":
#   1. removes provider_id before the model sees the instance
#   2. returns provider_id as "key" in each prediction record
#
# The seven remaining BigQuery columns preserve the same order
# used by the working XGBoost / sklearn batch input contract.
# ============================================================

job_service = (
    aiplatform_v1.JobServiceClient(
        client_options=ClientOptions(
            api_endpoint=(
                f"{REGION}-aiplatform.googleapis.com"
            )
        )
    )
)

parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)

batch_prediction_job = {
    "display_name": (
        f"provider-fraud-"
        f"{MODEL_TYPE.replace('_', '-')}-"
        f"batch"
    ),

    "model": model.resource_name,

    "input_config": {
        "instances_format": "bigquery",
        "bigquery_source": {
            "input_uri": INPUT_URI,
        },
    },

    "instance_config": {
        "key_field": "provider_id",
    },

    "output_config": {
        "predictions_format": "jsonl",
        "gcs_destination": {
            "output_uri_prefix": OUTPUT_URI,
        },
    },

    "dedicated_resources": {
        "machine_spec": {
            "machine_type": MACHINE_TYPE,
        },
        "starting_replica_count": 1,
        "max_replica_count": 1,
    },
}

batch_job = (
    job_service.create_batch_prediction_job(
        parent=parent,
        batch_prediction_job=(
            batch_prediction_job
        ),
    )
)

print(
    "\nBatch prediction job submitted: "
    f"{batch_job.name}"
)

terminal_states = {
    aiplatform_v1.JobState.JOB_STATE_SUCCEEDED,
    aiplatform_v1.JobState.JOB_STATE_FAILED,
    aiplatform_v1.JobState.JOB_STATE_CANCELLED,
    aiplatform_v1.JobState.JOB_STATE_PAUSED,
    aiplatform_v1.JobState.JOB_STATE_EXPIRED,
    (
        aiplatform_v1.JobState
        .JOB_STATE_PARTIALLY_SUCCEEDED
    ),
}

while True:

    batch_job = (
        job_service.get_batch_prediction_job(
            name=batch_job.name
        )
    )

    if batch_job.state in terminal_states:
        break

    print(
        f"Batch prediction state: "
        f"{batch_job.state}"
    )

    time.sleep(
        30
    )

if (
    batch_job.state
    != aiplatform_v1.JobState.JOB_STATE_SUCCEEDED
):

    error_message = (
        batch_job.error.message
        if batch_job.error
        else (
            "Unknown Vertex batch prediction error."
        )
    )

    raise RuntimeError(
        "Batch prediction did not succeed. "
        f"State={batch_job.state}. "
        f"Error={error_message}"
    )

print(
    "\nBatch prediction job complete."
)
print(
    f"Job resource: {batch_job.name}"
)
print(
    f"Job state: {batch_job.state}"
)
print(
    f"Output info: {batch_job.output_info}"
)


# ============================================================
# 11. FIND JSONL PREDICTION FILES
# ============================================================

output_info = (
    batch_job.output_info
)

resolved_gcs_directory = (
    output_info.gcs_output_directory
)

if resolved_gcs_directory:
    resolved_prefix = (
        resolved_gcs_directory
        .replace(
            f"gs://{BUCKET}/",
            "",
            1,
        )
        .rstrip("/")
        + "/"
    )
else:
    resolved_prefix = OUTPUT_PREFIX

print(
    f"Resolved GCS output: "
    f"gs://{BUCKET}/{resolved_prefix}"
)

bucket = storage_client.bucket(
    BUCKET
)

all_output_blobs = list(
    storage_client.list_blobs(
        BUCKET,
        prefix=resolved_prefix,
    )
)

prediction_blobs = [
    blob
    for blob in all_output_blobs
    if (
        "prediction.results" in blob.name
        or blob.name.endswith(".jsonl")
    )
    and "error" not in blob.name.lower()
]

error_blobs = [
    blob
    for blob in all_output_blobs
    if "error" in blob.name.lower()
]

if not prediction_blobs:

    if error_blobs:
        sample_error = (
            error_blobs[0]
            .download_as_text()
        )

        print(
            "\nROW-LEVEL PREDICTION ERRORS"
        )
        print(
            sample_error[:3000]
        )

    raise RuntimeError(
        "No JSONL prediction result files were found "
        f"under gs://{BUCKET}/{resolved_prefix}"
    )

print(
    f"Prediction files: "
    f"{len(prediction_blobs)}"
)


# ============================================================
# 12. PARSE KEYED PREDICTIONS
# ============================================================

def extract_score(
    prediction,
):
    """Normalize common Vertex prediction payload shapes."""

    if isinstance(
        prediction,
        (int, float),
    ):
        return float(
            prediction
        )

    if isinstance(
        prediction,
        str,
    ):
        return float(
            prediction
        )

    if isinstance(
        prediction,
        list,
    ):
        if not prediction:
            raise ValueError(
                "Prediction list is empty."
            )

        # Binary classifiers can return a one-value score or
        # a two-class probability vector. In the latter case,
        # use the positive-class probability.
        return float(
            prediction[-1]
        )

    if isinstance(
        prediction,
        dict,
    ):
        for field_name in [
            "fraud_score",
            "score",
            "probability",
            "value",
            "prediction",
        ]:
            if field_name in prediction:
                return extract_score(
                    prediction[
                        field_name
                    ]
                )

        for field_name in [
            "probabilities",
            "scores",
        ]:
            if field_name in prediction:
                return extract_score(
                    prediction[
                        field_name
                    ]
                )

    raise ValueError(
        "Unsupported prediction payload: "
        f"{prediction}"
    )


scored_providers = []

for blob in prediction_blobs:

    content = (
        blob.download_as_text()
    )

    for line_number, line in enumerate(
        content.splitlines(),
        1,
    ):

        if not line.strip():
            continue

        record = json.loads(
            line
        )

        provider_id = (
            record.get(
                "key"
            )
        )

        if provider_id is None:
            raise RuntimeError(
                "Vertex prediction record is missing "
                f"'key' in {blob.name}, "
                f"line {line_number}."
            )

        if "prediction" not in record:
            raise RuntimeError(
                "Vertex prediction record is missing "
                f"'prediction' in {blob.name}, "
                f"line {line_number}."
            )

        fraud_score = extract_score(
            record["prediction"]
        )

        scored_providers.append(
            {
                "provider_id": str(
                    provider_id
                ),
                "fraud_score": float(
                    fraud_score
                ),
            }
        )


# ============================================================
# 13. VALIDATE PREDICTION LINEAGE
# ============================================================

if (
    len(scored_providers)
    != input_table.num_rows
):
    raise RuntimeError(
        "Prediction count does not match batch input. "
        f"Input rows={input_table.num_rows}, "
        f"predictions={len(scored_providers)}."
    )

provider_ids = [
    row["provider_id"]
    for row in scored_providers
]

if (
    len(provider_ids)
    != len(set(provider_ids))
):
    raise RuntimeError(
        "Duplicate provider_id values were returned "
        "from batch prediction."
    )

print(
    "\n=============================================="
)
print(
    "RAW PREDICTION VALIDATION"
)
print(
    "=============================================="
)
print(
    f"Input providers: {input_table.num_rows}"
)
print(
    f"Predictions returned: {len(scored_providers)}"
)
print(
    "Provider lineage: PASS"
)


# ============================================================
# 14. RANK / CLASSIFY PROVIDERS
# ============================================================

scored_providers.sort(
    key=lambda row: (
        -row["fraud_score"],
        row["provider_id"],
    )
)

prediction_timestamp = (
    datetime.now(
        timezone.utc
    )
)

prediction_date = (
    prediction_timestamp
    .date()
    .isoformat()
)

rows_to_insert = []

for risk_rank, row in enumerate(
    scored_providers,
    1,
):

    fraud_score = row[
        "fraud_score"
    ]

    if fraud_score >= 0.80:
        risk_category = "HIGH"
        reason = (
            "Very high model fraud probability"
        )

    elif fraud_score >= 0.50:
        risk_category = "MEDIUM"
        reason = (
            "Elevated model fraud probability"
        )

    else:
        risk_category = "LOW"
        reason = (
            "Below investigation threshold"
        )

    rows_to_insert.append(
        {
            "prediction_date": (
                prediction_date
            ),
            "provider_id": (
                row["provider_id"]
            ),
            "experiment_id": (
                EXPERIMENT_ID
            ),
            "run_id": RUN_ID,
            "model_name": (
                MODEL_TYPE
            ),
            "model_version": (
                MODEL_VERSION
            ),
            "fraud_score": (
                fraud_score
            ),
            "risk_rank": (
                risk_rank
            ),
            "risk_category": (
                risk_category
            ),
            "reason": reason,
            "created_at": (
                prediction_timestamp
                .isoformat()
            ),
        }
    )


# ============================================================
# 15. IDEMPOTENT REPLACE FOR THIS RUN / MODEL
# ============================================================

delete_query = f"""
DELETE FROM
    `{PROVIDER_PREDICTION_TABLE}`

WHERE
    run_id = @run_id
    AND model_version = @model_version
"""

delete_config = (
    bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "run_id",
                "STRING",
                RUN_ID,
            ),
            bigquery.ScalarQueryParameter(
                "model_version",
                "STRING",
                MODEL_VERSION,
            ),
        ]
    )
)

bq.query(
    delete_query,
    job_config=delete_config,
).result()

errors = bq.insert_rows_json(
    PROVIDER_PREDICTION_TABLE,
    rows_to_insert,
)

if errors:
    raise RuntimeError(
        "Failed to write curated provider predictions: "
        f"{errors}"
    )


# ============================================================
# 16. VERIFY CURATED OUTPUT
# ============================================================

verification_query = f"""
SELECT
    COUNT(*) AS providers,

    COUNTIF(
        fraud_score >= 0.50
    ) AS predicted_fraud,

    MIN(
        fraud_score
    ) AS min_score,

    MAX(
        fraud_score
    ) AS max_score,

    AVG(
        fraud_score
    ) AS avg_score

FROM
    `{PROVIDER_PREDICTION_TABLE}`

WHERE
    run_id = @run_id
    AND model_version = @model_version
"""

verification_config = (
    bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "run_id",
                "STRING",
                RUN_ID,
            ),
            bigquery.ScalarQueryParameter(
                "model_version",
                "STRING",
                MODEL_VERSION,
            ),
        ]
    )
)

verification_row = next(
    iter(
        bq.query(
            verification_query,
            job_config=(
                verification_config
            ),
        ).result()
    )
)

print(
    "\n=============================================="
)
print(
    "CURATED PROVIDER PREDICTIONS"
)
print(
    "=============================================="
)
print(
    f"Output table: "
    f"{PROVIDER_PREDICTION_TABLE}"
)
print(
    f"Providers written: "
    f"{verification_row['providers']}"
)
print(
    f"Predicted fraud @ 0.50: "
    f"{verification_row['predicted_fraud']}"
)
print(
    f"Minimum fraud score: "
    f"{verification_row['min_score']}"
)
print(
    f"Maximum fraud score: "
    f"{verification_row['max_score']}"
)
print(
    f"Average fraud score: "
    f"{verification_row['avg_score']}"
)
print(
    "\nBatch scoring completed successfully."
)
