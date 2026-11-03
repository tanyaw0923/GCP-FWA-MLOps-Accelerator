import argparse
import json
import re
import time
from datetime import datetime, timezone

from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import bigquery

from src.config.load_config import load_config
from src.models.approval_ledger import resolve_scoring_model
from src.config.feature_contract import (
    resolve_model_features,
    validate_registered_features,
    validate_source_columns,
    validate_input_columns,
)
from src.utils.observability import (
    configure_observability,
    log_event,
)


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
    required=False,
    default=None,
    help=(
        "Optional Vertex model ID. If omitted, use the active approved "
        "model in BigQuery. An explicit ID must match the approval."
    ),
)

parser.add_argument(
    "--run-id",
    required=False,
    default=None,
    help=(
        "Optional experiment run ID. "
        "If omitted, the active approved model's "
        "approved_run_id is used."
    ),
)

parser.add_argument(
    "--machine-type",
    required=False,
    default="n1-standard-4",
    help=(
        "Vertex batch prediction machine type. "
        "Default: n1-standard-4"
    ),
)

args = parser.parse_args()


# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

config = load_config()

PROJECT_ID = config[
    "project_id"
]

REGION = config[
    "region"
]

DATA_CONFIG = config[
    "data"
]

MODEL_CONFIG = config[
    "model"
]


MODEL_ID = args.model_id
RUN_ID = args.run_id
MACHINE_TYPE = args.machine_type


EXPERIMENT_RUNS_TABLE = (
    DATA_CONFIG[
        "experiment_runs_table"
    ]
)

PROVIDER_PREDICTION_TABLE = (
    DATA_CONFIG[
        "provider_prediction_table"
    ]
)


OUTPUT_DATASET = (
    f"{PROJECT_ID}."
    "fraud_experiments"
)


# ============================================================
# 3. INITIALIZE CLIENTS
# ============================================================

bq = bigquery.Client(
    project=PROJECT_ID
)


aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)

job_client = (
    aiplatform_v1.JobServiceClient(
        client_options={
            "api_endpoint": (
                f"{REGION}-"
                "aiplatform.googleapis.com"
            )
        }
    )
)

logger = configure_observability(
    project_id=PROJECT_ID,
    service_name="fwa-batch-scoring",
)

log_event(
    logger,
    "batch_scoring_started",
    fields={
        "model_id": MODEL_ID,
        "requested_run_id": RUN_ID,
        "machine_type": MACHINE_TYPE,
    },
)


# ============================================================
# 3A. ENFORCE GOVERNED APPROVAL BEFORE ANY SCORING JOB
#
# A manually supplied --model-id must match the active approved
# Vertex model in BigQuery. This prevents bypassing model rollback
# and manual candidate approval by selecting an arbitrary model.
# ============================================================

state_table = config["monitoring"]["training_state_table"]
approval_rows = list(
    bq.query(
        f"""
        SELECT approved_model_resource, approved_run_id, approved_model_type
        FROM `{state_table}`
        WHERE model_name = @state_key
        """,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "state_key", "STRING", "provider_fraud_model"
                )
            ]
        ),
    ).result()
)

if len(approval_rows) != 1:
    raise RuntimeError("Expected exactly one approved model state row")

try:
    approved = approval_rows[0]
    MODEL_RESOURCE, APPROVED_RUN_ID = resolve_scoring_model(
        MODEL_ID, approved.approved_model_resource, approved.approved_run_id
    )
    if RUN_ID is not None and RUN_ID != APPROVED_RUN_ID:
        raise ValueError("Explicit --run-id does not match the approved model run")
    RUN_ID = APPROVED_RUN_ID
    APPROVED_MODEL_TYPE = approved.approved_model_type
    if APPROVED_MODEL_TYPE not in {"random_forest", "xgboost"}:
        raise ValueError("Approved model type is missing or unsupported")
    MODEL_ID = MODEL_RESOURCE.rsplit("/", 1)[-1]
except ValueError as exc:
    raise RuntimeError(
        f"Batch inference approval gate blocked model: {exc}"
    ) from exc


# ============================================================
# 4. LOAD EXACT VERTEX MODEL
# ============================================================

model = aiplatform.Model(
    model_name=MODEL_RESOURCE
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
    f"Model ID: "
    f"{MODEL_ID}"
)

print(
    f"Model resource: "
    f"{model.resource_name}"
)

print(
    f"Display name: "
    f"{model.display_name}"
)

print(
    f"Machine type: "
    f"{MACHINE_TYPE}"
)


# ============================================================
# 5. INFER MODEL TYPE
#
# Custom XGBoost model display name:
#
#   fwa-xgboost-custom-serving
#
# Standard candidate:
#
#   fwa-xgboost-candidate
#
# Random Forest:
#
#   fwa-random-forest-candidate
# ============================================================

display_name_lower = (
    model.display_name.lower()
)


if (
    "xgboost"
    in display_name_lower
    or "xgb"
    in display_name_lower
):

    MODEL_TYPE = (
        "xgboost"
    )


elif (
    "random_forest"
    in display_name_lower
    or "random-forest"
    in display_name_lower
    or "random forest"
    in display_name_lower
    or "sklearn"
    in display_name_lower
):

    MODEL_TYPE = (
        "random_forest"
    )


else:

    raise ValueError(
        "Unable to infer model type from "
        f"Vertex display name: "
        f"{model.display_name}"
    )


print(
    f"Model type: "
    f"{MODEL_TYPE}"
)

if MODEL_TYPE != APPROVED_MODEL_TYPE:
    raise RuntimeError(
        "Registry model type differs from the approved model type: "
        f"registry={MODEL_TYPE}; approved={APPROVED_MODEL_TYPE}"
    )


# ============================================================
# 5A. PREFLIGHT: VERIFY REGISTERED VERTEX SERVING CONTRACT
#
# Fail before starting a 10-15 minute batch job when the
# registered Vertex model still points to an old serving image
# or is missing the feature contract environment variable.
# ============================================================

# The active approved Vertex resource defines its serving image.
# A restored prior release may intentionally differ from current YAML.
# Do not accidentally reject rollback solely because a newer container
# was configured after the previous production release.
registered_container_spec = model.gca_resource.container_spec
registered_serving_image = registered_container_spec.image_uri
if not registered_serving_image:
    raise RuntimeError("Approved Vertex model lacks a serving container image")
registered_env = {
    item.name: item.value for item in registered_container_spec.env
}
print(f"Approved registered serving image: {registered_serving_image}")

registered_feature_json = (
    registered_env.get(
        "FEATURE_COLUMNS_JSON"
    )
)

if not registered_feature_json:
    raise RuntimeError(
        "VERTEX MODEL PREFLIGHT FAILED: the registered model "
        "does not contain FEATURE_COLUMNS_JSON. "
        "Re-register it through the updated training pipeline."
    )

log_event(
    logger,
    "batch_model_resolved",
    fields={
        "model_id": MODEL_ID,
        "model_type": MODEL_TYPE,
        "model_display_name": model.display_name,
        "model_resource": model.resource_name,
    },
)


# ============================================================
# 6. RESOLVE MODEL FEATURE CONTRACT
#
# pipeline_config.yaml is the single source of truth.
# The active model's feature list drives:
#
#   - training
#   - serving registration
#   - batch input materialization
#   - Vertex batch inference
#
# No model-specific scoring SQL needs to be edited when features
# are added or removed.
# ============================================================

# The approved registered model, not the current candidate YAML,
# drives the scoring feature order after a governed rollback.
from src.config.feature_contract import validate_features
try:
    MODEL_FEATURES = validate_features(json.loads(registered_feature_json))
except (TypeError, ValueError) as exc:
    raise RuntimeError(
        "Approved Vertex model has invalid FEATURE_COLUMNS_JSON"
    ) from exc


print(
    "Vertex model preflight: PASS"
)

print(
    f"Model feature count: "
    f"{len(MODEL_FEATURES)}"
)

print(
    f"Model features: "
    f"{MODEL_FEATURES}"
)


# ============================================================
# 7. DYNAMICALLY MATERIALIZE SHARED BATCH INPUT
#
# The provider feature table is the reusable superset feature
# source. Before every batch prediction we create one shared
# scoring table with exactly:
#
#   provider_id + MODEL_FEATURES
#
# Example:
#   XGBoost configured with 7 features -> table has 7 features.
#   XGBoost configured with 8 features -> table has 8 features.
#   Random Forest configured with 8 -> table has 8 features.
#
# This removes RF/XGBoost-specific batch-input schemas.
# ============================================================

SOURCE_FEATURE_TABLE = (
    DATA_CONFIG[
        "feature_table"
    ]
)

INPUT_TABLE = (
    DATA_CONFIG[
        "batch_input_table"
    ]
)

source_table = (
    bq.get_table(
        SOURCE_FEATURE_TABLE
    )
)

source_columns = {
    field.name
    for field in source_table.schema
}

try:
    validate_source_columns(source_columns, MODEL_FEATURES)
except ValueError as exc:
    raise RuntimeError(
        f"Provider feature table validation failed: {exc}. "
        "Refresh provider features before scoring."
    ) from exc


feature_select_sql = ",\n".join(
    (
        f"  CAST(`{feature}` AS FLOAT64) "
        f"AS `{feature}`"
    )
    for feature in MODEL_FEATURES
)

create_batch_input_query = f"""
CREATE OR REPLACE TABLE `{INPUT_TABLE}` AS
SELECT
  CAST(provider_id AS STRING) AS provider_id,
{feature_select_sql}
FROM `{SOURCE_FEATURE_TABLE}`
WHERE dataset_split = 'TEST'
"""

print(
    "\nMaterializing batch input from pipeline_config.yaml..."
)

print(
    f"Source feature table: "
    f"{SOURCE_FEATURE_TABLE}"
)

print(
    f"Shared batch input table: "
    f"{INPUT_TABLE}"
)

bq.query(
    create_batch_input_query
).result()

input_table = (
    bq.get_table(
        INPUT_TABLE
    )
)

if input_table.num_rows == 0:
    raise RuntimeError(
        "Dynamic batch input table is empty."
    )

input_columns = [
    field.name
    for field in input_table.schema
]

try:
    validate_input_columns(input_columns, MODEL_FEATURES)
except ValueError as exc:
    raise RuntimeError(f"Dynamic batch-input schema invalid: {exc}") from exc

print(
    f"Batch input rows: "
    f"{input_table.num_rows}"
)

print(
    f"Batch input columns: "
    f"{input_columns}"
)

print(
    "Vertex instance transformation: "
    "object instance; provider_id excluded from model input "
    "and attached back to BigQuery output."
)

log_event(
    logger,
    "batch_input_materialized",
    fields={
        "source_feature_table": SOURCE_FEATURE_TABLE,
        "input_table": INPUT_TABLE,
        "input_rows": input_table.num_rows,
        "model_type": MODEL_TYPE,
        "model_features": MODEL_FEATURES,
    },
)

INPUT_URI = (
    f"bq://{INPUT_TABLE}"
)

OUTPUT_URI = (
    f"bq://{OUTPUT_DATASET}"
)


# ============================================================
# 8. RESOLVE EXPERIMENT RUN
#
# If --run-id is supplied, use it.
#
# Otherwise use the latest successful experiment run
# matching the model type.
# ============================================================

# Always use approved_run_id, even if newer experiment runs exist.
# Lineage must describe the exact model that produced the predictions.
run_query = f"""
SELECT experiment_id, run_id
FROM `{EXPERIMENT_RUNS_TABLE}`
WHERE run_id = @run_id AND model_type = @model_type
LIMIT 1
"""
run_rows = list(bq.query(
    run_query,
    job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("run_id", "STRING", RUN_ID),
        bigquery.ScalarQueryParameter("model_type", "STRING", MODEL_TYPE),
    ]),
).result())
if len(run_rows) != 1:
    raise RuntimeError(
        f"Approved experiment run not found for {MODEL_TYPE}: {RUN_ID}"
    )
EXPERIMENT_ID = run_rows[0]["experiment_id"]


print(
    f"Experiment ID: "
    f"{EXPERIMENT_ID}"
)

print(
    f"Run ID: "
    f"{RUN_ID}"
)

log_event(
    logger,
    "batch_lineage_resolved",
    fields={
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_id": MODEL_ID,
    },
)


# ============================================================
# 9. MODEL VERSION
#
# Keep the Vertex Model Registry resource ID as part of
# downstream lineage.
# ============================================================

MODEL_VERSION = (
    f"{MODEL_TYPE}:"
    f"{MODEL_ID}"
)


print(
    f"Model version: "
    f"{MODEL_VERSION}"
)


# ============================================================
# 10. RUN VERTEX BATCH PREDICTION
#
# IMPORTANT:
# aiplatform.Model.batch_predict() does not expose Vertex
# BatchPredictionJob.instance_config. We therefore use the
# lower-level JobService API.
#
# BigQuery output does not support key_field, so use:
#
# instance_type="object"
# excluded_fields=["provider_id"]
#
# Vertex removes provider_id from the instance sent to the model
# and attaches the excluded field back to the BigQuery output.
#
# The custom serving apps accept object instances and explicitly
# select FEATURES by name, so feature ordering is deterministic.
#
# This prevents identifiers such as "P00175" from ever being
# passed into RandomForestClassifier / XGBoost as a feature.
# ============================================================

print(
    "\n=============================================="
)
print(
    "STARTING BATCH PREDICTION"
)
print(
    "=============================================="
)

parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)

batch_prediction_job_spec = {
    "display_name": (
        f"provider-fraud-"
        f"{MODEL_TYPE.replace('_', '-')}-"
        f"batch"
    ),

    "model": (
        model.resource_name
    ),

    "input_config": {
        "instances_format": "bigquery",
        "bigquery_source": {
            "input_uri": INPUT_URI,
        },
    },

    "instance_config": {
        "instance_type": "object",
        "excluded_fields": [
            "provider_id"
        ],
    },

    "output_config": {
        "predictions_format": "bigquery",
        "bigquery_destination": {
            "output_uri": OUTPUT_URI,
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
    job_client.create_batch_prediction_job(
        parent=parent,
        batch_prediction_job=(
            batch_prediction_job_spec
        ),
    )
)

print(
    f"Batch job resource: "
    f"{batch_job.name}"
)

terminal_states = {
    (
        aiplatform_v1.types
        .JobState
        .JOB_STATE_SUCCEEDED
    ),
    (
        aiplatform_v1.types
        .JobState
        .JOB_STATE_FAILED
    ),
    (
        aiplatform_v1.types
        .JobState
        .JOB_STATE_CANCELLED
    ),
    (
        aiplatform_v1.types
        .JobState
        .JOB_STATE_PAUSED
    ),
}

last_state = None

while True:
    batch_job = (
        job_client.get_batch_prediction_job(
            name=batch_job.name
        )
    )

    if batch_job.state != last_state:
        print(
            f"Batch job state: "
            f"{batch_job.state.name}"
        )

        last_state = (
            batch_job.state
        )

    if batch_job.state in terminal_states:
        break

    time.sleep(30)

if (
    batch_job.state
    !=
    aiplatform_v1.types
    .JobState
    .JOB_STATE_SUCCEEDED
):
    raise RuntimeError(
        "Vertex batch prediction job failed. "
        f"State={batch_job.state.name}; "
        f"Error={batch_job.error}"
    )

print(
    "\nBatch prediction job complete."
)
print(
    f"Job resource: "
    f"{batch_job.name}"
)
print(
    f"Job state: "
    f"{batch_job.state.name}"
)
print(
    f"Output info: "
    f"{batch_job.output_info}"
)


# ============================================================
# 11. RESOLVE CURRENT JOB OUTPUT
#
# IMPORTANT:
# Never select an arbitrary "latest" predictions_* table from
# the shared dataset. That can accidentally reuse predictions
# from an earlier batch job when the current job produced only
# row-level errors.
#
# Vertex reports the table for THIS batch job. Its suffix is
# also used to locate the matching predictions/errors table.
# ============================================================

output_info = (
    batch_job.output_info
)

output_dataset_uri = (
    output_info.bigquery_output_dataset
)

reported_output_table = (
    output_info.bigquery_output_table
)

if not output_dataset_uri:
    raise RuntimeError(
        "Vertex did not return a BigQuery output dataset."
    )

if not reported_output_table:
    raise RuntimeError(
        "Vertex did not return a BigQuery output table "
        "for the current batch job."
    )

resolved_output_dataset = (
    output_dataset_uri
    .replace(
        "bq://",
        "",
        1,
    )
)

print(
    f"\nResolved output dataset: "
    f"{resolved_output_dataset}"
)
print(
    f"Vertex-reported output table: "
    f"{reported_output_table}"
)


# ============================================================
# 12. MATCH PREDICTIONS / ERRORS TO THIS EXACT JOB
# ============================================================

if reported_output_table.startswith(
    "predictions_"
):
    output_suffix = (
        reported_output_table[
            len("predictions_"):
        ]
    )

elif reported_output_table.startswith(
    "errors_"
):
    output_suffix = (
        reported_output_table[
            len("errors_"):
        ]
    )

else:
    raise RuntimeError(
        "Unexpected Vertex BigQuery output table name: "
        f"{reported_output_table}"
    )

prediction_table_id = (
    f"{resolved_output_dataset}."
    f"predictions_{output_suffix}"
)

error_table_id = (
    f"{resolved_output_dataset}."
    f"errors_{output_suffix}"
)

try:
    prediction_table = (
        bq.get_table(
            prediction_table_id
        )
    )
except Exception:
    prediction_table = None

try:
    error_table = (
        bq.get_table(
            error_table_id
        )
    )
except Exception:
    error_table = None

prediction_rows = (
    prediction_table.num_rows
    if prediction_table
    else 0
)

error_rows = (
    error_table.num_rows
    if error_table
    else 0
)

print(
    f"Current-job prediction rows: "
    f"{prediction_rows}"
)
print(
    f"Current-job error rows: "
    f"{error_rows}"
)


# ============================================================
# 13. FAIL FAST ON CURRENT-JOB ROW ERRORS
#
# For this demo pipeline we require a complete provider batch:
# every provider must receive exactly one prediction.
# Any row-level errors therefore fail the scoring run instead
# of allowing stale or partial predictions to flow downstream.
# ============================================================

if error_rows > 0:

    log_event(
        logger,
        "batch_row_errors",
        severity="ERROR",
        fields={
            "model_id": MODEL_ID,
            "run_id": RUN_ID,
            "error_rows": error_rows,
            "prediction_rows": prediction_rows,
            "error_table": error_table_id,
        },
    )

    print(
        "\n=============================================="
    )
    print(
        "CURRENT BATCH ROW-LEVEL ERRORS"
    )
    print(
        "=============================================="
    )

    error_query = f"""
    SELECT
        *

    FROM
        `{error_table_id}`

    LIMIT 5
    """

    for row in (
        bq.query(
            error_query
        ).result()
    ):
        print(
            dict(
                row.items()
            )
        )

    raise RuntimeError(
        "Current Vertex batch job produced "
        f"{error_rows} row-level prediction errors. "
        "Curated provider_predictions was NOT updated. "
        f"Inspect {error_table_id}."
    )

if prediction_rows == 0:
    raise RuntimeError(
        "Current Vertex batch job produced zero "
        "successful prediction rows. "
        "Curated provider_predictions was NOT updated."
    )

if (
    prediction_rows
    != input_table.num_rows
):
    raise RuntimeError(
        "Current batch prediction count does not match "
        "the input provider count. "
        f"Input={input_table.num_rows}, "
        f"Predictions={prediction_rows}. "
        "Curated provider_predictions was NOT updated."
    )

print(
    f"\nRaw prediction table: "
    f"{prediction_table_id}"
)


# ============================================================
# 14. VALIDATE RAW PREDICTION COLUMN
#
# The custom XGBoost container returns probabilities, but
# Vertex writes the `prediction` field into BigQuery as STRING.
#
# Example:
#
#   "0.028491"
#   "0.732199"
#
# Downstream scoring therefore uses:
#
#   SAFE_CAST(prediction AS FLOAT64)
# ============================================================

invalid_prediction_query = f"""
SELECT
    COUNT(*) AS invalid_predictions

FROM
    `{prediction_table_id}`

WHERE
    prediction IS NOT NULL
    AND SAFE_CAST(
        prediction AS FLOAT64
    ) IS NULL
"""


invalid_row = next(
    iter(
        bq.query(
            invalid_prediction_query
        ).result()
    )
)


invalid_predictions = int(
    invalid_row[
        "invalid_predictions"
    ]
)


if invalid_predictions > 0:

    raise RuntimeError(
        "Raw Vertex prediction table contains "
        f"{invalid_predictions} values that cannot "
        "be converted to FLOAT64."
    )


# ============================================================
# 15. PRINT RAW SCORE DISTRIBUTION
# ============================================================

distribution_query = f"""
SELECT
    COUNT(*) AS total_predictions,

    COUNTIF(
        SAFE_CAST(
            prediction AS FLOAT64
        ) >= 0.50
    ) AS predicted_fraud,

    COUNTIF(
        SAFE_CAST(
            prediction AS FLOAT64
        ) < 0.50
    ) AS predicted_normal,

    MIN(
        SAFE_CAST(
            prediction AS FLOAT64
        )
    ) AS min_probability,

    MAX(
        SAFE_CAST(
            prediction AS FLOAT64
        )
    ) AS max_probability,

    AVG(
        SAFE_CAST(
            prediction AS FLOAT64
        )
    ) AS avg_probability

FROM
    `{prediction_table_id}`
"""


distribution_row = next(
    iter(
        bq.query(
            distribution_query
        ).result()
    )
)


print(
    "\n=============================================="
)

print(
    "RAW PREDICTION DISTRIBUTION"
)

print(
    "=============================================="
)


print(
    f"Total predictions: "
    f"{distribution_row['total_predictions']}"
)

print(
    f"Predicted fraud @ 0.50: "
    f"{distribution_row['predicted_fraud']}"
)

print(
    f"Predicted normal @ 0.50: "
    f"{distribution_row['predicted_normal']}"
)

print(
    f"Minimum probability: "
    f"{distribution_row['min_probability']}"
)

print(
    f"Maximum probability: "
    f"{distribution_row['max_probability']}"
)

print(
    f"Average probability: "
    f"{distribution_row['avg_probability']}"
)


# ============================================================
# 16. DETERMINE PROVIDER IDENTIFIER STRATEGY
#
# Vertex instance_config.excluded_fields removes provider_id
# from the model instance and attaches it back to BigQuery output.
#
# Keep "key" as a compatibility fallback for older outputs.
# ============================================================

raw_table = (
    bq.get_table(
        prediction_table_id
    )
)

raw_columns = {
    field.name
    for field in raw_table.schema
}

if "provider_id" in raw_columns:

    provider_identifier_column = (
        "provider_id"
    )

elif "key" in raw_columns:

    provider_identifier_column = (
        "key"
    )

else:

    raise RuntimeError(
        "Provider lineage is missing from the Vertex "
        "prediction output. Expected provider_id to be "
        "attached from instance_config.excluded_fields."
    )

print(
    "Provider lineage column: "
    f"{provider_identifier_column}"
)


# ============================================================
# 17. IDEMPOTENT DELETE FOR THIS RUN / MODEL
#
# Rerunning the same scoring command should not duplicate the
# curated provider_predictions rows.
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


# ============================================================
# 18. MATERIALIZE CURATED PROVIDER PREDICTIONS
#
# IMPORTANT:
#
# Raw Vertex:
#
#   prediction STRING
#
# Curated table:
#
#   fraud_score FLOAT64
#
# We preserve the probability rather than converting it to
# a binary 0/1 flag.
#
# Ranking is therefore based on the continuous fraud score.
# ============================================================

prediction_date = (
    datetime.now(
        timezone.utc
    )
    .date()
    .isoformat()
)


materialize_query = f"""
INSERT INTO
    `{PROVIDER_PREDICTION_TABLE}`
(
    prediction_date,
    provider_id,
    experiment_id,
    run_id,
    model_name,
    model_version,
    fraud_score,
    risk_rank,
    risk_category,
    reason,
    created_at
)

WITH scored AS
(
    SELECT
        CAST(
            `{provider_identifier_column}`
            AS STRING
        ) AS provider_id,

        SAFE_CAST(
            prediction AS FLOAT64
        ) AS fraud_score

    FROM
        `{prediction_table_id}`

    WHERE
        SAFE_CAST(
            prediction AS FLOAT64
        ) IS NOT NULL
),

ranked AS
(
    SELECT
        provider_id,
        fraud_score,

        ROW_NUMBER() OVER
        (
            ORDER BY
                fraud_score DESC,
                provider_id
        ) AS risk_rank

    FROM
        scored
)

SELECT
    DATE(@prediction_date)
        AS prediction_date,

    provider_id,

    @experiment_id
        AS experiment_id,

    @run_id
        AS run_id,

    @model_name
        AS model_name,

    @model_version
        AS model_version,

    fraud_score,

    risk_rank,

    CASE
        WHEN fraud_score >= 0.80
            THEN 'HIGH'

        WHEN fraud_score >= 0.50
            THEN 'MEDIUM'

        ELSE 'LOW'
    END
        AS risk_category,

    CASE
        WHEN fraud_score >= 0.80
            THEN 'Very high model fraud probability'

        WHEN fraud_score >= 0.50
            THEN 'Elevated model fraud probability'

        ELSE 'Below investigation threshold'
    END
        AS reason,

    CURRENT_TIMESTAMP()
        AS created_at

FROM
    ranked
"""


materialize_config = (
    bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "prediction_date",
                "DATE",
                prediction_date,
            ),

            bigquery.ScalarQueryParameter(
                "experiment_id",
                "STRING",
                EXPERIMENT_ID,
            ),

            bigquery.ScalarQueryParameter(
                "run_id",
                "STRING",
                RUN_ID,
            ),

            bigquery.ScalarQueryParameter(
                "model_name",
                "STRING",
                MODEL_TYPE,
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
    materialize_query,
    job_config=materialize_config,
).result()


# ============================================================
# 19. VERIFY CURATED OUTPUT
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
            job_config=verification_config,
        ).result()
    )
)


# ============================================================
# 20. FINAL RESULT
# ============================================================

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
