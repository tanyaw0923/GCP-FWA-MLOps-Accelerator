import argparse
import time
from datetime import datetime, timezone

from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import bigquery

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

PROJECT_ID = config[
    "project_id"
]

REGION = config[
    "region"
]

DATA_CONFIG = config[
    "data"
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


# ============================================================
# 6. SELECT MODEL-SPECIFIC BATCH INPUT TABLE
# ============================================================

if MODEL_TYPE == "xgboost":

    INPUT_TABLE = (
        DATA_CONFIG[
            "xgboost_batch_input_table"
        ]
    )


elif MODEL_TYPE == "random_forest":

    INPUT_TABLE = (
        DATA_CONFIG[
            "random_forest_batch_input_table"
        ]
    )


else:

    raise ValueError(
        f"Unsupported model type: "
        f"{MODEL_TYPE}"
    )


INPUT_URI = (
    f"bq://{INPUT_TABLE}"
)


OUTPUT_URI = (
    f"bq://{OUTPUT_DATASET}"
)


print(
    f"Input table: "
    f"{INPUT_TABLE}"
)


# ============================================================
# 7. VALIDATE INPUT TABLE
# ============================================================

input_table = (
    bq.get_table(
        INPUT_TABLE
    )
)


if input_table.num_rows == 0:

    raise RuntimeError(
        "Batch input table is empty."
    )


print(
    f"Input rows: "
    f"{input_table.num_rows}"
)


# ============================================================
# 8. RESOLVE EXPERIMENT RUN
#
# If --run-id is supplied, use it.
#
# Otherwise use the latest successful experiment run
# matching the model type.
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
        run_rows[0][
            "experiment_id"
        ]
    )

    RUN_ID = (
        run_rows[0][
            "run_id"
        ]
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
            f"Experiment run not found: "
            f"{RUN_ID}"
        )


    EXPERIMENT_ID = (
        run_rows[0][
            "experiment_id"
        ]
    )


print(
    f"Experiment ID: "
    f"{EXPERIMENT_ID}"
)

print(
    f"Run ID: "
    f"{RUN_ID}"
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


# Vertex key_field keeps provider_id out of the numeric model
# matrix while preserving it in the output as column "key".
#
# We use the GAPIC JobService here because the high-level
# Model.batch_predict helper does not expose instance_config.
job_service = aiplatform_v1.JobServiceClient(
    client_options=ClientOptions(
        api_endpoint=(
            f"{REGION}-aiplatform.googleapis.com"
        )
    )
)

parent = (
    f"projects/{PROJECT_ID}/locations/{REGION}"
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
        "instance_type": "array",
        "key_field": "provider_id",
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
    job_service.create_batch_prediction_job(
        parent=parent,
        batch_prediction_job=(
            batch_prediction_job
        ),
    )
)

print(
    f"\nBatch prediction job submitted: "
    f"{batch_job.name}"
)

terminal_states = {
    aiplatform_v1.JobState.JOB_STATE_SUCCEEDED,
    aiplatform_v1.JobState.JOB_STATE_FAILED,
    aiplatform_v1.JobState.JOB_STATE_CANCELLED,
    aiplatform_v1.JobState.JOB_STATE_PAUSED,
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
        else "Unknown Vertex batch prediction error."
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
    f"Job resource: "
    f"{batch_job.name}"
)

print(
    f"Job state: "
    f"{batch_job.state}"
)

print(
    f"Output info: "
    f"{batch_job.output_info}"
)


# ============================================================
# 11. RESOLVE VERTEX OUTPUT TABLE
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
        "Vertex did not return a "
        "BigQuery output dataset."
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


if reported_output_table:

    print(
        f"Vertex-reported output table: "
        f"{reported_output_table}"
    )


# ============================================================
# 12. IDENTIFY PREDICTION OR ERROR TABLE
#
# Vertex can create:
#
#   predictions_<timestamp>
#
# or:
#
#   errors_<timestamp>
#
# We use the tables created around the batch job completion
# and prioritize the exact table name reported by Vertex.
# ============================================================

tables = list(
    bq.list_tables(
        resolved_output_dataset
    )
)


prediction_table_id = None
error_table_id = None


# First try exact Vertex-reported prediction table.
if reported_output_table:

    candidate_id = (
        f"{resolved_output_dataset}."
        f"{reported_output_table}"
    )

    try:

        bq.get_table(
            candidate_id
        )

        prediction_table_id = (
            candidate_id
        )

    except Exception:

        prediction_table_id = None


# If exact output table is unavailable, find most recent
# predictions_* / errors_* tables.
if prediction_table_id is None:

    prediction_candidates = []
    error_candidates = []


    for table_item in tables:

        full_table_id = (
            f"{resolved_output_dataset}."
            f"{table_item.table_id}"
        )


        if table_item.table_id.startswith(
            "predictions_"
        ):

            table = (
                bq.get_table(
                    full_table_id
                )
            )

            prediction_candidates.append(
                (
                    table.created,
                    full_table_id,
                )
            )


        elif table_item.table_id.startswith(
            "errors_"
        ):

            table = (
                bq.get_table(
                    full_table_id
                )
            )

            error_candidates.append(
                (
                    table.created,
                    full_table_id,
                )
            )


    if prediction_candidates:

        prediction_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        prediction_table_id = (
            prediction_candidates[0][1]
        )


    if error_candidates:

        error_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        error_table_id = (
            error_candidates[0][1]
        )


# ============================================================
# 13. FAIL FAST IF ONLY ERROR TABLE EXISTS
# ============================================================

if prediction_table_id is None:

    if error_table_id:

        print(
            "\n=============================================="
        )

        print(
            "ROW-LEVEL PREDICTION ERRORS"
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
            "Vertex batch job completed but "
            "no predictions table was produced. "
            f"Inspect {error_table_id}."
        )


    raise RuntimeError(
        "No Vertex predictions table found."
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
# Your production xgboost_batch_input should contain
# provider_id so the scored provider can be linked downstream.
#
# If provider_id is not present in the raw Vertex output,
# fail instead of silently losing lineage.
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

    # key_field="provider_id" produces the output column "key".
    provider_identifier_column = (
        "key"
    )


else:

    raise RuntimeError(
        "Provider lineage is missing from the Vertex "
        "prediction output. Expected provider_id or key."
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
