from google.cloud import aiplatform
from google.cloud import bigquery


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "fwa-mlops-accelerator-demo"
REGION = "us-central1"

# New XGBoost model registered after switching to model.joblib
MODEL_ID = "977871007127699456"

INPUT_TABLE = (
    f"{PROJECT_ID}."
    "fraud_features."
    "xgboost_test_features_only"
)

OUTPUT_DATASET = (
    f"{PROJECT_ID}."
    "fraud_experiments"
)

MACHINE_TYPE = "n1-standard-4"


# ============================================================
# INITIALIZE CLIENTS
# ============================================================

bq = bigquery.Client(
    project=PROJECT_ID
)

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)


# ============================================================
# 1. VALIDATE INPUT TABLE
# ============================================================

print(
    "\n=============================================="
)

print(
    "XGBOOST VERTEX CONSISTENCY TEST"
)

print(
    "=============================================="
)

print(
    f"Model ID: {MODEL_ID}"
)

print(
    f"Region: {REGION}"
)

print(
    f"Machine type: {MACHINE_TYPE}"
)

print(
    f"Input table: {INPUT_TABLE}"
)


input_table = (
    bq.get_table(
        INPUT_TABLE
    )
)

print(
    f"Input rows: {input_table.num_rows}"
)


# ============================================================
# 2. DISPLAY INPUT SCHEMA
#
# Expected feature-only input:
#
#   high_acuity_pct
#   pct_99214
#   pct_99215
#   referral_rate
#   eye_procedure_pct
#   avg_paid_per_claim
#   high_acuity_oe_ratio
#
# provider_id is intentionally excluded.
# ============================================================

print(
    "\nInput schema:"
)

for field in input_table.schema:

    print(
        f"  {field.name}: "
        f"{field.field_type}"
    )


# ============================================================
# 3. LOAD REGISTERED VERTEX MODEL
# ============================================================

model = aiplatform.Model(
    model_name=MODEL_ID
)


print(
    "\n=============================================="
)

print(
    "VERTEX MODEL"
)

print(
    "=============================================="
)

print(
    f"Resource: "
    f"{model.resource_name}"
)

print(
    f"Display name: "
    f"{model.display_name}"
)


# ============================================================
# 4. BUILD BIGQUERY URIS
# ============================================================

input_uri = (
    f"bq://{INPUT_TABLE}"
)

output_uri = (
    f"bq://{OUTPUT_DATASET}"
)


print(
    "\n=============================================="
)

print(
    "STARTING BATCH PREDICTION"
)

print(
    "=============================================="
)

print(
    f"Input URI: "
    f"{input_uri}"
)

print(
    f"Output URI: "
    f"{output_uri}"
)

print(
    f"Machine type: "
    f"{MACHINE_TYPE}"
)


# ============================================================
# 5. RUN BATCH PREDICTION
#
# Custom registered models require a machine type.
#
# n1-standard-8 is being used here because smaller machine
# types previously returned temporary capacity errors.
# ============================================================

try:

    batch_job = model.batch_predict(
        job_display_name=(
            "xgboost-test-set-consistency-check"
        ),

        bigquery_source=(
            input_uri
        ),

        bigquery_destination_prefix=(
            output_uri
        ),

        instances_format=(
            "bigquery"
        ),

        predictions_format=(
            "bigquery"
        ),

        machine_type=(
            MACHINE_TYPE
        ),

        sync=True,
    )

except RuntimeError as error:

    print(
        "\n=============================================="
    )

    print(
        "BATCH PREDICTION FAILED"
    )

    print(
        "=============================================="
    )

    print(
        str(error)
    )

    print(
        "\nIf the error contains:"
    )

    print(
        "Machine type temporarily unavailable"
    )

    print(
        "\nthen Vertex could not allocate "
        f"{MACHINE_TYPE} in {REGION}."
    )

    print(
        "That is an infrastructure capacity issue, "
        "not evidence that the XGBoost model failed."
    )

    raise


# ============================================================
# 6. DISPLAY BATCH JOB RESULT
# ============================================================

print(
    "\n=============================================="
)

print(
    "BATCH JOB RESULT"
)

print(
    "=============================================="
)

print(
    f"Resource: "
    f"{batch_job.resource_name}"
)

print(
    f"State: "
    f"{batch_job.state}"
)

print(
    f"Output info: "
    f"{batch_job.output_info}"
)


# ============================================================
# 7. RESOLVE OUTPUT INFORMATION
# ============================================================

output_info = (
    batch_job.output_info
)

output_dataset_uri = (
    output_info.bigquery_output_dataset
)

output_table_name = (
    output_info.bigquery_output_table
)


if not output_dataset_uri:

    raise RuntimeError(
        "Vertex did not return a "
        "BigQuery output dataset."
    )


output_dataset = (
    output_dataset_uri
    .replace(
        "bq://",
        "",
        1,
    )
)


print(
    "\nResolved output dataset:"
)

print(
    output_dataset
)


if output_table_name:

    print(
        "\nVertex-reported output table:"
    )

    print(
        output_table_name
    )


# ============================================================
# 8. LIST OUTPUT TABLES
#
# Vertex may create:
#
#   predictions_<timestamp>
#
# or:
#
#   errors_<timestamp>
#
# The overall BatchPredictionJob can succeed even if rows
# fail inside the prediction container.
# ============================================================

tables = list(
    bq.list_tables(
        output_dataset
    )
)


print(
    "\n=============================================="
)

print(
    "OUTPUT TABLES"
)

print(
    "=============================================="
)


for table_item in tables:

    print(
        table_item.table_id
    )


# ============================================================
# 9. FIND LATEST PREDICTION / ERROR TABLE
# ============================================================

candidate_tables = []

for table_item in tables:

    if (
        table_item.table_id.startswith(
            "predictions_"
        )
        or table_item.table_id.startswith(
            "errors_"
        )
    ):

        full_table_id = (
            f"{output_dataset}."
            f"{table_item.table_id}"
        )

        table = (
            bq.get_table(
                full_table_id
            )
        )

        candidate_tables.append(
            (
                table.created,
                table_item.table_id,
                full_table_id,
            )
        )


if not candidate_tables:

    raise RuntimeError(
        "No predictions_* or errors_* "
        "table was found."
    )


candidate_tables.sort(
    key=lambda item: item[0],
    reverse=True,
)


latest_created_at = (
    candidate_tables[0][0]
)


latest_tables = [
    item
    for item in candidate_tables
    if item[0] == latest_created_at
]


# ============================================================
# 10. INSPECT LATEST OUTPUT
# ============================================================

prediction_table_found = False
error_table_found = False


for (
    created_at,
    table_name,
    full_table_id,
) in latest_tables:

    print(
        "\n=============================================="
    )

    print(
        f"INSPECTING: "
        f"{table_name}"
    )

    print(
        "=============================================="
    )

    print(
        f"Created: "
        f"{created_at}"
    )


    table = (
        bq.get_table(
            full_table_id
        )
    )


    print(
        f"Rows: "
        f"{table.num_rows}"
    )


    print(
        "\nSchema:"
    )

    for field in table.schema:

        print(
            f"  {field.name}: "
            f"{field.field_type}"
        )


    # ========================================================
    # SUCCESSFUL PREDICTIONS
    # ========================================================

    if table_name.startswith(
        "predictions_"
    ):

        prediction_table_found = True

        print(
            "\nPrediction table found."
        )


        # ----------------------------------------------------
        # Prediction distribution
        # ----------------------------------------------------

        query = f"""
        SELECT
            prediction,
            COUNT(*) AS providers

        FROM `{full_table_id}`

        GROUP BY
            prediction

        ORDER BY
            prediction
        """


        rows = list(
            bq.query(
                query
            ).result()
        )


        print(
            "\nPrediction distribution:"
        )

        for row in rows:

            print(
                f"prediction="
                f"{row.prediction} | "
                f"providers="
                f"{row.providers}"
            )


        # ----------------------------------------------------
        # Total rows
        # ----------------------------------------------------

        total_query = f"""
        SELECT
            COUNT(*) AS total_rows

        FROM `{full_table_id}`
        """


        total_row = next(
            iter(
                bq.query(
                    total_query
                ).result()
            )
        )


        print(
            f"\nTotal prediction rows: "
            f"{total_row.total_rows}"
        )


    # ========================================================
    # ROW-LEVEL ERRORS
    # ========================================================

    elif table_name.startswith(
        "errors_"
    ):

        error_table_found = True

        print(
            "\nVertex produced row-level errors."
        )

        print(
            "\nSample error rows:"
        )


        query = f"""
        SELECT
            *

        FROM `{full_table_id}`

        LIMIT 10
        """


        rows = list(
            bq.query(
                query
            ).result()
        )


        for index, row in enumerate(
            rows,
            start=1,
        ):

            print(
                "\n----------------------------------------------"
            )

            print(
                f"Error row {index}"
            )

            print(
                "----------------------------------------------"
            )

            print(
                dict(
                    row.items()
                )
            )


# ============================================================
# 11. FINAL DIAGNOSTIC
# ============================================================

print(
    "\n=============================================="
)

print(
    "CONSISTENCY TEST RESULT"
)

print(
    "=============================================="
)


if (
    prediction_table_found
    and not error_table_found
):

    print(
        "SUCCESS"
    )

    print(
        "\nVertex generated predictions "
        "without row-level errors."
    )

    print(
        "\nOffline reference result:"
    )

    print(
        "  TEST providers: 200"
    )

    print(
        "  Predicted fraud: 16"
    )

    print(
        "  Predicted normal: 184"
    )

    print(
        "  Precision: 0.2500"
    )

    print(
        "  Recall: 0.2000"
    )

    print(
        "  F1: 0.2222"
    )


elif (
    prediction_table_found
    and error_table_found
):

    print(
        "PARTIAL SUCCESS"
    )

    print(
        "\nVertex produced both predictions "
        "and row-level errors."
    )


elif error_table_found:

    print(
        "FAILED AT MODEL SERVING"
    )

    print(
        "\nThe infrastructure job ran, "
        "but individual prediction rows failed."
    )

    print(
        "\nInspect prediction_error above."
    )


else:

    print(
        "UNKNOWN OUTPUT STATE"
    )


print(
    "\n=============================================="
)

print(
    "TEST COMPLETE"
)

print(
    "=============================================="
)
