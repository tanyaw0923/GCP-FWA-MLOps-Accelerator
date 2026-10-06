import json
import time
import yaml
from datetime import datetime, timezone

from google.cloud import aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct, Value

from src.config.load_config import load_config


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]
BUCKET = config["bucket"]

DATA_CONFIG = config["data"]
MONITORING_CONFIG = config["monitoring"]
PROMOTION_CONFIG = config["promotion"]

FEATURE_COLUMNS = config["features"]
FEATURE_SQL = config["feature_sql"]
DERIVED_FEATURE_SQL = config["derived_feature_sql"]


# ============================================================
# 2. DATA / ROLLING WINDOW CONFIGURATION
#
# Use Case 1:
#
# current_data_month is the main demo control.
#
# Example:
#
# 2026-09-01
#     ↓
# 2026-10-01
#
# The Vertex pipeline calculates all actual TRAIN / TEST dates.
# ============================================================

CURRENT_DATA_MONTH = str(
    DATA_CONFIG["current_data_month"]
)

ROLLING_WINDOW = (
    DATA_CONFIG["rolling_window"]
)

TRAINING_MONTHS = int(
    ROLLING_WINDOW["training_months"]
)

TESTING_MONTHS = int(
    ROLLING_WINDOW["testing_months"]
)

CLAIMS_TABLE = DATA_CONFIG["claims_table"]
FEATURE_TABLE = DATA_CONFIG["feature_table"]


# ============================================================
# 3. MONITORING / CONTINUOUS TRAINING STATE
#
# These tables are used by the retraining-decision component.
# ============================================================

DATA_QUALITY_TABLE = (
    MONITORING_CONFIG["data_quality_table"]
)

FEATURE_DRIFT_TABLE = (
    MONITORING_CONFIG["feature_drift_table"]
)

PREDICTION_DRIFT_TABLE = (
    MONITORING_CONFIG["prediction_drift_table"]
)

TRAINING_STATE_TABLE = (
    MONITORING_CONFIG["training_state_table"]
)


# ============================================================
# 4. COMPILED PIPELINE TEMPLATE
# ============================================================

PIPELINE_TEMPLATE = (
    "random_forest_training_pipeline.yaml"
)


# ============================================================
# 5. PIPELINE ARTIFACT LOCATION
# ============================================================

PIPELINE_ARTIFACT_BUCKET = (
    "fwa-mlops-accelerator-demo-pipeline-artifacts"
)

PIPELINE_ROOT = (
    f"gs://{PIPELINE_ARTIFACT_BUCKET}/"
    "pipeline_root/random_forest"
)


# ============================================================
# 6. CANDIDATE MODEL ARTIFACT
#
# Candidate and production artifacts are intentionally
# separated.
#
# The training pipeline writes only to the candidate location.
# Production stays unchanged until manual promotion.
# ============================================================

CANDIDATE_ARTIFACT_PATH = (
    PROMOTION_CONFIG[
        "candidate_artifact_path"
    ]
)

MODEL_OUTPUT_URI = (
    f"gs://{BUCKET}/"
    f"{CANDIDATE_ARTIFACT_PATH}"
    "model.joblib"
)

MODEL_ARTIFACT_URI = (
    f"gs://{BUCKET}/"
    f"{CANDIDATE_ARTIFACT_PATH}"
)


# ============================================================
# 7. VERTEX MODEL REGISTRY CONFIGURATION
#
# Registering this model does not automatically make it
# production.
# ============================================================

MODEL_DISPLAY_NAME = (
    "fwa-random-forest-candidate"
)

SERVING_CONTAINER_IMAGE_URI = (
    "us-docker.pkg.dev/"
    "vertex-ai/prediction/"
    "sklearn-cpu.1-6:latest"
)


# ============================================================
# 8. SERIALIZE CONFIG-DRIVEN FEATURES
#
# feature_sql_json:
# provider-level feature engineering definitions
#
# derived_feature_sql_json:
# provider-level derived features
#
# feature_columns_json:
# features actually used by Random Forest
#
# Use Case 2 changes these through pipeline_config.yaml rather
# than changing the pipeline code.
# ============================================================

FEATURE_SQL_JSON = json.dumps(
    FEATURE_SQL
)

DERIVED_FEATURE_SQL_JSON = json.dumps(
    DERIVED_FEATURE_SQL
)

FEATURE_COLUMNS_JSON = json.dumps(
    FEATURE_COLUMNS
)


# ============================================================
# 9. RUNTIME PIPELINE PARAMETERS
# ============================================================

runtime_parameter_values = {
    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------
    "project_id": Value(
        string_value=PROJECT_ID
    ),

    "region": Value(
        string_value=REGION
    ),

    # --------------------------------------------------------
    # Raw data / rolling window
    # --------------------------------------------------------
    "claims_table": Value(
        string_value=CLAIMS_TABLE
    ),

    "current_data_month": Value(
        string_value=CURRENT_DATA_MONTH
    ),

    "training_months": Value(
        number_value=TRAINING_MONTHS
    ),

    "testing_months": Value(
        number_value=TESTING_MONTHS
    ),

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------
    "feature_table": Value(
        string_value=FEATURE_TABLE
    ),

    "feature_sql_json": Value(
        string_value=FEATURE_SQL_JSON
    ),

    "derived_feature_sql_json": Value(
        string_value=(
            DERIVED_FEATURE_SQL_JSON
        )
    ),

    "feature_columns_json": Value(
        string_value=(
            FEATURE_COLUMNS_JSON
        )
    ),

    # --------------------------------------------------------
    # Candidate model artifact
    # --------------------------------------------------------
    "model_output_uri": Value(
        string_value=MODEL_OUTPUT_URI
    ),

    "model_artifact_uri": Value(
        string_value=MODEL_ARTIFACT_URI
    ),

    # --------------------------------------------------------
    # Vertex Model Registry
    # --------------------------------------------------------
    "model_display_name": Value(
        string_value=MODEL_DISPLAY_NAME
    ),

    "serving_container_image_uri": Value(
        string_value=(
            SERVING_CONTAINER_IMAGE_URI
        )
    ),

    # --------------------------------------------------------
    # Monitoring / continuous training
    # --------------------------------------------------------
    "data_quality_table": Value(
        string_value=DATA_QUALITY_TABLE
    ),

    "feature_drift_table": Value(
        string_value=FEATURE_DRIFT_TABLE
    ),

    "prediction_drift_table": Value(
        string_value=(
            PREDICTION_DRIFT_TABLE
        )
    ),

    "training_state_table": Value(
        string_value=TRAINING_STATE_TABLE
    ),
}


# ============================================================
# 10. VALIDATE RUNTIME PARAMETERS
# ============================================================

required_parameters = [
    "project_id",
    "region",
    "claims_table",
    "current_data_month",
    "training_months",
    "testing_months",
    "feature_table",
    "feature_sql_json",
    "derived_feature_sql_json",
    "feature_columns_json",
    "model_output_uri",
    "model_artifact_uri",
    "model_display_name",
    "serving_container_image_uri",
    "data_quality_table",
    "feature_drift_table",
    "prediction_drift_table",
    "training_state_table",
]

missing_parameters = [
    parameter
    for parameter in required_parameters
    if parameter
    not in runtime_parameter_values
]

if missing_parameters:
    raise ValueError(
        "Missing runtime pipeline parameters: "
        f"{missing_parameters}"
    )


# ============================================================
# 11. LOAD COMPILED KFP PIPELINE
# ============================================================

with open(
    PIPELINE_TEMPLATE,
    "r",
) as f:
    compiled_pipeline = yaml.safe_load(
        f
    )

if "pipelineSpec" in compiled_pipeline:
    pipeline_spec_dict = (
        compiled_pipeline[
            "pipelineSpec"
        ]
    )
else:
    pipeline_spec_dict = (
        compiled_pipeline
    )

pipeline_spec = Struct()

json_format.ParseDict(
    pipeline_spec_dict,
    pipeline_spec,
)


# ============================================================
# 12. CREATE VERTEX RUNTIME CONFIGURATION
# ============================================================

runtime_config = (
    aiplatform_v1.types.PipelineJob.RuntimeConfig(
        gcs_output_directory=(
            PIPELINE_ROOT
        ),
        parameter_values=(
            runtime_parameter_values
        ),
    )
)


# ============================================================
# 13. CREATE PIPELINE JOB
#
# Include current_data_month in the display name so different
# demo runs are easy to identify in the Vertex UI.
# ============================================================

data_month_label = (
    CURRENT_DATA_MONTH
    .replace("-", "")
)

pipeline_job = (
    aiplatform_v1.types.PipelineJob(
        display_name=(
            "rf-candidate-"
            f"{data_month_label}"
        ),
        pipeline_spec=(
            pipeline_spec
        ),
        runtime_config=(
            runtime_config
        ),
    )
)


# ============================================================
# 14. CREATE VERTEX PIPELINE CLIENT
# ============================================================

client = (
    aiplatform_v1.PipelineServiceClient(
        client_options={
            "api_endpoint":
                f"{REGION}-"
                "aiplatform.googleapis.com"
        }
    )
)


# ============================================================
# 15. CREATE UNIQUE JOB ID
# ============================================================

timestamp = (
    datetime.now(
        timezone.utc
    )
    .strftime(
        "%Y%m%d%H%M%S"
    )
)

job_id = (
    "rf-candidate-"
    f"{data_month_label}-"
    f"{timestamp}"
)

parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)


# ============================================================
# 16. PRINT DEMO CONFIGURATION
# ============================================================

print(
    "=" * 70
)
print(
    "Random Forest Continuous Training"
)
print(
    "=" * 70
)

print(
    f"Project: "
    f"{PROJECT_ID}"
)

print(
    f"Region: "
    f"{REGION}"
)

print(
    f"Job ID: "
    f"{job_id}"
)

print(
    f"Current data month: "
    f"{CURRENT_DATA_MONTH}"
)

print(
    f"Rolling training months: "
    f"{TRAINING_MONTHS}"
)

print(
    f"Rolling testing months: "
    f"{TESTING_MONTHS}"
)


# ============================================================
# 17. PRINT DATA CONFIGURATION
# ============================================================

print(
    "\nDATA"
)
print(
    "-" * 70
)

print(
    f"Claims table: "
    f"{CLAIMS_TABLE}"
)

print(
    f"Provider feature table: "
    f"{FEATURE_TABLE}"
)


# ============================================================
# 18. PRINT MODEL FEATURES
#
# This makes Use Case 2 very easy to demonstrate.
#
# Before:
# seven features
#
# After YAML change:
# avg_member_provider_distance appears here automatically.
# ============================================================

print(
    "\nRANDOM FOREST FEATURES"
)
print(
    "-" * 70
)

for feature in FEATURE_COLUMNS:
    print(
        f"  - {feature}"
    )


# ============================================================
# 19. PRINT FEATURE ENGINEERING DEFINITIONS
# ============================================================

print(
    "\nFEATURE ENGINEERING"
)
print(
    "-" * 70
)

for (
    feature_name,
    expression,
) in FEATURE_SQL.items():

    print(
        f"{feature_name}: "
        f"{expression}"
    )

for (
    feature_name,
    expression,
) in DERIVED_FEATURE_SQL.items():

    print(
        f"{feature_name}: "
        f"{expression}"
    )


# ============================================================
# 20. PRINT CANDIDATE MODEL CONFIGURATION
# ============================================================

print(
    "\nCANDIDATE MODEL"
)
print(
    "-" * 70
)

print(
    f"Model output URI: "
    f"{MODEL_OUTPUT_URI}"
)

print(
    f"Model artifact URI: "
    f"{MODEL_ARTIFACT_URI}"
)

print(
    f"Registry display name: "
    f"{MODEL_DISPLAY_NAME}"
)

print(
    "\nCandidate will NOT replace "
    "production automatically."
)


# ============================================================
# 21. PRINT CONTINUOUS TRAINING STATE
# ============================================================

print(
    "\nCONTINUOUS TRAINING"
)
print(
    "-" * 70
)

print(
    f"Training state table: "
    f"{TRAINING_STATE_TABLE}"
)

print(
    "Pipeline will check for:"
)

print(
    "  - new monthly data"
)

print(
    "  - data quality alert"
)

print(
    "  - feature drift"
)

print(
    "  - prediction drift"
)


# ============================================================
# 22. CREATE PIPELINE SUBMISSION REQUEST
# ============================================================

request = (
    aiplatform_v1.CreatePipelineJobRequest(
        parent=parent,
        pipeline_job=(
            pipeline_job
        ),
        pipeline_job_id=(
            job_id
        ),
    )
)


# ============================================================
# 23. SUBMIT PIPELINE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Submitting candidate pipeline "
    "to Vertex AI..."
)

print(
    "=" * 70
)

response = (
    client.create_pipeline_job(
        request=request
    )
)

print(
    "\nPipeline submitted successfully."
)

print(
    f"Resource name: "
    f"{response.name}"
)

print(
    f"Display name: "
    f"{response.display_name}"
)


# ============================================================
# 24. WAIT FOR PIPELINE EXECUTION
# ============================================================

terminal_states = {
    (
        aiplatform_v1.types
        .PipelineState
        .PIPELINE_STATE_SUCCEEDED
    ),
    (
        aiplatform_v1.types
        .PipelineState
        .PIPELINE_STATE_FAILED
    ),
    (
        aiplatform_v1.types
        .PipelineState
        .PIPELINE_STATE_CANCELLED
    ),
    (
        aiplatform_v1.types
        .PipelineState
        .PIPELINE_STATE_PAUSED
    ),
}

print(
    "\n"
    + "=" * 70
)

print(
    "Waiting for candidate pipeline"
)

print(
    "=" * 70
)

last_state = None

while True:
    pipeline_job_status = (
        client.get_pipeline_job(
            name=response.name
        )
    )

    state = (
        pipeline_job_status.state
    )

    if state != last_state:
        current_time = (
            datetime.now()
            .strftime(
                "%H:%M:%S"
            )
        )

        print(
            f"[{current_time}] "
            f"Pipeline state: "
            f"{state.name}"
        )

        last_state = state

    if state in terminal_states:
        break

    time.sleep(
        30
    )


# ============================================================
# 25. FINAL RESULT
# ============================================================

print(
    "\n"
    + "=" * 70
)

if (
    pipeline_job_status.state
    ==
    aiplatform_v1.types
    .PipelineState
    .PIPELINE_STATE_SUCCEEDED
):
    print(
        "CANDIDATE PIPELINE COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nData month: "
        f"{CURRENT_DATA_MONTH}"
    )

    print(
        f"Features used: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nIf retraining was triggered, "
        "the candidate should now be "
        "PENDING_APPROVAL."
    )

    print(
        "\nProduction model remains "
        "unchanged until manual approval."
    )

else:
    print(
        "CANDIDATE PIPELINE FAILED"
    )

    print(
        "=" * 70
    )

    print(
        f"Final state: "
        f"{pipeline_job_status.state.name}"
    )

    if pipeline_job_status.error:
        print(
            f"Error: "
            f"{pipeline_job_status.error}"
        )

    raise RuntimeError(
        "Vertex AI candidate pipeline failed."
    )

