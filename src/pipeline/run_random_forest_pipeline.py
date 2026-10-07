import hashlib
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
# 2. FEATURE SIGNATURE
#
# Use Case 2:
# Detect whether the approved feature configuration changed.
#
# The signature includes:
#   - provider-level aggregation definitions
#   - derived feature definitions
#   - features passed into the model
#
# If any of these change, the signature changes.
# ============================================================

feature_signature_payload = {
    "feature_sql": FEATURE_SQL,
    "derived_feature_sql": DERIVED_FEATURE_SQL,
    "features": FEATURE_COLUMNS,
}

feature_signature_json = json.dumps(
    feature_signature_payload,
    sort_keys=True,
)

CURRENT_FEATURE_SIGNATURE = hashlib.sha256(
    feature_signature_json.encode("utf-8")
).hexdigest()


# ============================================================
# 3. DATA / ROLLING WINDOW CONFIGURATION
#
# Use Case 1:
# Change current_data_month from September to October.
#
# Use Case 2:
# Keep current_data_month unchanged and enable a new feature.
# ============================================================

CURRENT_DATA_MONTH = str(
    DATA_CONFIG["current_data_month"]
)

ROLLING_WINDOW = DATA_CONFIG["rolling_window"]

TRAINING_MONTHS = int(
    ROLLING_WINDOW["training_months"]
)

TESTING_MONTHS = int(
    ROLLING_WINDOW["testing_months"]
)

CLAIMS_TABLE = DATA_CONFIG["claims_table"]
FEATURE_TABLE = DATA_CONFIG["feature_table"]


# ============================================================
# 4. MONITORING / CONTINUOUS TRAINING STATE
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
# 5. COMPILED PIPELINE TEMPLATE
# ============================================================

PIPELINE_TEMPLATE = (
    "random_forest_training_pipeline.yaml"
)


# ============================================================
# 6. PIPELINE ARTIFACT LOCATION
# ============================================================

PIPELINE_ARTIFACT_BUCKET = (
    "fwa-mlops-accelerator-demo-pipeline-artifacts"
)

PIPELINE_ROOT = (
    f"gs://{PIPELINE_ARTIFACT_BUCKET}/"
    "pipeline_root/random_forest"
)


# ============================================================
# 7. CANDIDATE MODEL ARTIFACT LOCATION
#
# Candidate and production models remain separate.
# ============================================================

CANDIDATE_ARTIFACT_PATH = (
    PROMOTION_CONFIG["candidate_artifact_path"]
)

MODEL_OUTPUT_URI = (
    f"gs://{BUCKET}/"
    f"{CANDIDATE_ARTIFACT_PATH.rstrip('/')}/"
    "model.joblib"
)

MODEL_ARTIFACT_URI = (
    f"gs://{BUCKET}/"
    f"{CANDIDATE_ARTIFACT_PATH.rstrip('/')}/"
)


# ============================================================
# 8. VERTEX MODEL REGISTRY CONFIGURATION
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
# 9. SERIALIZE FEATURE CONFIGURATION
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
# 10. RUNTIME PIPELINE PARAMETERS
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
    # Data / continuous training
    # --------------------------------------------------------
    "claims_table": Value(
        string_value=CLAIMS_TABLE
    ),

    "current_data_month": Value(
        string_value=CURRENT_DATA_MONTH
    ),

    "current_feature_signature": Value(
        string_value=CURRENT_FEATURE_SIGNATURE
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
        string_value=DERIVED_FEATURE_SQL_JSON
    ),

    "feature_columns_json": Value(
        string_value=FEATURE_COLUMNS_JSON
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
        string_value=SERVING_CONTAINER_IMAGE_URI
    ),

    # --------------------------------------------------------
    # Monitoring / state
    # --------------------------------------------------------
    "data_quality_table": Value(
        string_value=DATA_QUALITY_TABLE
    ),

    "feature_drift_table": Value(
        string_value=FEATURE_DRIFT_TABLE
    ),

    "prediction_drift_table": Value(
        string_value=PREDICTION_DRIFT_TABLE
    ),

    "training_state_table": Value(
        string_value=TRAINING_STATE_TABLE
    ),
}


# ============================================================
# 11. VALIDATE RUNTIME PARAMETERS
# ============================================================

required_parameters = [
    "project_id",
    "region",
    "claims_table",
    "current_data_month",
    "current_feature_signature",
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
    if parameter not in runtime_parameter_values
]

if missing_parameters:
    raise ValueError(
        "Missing runtime pipeline parameters: "
        f"{missing_parameters}"
    )


# ============================================================
# 12. LOAD COMPILED KFP PIPELINE
# ============================================================

with open(
    PIPELINE_TEMPLATE,
    "r",
) as f:
    compiled_pipeline = yaml.safe_load(f)

if "pipelineSpec" in compiled_pipeline:
    pipeline_spec_dict = (
        compiled_pipeline["pipelineSpec"]
    )
else:
    pipeline_spec_dict = compiled_pipeline

pipeline_spec = Struct()

json_format.ParseDict(
    pipeline_spec_dict,
    pipeline_spec,
)


# ============================================================
# 13. CREATE VERTEX RUNTIME CONFIG
# ============================================================

runtime_config = (
    aiplatform_v1.types.PipelineJob.RuntimeConfig(
        gcs_output_directory=PIPELINE_ROOT,
        parameter_values=runtime_parameter_values,
    )
)


# ============================================================
# 14. CREATE PIPELINE JOB
# ============================================================

data_month_label = (
    CURRENT_DATA_MONTH.replace("-", "")
)

pipeline_job = (
    aiplatform_v1.types.PipelineJob(
        display_name=(
            "rf-candidate-"
            f"{data_month_label}"
        ),
        pipeline_spec=pipeline_spec,
        runtime_config=runtime_config,
    )
)


# ============================================================
# 15. CREATE VERTEX PIPELINE CLIENT
# ============================================================

client = (
    aiplatform_v1.PipelineServiceClient(
        client_options={
            "api_endpoint":
                f"{REGION}-aiplatform.googleapis.com"
        }
    )
)


# ============================================================
# 16. CREATE UNIQUE JOB ID
# ============================================================

timestamp = (
    datetime.now(timezone.utc)
    .strftime("%Y%m%d%H%M%S")
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
# 17. PRINT PIPELINE CONFIGURATION
# ============================================================

print("=" * 70)
print("Random Forest Continuous Training")
print("=" * 70)

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
    f"Training months: "
    f"{TRAINING_MONTHS}"
)

print(
    f"Testing months: "
    f"{TESTING_MONTHS}"
)


# ============================================================
# 18. PRINT FEATURE CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL FEATURES")
print("=" * 70)

for feature in FEATURE_COLUMNS:
    print(
        f"  - {feature}"
    )

print(
    f"\nFeature count: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Feature signature: "
    f"{CURRENT_FEATURE_SIGNATURE[:12]}"
)


# ============================================================
# 19. PRINT FEATURE ENGINEERING CONFIG
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

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
# 20. PRINT DATA CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("DATA")
print("=" * 70)

print(
    f"Claims table: "
    f"{CLAIMS_TABLE}"
)

print(
    f"Provider feature table: "
    f"{FEATURE_TABLE}"
)


# ============================================================
# 21. PRINT CANDIDATE MODEL CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE MODEL")
print("=" * 70)

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
    "\nCandidate does not replace production "
    "until manual approval."
)


# ============================================================
# 22. PRINT RETRAINING TRIGGER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RETRAINING TRIGGERS")
print("=" * 70)

print(
    "Use Case 1:"
)

print(
    "  New monthly data"
)

print(
    "\nUse Case 2:"
)

print(
    "  Feature configuration change"
)

print(
    "\nMonitoring:"
)

print(
    "  Data quality alert"
)

print(
    "  Feature drift alert"
)

print(
    "  Prediction drift alert"
)


# ============================================================
# 23. CREATE PIPELINE SUBMISSION REQUEST
# ============================================================

request = (
    aiplatform_v1.CreatePipelineJobRequest(
        parent=parent,
        pipeline_job=pipeline_job,
        pipeline_job_id=job_id,
    )
)


# ============================================================
# 24. SUBMIT PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("Submitting candidate pipeline to Vertex AI...")
print("=" * 70)

response = client.create_pipeline_job(
    request=request
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
# 25. WAIT FOR PIPELINE EXECUTION
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

print("\n" + "=" * 70)
print("Waiting for candidate pipeline")
print("=" * 70)

last_state = None

while True:
    pipeline_job_status = (
        client.get_pipeline_job(
            name=response.name
        )
    )

    state = pipeline_job_status.state

    if state != last_state:
        current_time = (
            datetime.now()
            .strftime("%H:%M:%S")
        )

        print(
            f"[{current_time}] "
            f"Pipeline state: "
            f"{state.name}"
        )

        last_state = state

    if state in terminal_states:
        break

    time.sleep(30)


# ============================================================
# 26. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

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

    print("=" * 70)

    print(
        f"\nData month: "
        f"{CURRENT_DATA_MONTH}"
    )

    print(
        f"Features used: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Feature signature: "
        f"{CURRENT_FEATURE_SIGNATURE[:12]}"
    )

    print(
        "\nIf retraining was triggered, "
        "the new model should now be "
        "PENDING_APPROVAL."
    )

    print(
        "\nProduction remains unchanged "
        "until manual approval."
    )

else:
    print(
        "CANDIDATE PIPELINE FAILED"
    )

    print("=" * 70)

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
