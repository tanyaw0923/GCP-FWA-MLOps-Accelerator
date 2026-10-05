import json
import time
import yaml
from datetime import datetime, timezone

from google.cloud import aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct, Value

from src.config.load_config import load_config


# =========================================================
# 1. Load configuration
# =========================================================
config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]

DATA_CONFIG = config["data"]
MONITORING_CONFIG = config["monitoring"]

FEATURE_TABLE = DATA_CONFIG["feature_table"]
FEEDBACK_TABLE = DATA_CONFIG["feedback_table"]
EXPERIMENT_RUNS_TABLE = DATA_CONFIG["experiment_runs_table"]
PROVIDER_PREDICTION_TABLE = DATA_CONFIG["provider_prediction_table"]
BATCH_INPUT_TABLE = DATA_CONFIG["batch_input_table"]

CURRENT_DATA_MONTH = str(
    DATA_CONFIG["current_data_month"]
)

FEATURE_COLUMNS = config["features"]


# =========================================================
# 2. Monitoring configuration
# =========================================================
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

FEATURE_SHIFT_THRESHOLD = float(
    MONITORING_CONFIG["feature_shift_threshold"]
)

PREDICTION_SHIFT_THRESHOLD = float(
    MONITORING_CONFIG["prediction_shift_threshold"]
)

BASELINE_POSITIVE_RATE = float(
    MONITORING_CONFIG["baseline_positive_rate"]
)


# =========================================================
# 3. Compiled pipeline template
# =========================================================
PIPELINE_TEMPLATE = (
    "random_forest_training_pipeline.yaml"
)


# =========================================================
# 4. Pipeline artifact location
# =========================================================
PIPELINE_ARTIFACT_BUCKET = (
    "fwa-mlops-accelerator-demo-pipeline-artifacts"
)

PIPELINE_ROOT = (
    f"gs://{PIPELINE_ARTIFACT_BUCKET}/"
    "pipeline_root/random_forest"
)


# =========================================================
# 5. Random Forest model artifact locations
# =========================================================
MODEL_OUTPUT_URI = (
    f"gs://{config['bucket']}/"
    "models/random_forest/pipeline/model.joblib"
)

MODEL_ARTIFACT_URI = (
    f"gs://{config['bucket']}/"
    "models/random_forest/pipeline/"
)


# =========================================================
# 6. Vertex Model Registry configuration
# =========================================================
MODEL_DISPLAY_NAME = (
    "fwa-random-forest"
)

SERVING_CONTAINER_IMAGE_URI = (
    "us-docker.pkg.dev/"
    "vertex-ai/prediction/"
    "sklearn-cpu.1-6:latest"
)


# =========================================================
# 7. Batch prediction output dataset
# =========================================================
BATCH_OUTPUT_DATASET = (
    f"{PROJECT_ID}.fraud_experiments"
)


# =========================================================
# 8. Runtime parameters
# =========================================================
runtime_parameter_values = {
    # -----------------------------------------------------
    # Project
    # -----------------------------------------------------
    "project_id": Value(
        string_value=PROJECT_ID
    ),

    "region": Value(
        string_value=REGION
    ),

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    "feature_table": Value(
        string_value=FEATURE_TABLE
    ),

    "feedback_table": Value(
        string_value=FEEDBACK_TABLE
    ),

    "feature_columns_json": Value(
        string_value=json.dumps(
            FEATURE_COLUMNS
        )
    ),

    # -----------------------------------------------------
    # Model artifacts
    # -----------------------------------------------------
    "model_output_uri": Value(
        string_value=MODEL_OUTPUT_URI
    ),

    "model_artifact_uri": Value(
        string_value=MODEL_ARTIFACT_URI
    ),

    # -----------------------------------------------------
    # Model Registry
    # -----------------------------------------------------
    "model_display_name": Value(
        string_value=MODEL_DISPLAY_NAME
    ),

    "serving_container_image_uri": Value(
        string_value=SERVING_CONTAINER_IMAGE_URI
    ),

    # -----------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------
    "batch_input_table": Value(
        string_value=BATCH_INPUT_TABLE
    ),

    "batch_output_dataset": Value(
        string_value=BATCH_OUTPUT_DATASET
    ),

    # -----------------------------------------------------
    # Post-processing
    # -----------------------------------------------------
    "experiment_runs_table": Value(
        string_value=EXPERIMENT_RUNS_TABLE
    ),

    "provider_prediction_table": Value(
        string_value=PROVIDER_PREDICTION_TABLE
    ),

    # -----------------------------------------------------
    # Monitoring tables
    # -----------------------------------------------------
    "data_quality_table": Value(
        string_value=DATA_QUALITY_TABLE
    ),

    "feature_drift_table": Value(
        string_value=FEATURE_DRIFT_TABLE
    ),

    "prediction_drift_table": Value(
        string_value=PREDICTION_DRIFT_TABLE
    ),

    # -----------------------------------------------------
    # Monitoring thresholds
    # -----------------------------------------------------
    "feature_shift_threshold": Value(
        number_value=FEATURE_SHIFT_THRESHOLD
    ),

    "prediction_shift_threshold": Value(
        number_value=PREDICTION_SHIFT_THRESHOLD
    ),

    "baseline_positive_rate": Value(
        number_value=BASELINE_POSITIVE_RATE
    ),

    # -----------------------------------------------------
    # Continuous Training
    # -----------------------------------------------------
    "current_data_month": Value(
        string_value=CURRENT_DATA_MONTH
    ),

    "training_state_table": Value(
        string_value=TRAINING_STATE_TABLE
    ),
}


# =========================================================
# 9. Validate runtime parameters
# =========================================================
required_parameters = [
    "project_id",
    "region",
    "feature_table",
    "feedback_table",
    "feature_columns_json",
    "model_output_uri",
    "model_artifact_uri",
    "model_display_name",
    "serving_container_image_uri",
    "batch_input_table",
    "batch_output_dataset",
    "experiment_runs_table",
    "provider_prediction_table",
    "data_quality_table",
    "feature_drift_table",
    "prediction_drift_table",
    "feature_shift_threshold",
    "prediction_shift_threshold",
    "baseline_positive_rate",
    "current_data_month",
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


# =========================================================
# 10. Load compiled KFP pipeline specification
# =========================================================
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
    pipeline_spec_dict = (
        compiled_pipeline
    )


pipeline_spec = Struct()

json_format.ParseDict(
    pipeline_spec_dict,
    pipeline_spec,
)


# =========================================================
# 11. Create Vertex runtime configuration
# =========================================================
runtime_config = (
    aiplatform_v1.types.PipelineJob.RuntimeConfig(
        gcs_output_directory=PIPELINE_ROOT,
        parameter_values=runtime_parameter_values,
    )
)


# =========================================================
# 12. Create PipelineJob definition
# =========================================================
pipeline_job = (
    aiplatform_v1.types.PipelineJob(
        display_name=(
            "random-forest-fraud-training"
        ),
        pipeline_spec=pipeline_spec,
        runtime_config=runtime_config,
    )
)


# =========================================================
# 13. Create Vertex PipelineServiceClient
# =========================================================
client = (
    aiplatform_v1.PipelineServiceClient(
        client_options={
            "api_endpoint":
                f"{REGION}-aiplatform.googleapis.com"
        }
    )
)


# =========================================================
# 14. Create unique pipeline job ID
# =========================================================
timestamp = (
    datetime.now(timezone.utc)
    .strftime("%Y%m%d%H%M%S")
)

job_id = (
    f"random-forest-fraud-training-"
    f"{timestamp}"
)

parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)


# =========================================================
# 15. Print pipeline configuration
# =========================================================
print("=" * 70)
print("Random Forest Fraud Pipeline")
print("=" * 70)

print(
    f"Project: {PROJECT_ID}"
)

print(
    f"Region: {REGION}"
)

print(
    f"Job ID: {job_id}"
)

print(
    f"Pipeline root: {PIPELINE_ROOT}"
)


print("\nTRAINING")
print("-" * 70)

print(
    f"Feature table: "
    f"{FEATURE_TABLE}"
)

print(
    f"Feedback table: "
    f"{FEEDBACK_TABLE}"
)

print(
    f"Model output URI: "
    f"{MODEL_OUTPUT_URI}"
)


print("\nFEATURES")
print("-" * 70)

for feature in FEATURE_COLUMNS:
    print(
        f"  - {feature}"
    )


print("\nMODEL REGISTRY")
print("-" * 70)

print(
    f"Model display name: "
    f"{MODEL_DISPLAY_NAME}"
)

print(
    f"Model artifact URI: "
    f"{MODEL_ARTIFACT_URI}"
)

print(
    f"Serving container: "
    f"{SERVING_CONTAINER_IMAGE_URI}"
)


print("\nBATCH PREDICTION")
print("-" * 70)

print(
    f"Batch input table: "
    f"{BATCH_INPUT_TABLE}"
)

print(
    f"Batch output dataset: "
    f"{BATCH_OUTPUT_DATASET}"
)


print("\nPOST-PROCESSING")
print("-" * 70)

print(
    f"Experiment runs table: "
    f"{EXPERIMENT_RUNS_TABLE}"
)

print(
    f"Provider prediction table: "
    f"{PROVIDER_PREDICTION_TABLE}"
)


print("\nMONITORING")
print("-" * 70)

print(
    f"Data quality table: "
    f"{DATA_QUALITY_TABLE}"
)

print(
    f"Feature drift table: "
    f"{FEATURE_DRIFT_TABLE}"
)

print(
    f"Prediction drift table: "
    f"{PREDICTION_DRIFT_TABLE}"
)

print(
    f"Feature shift threshold: "
    f"{FEATURE_SHIFT_THRESHOLD:.2%}"
)

print(
    f"Prediction shift threshold: "
    f"{PREDICTION_SHIFT_THRESHOLD:.2%}"
)

print(
    f"Baseline positive rate: "
    f"{BASELINE_POSITIVE_RATE:.2%}"
)


print("\nCONTINUOUS TRAINING")
print("-" * 70)

print(
    f"Current data month: "
    f"{CURRENT_DATA_MONTH}"
)

print(
    f"Training state table: "
    f"{TRAINING_STATE_TABLE}"
)


# =========================================================
# 16. Print runtime parameters
# =========================================================
print("\nRUNTIME PARAMETERS")
print("-" * 70)

for (
    name,
    value,
) in runtime_parameter_values.items():

    if value.string_value:
        display_value = (
            value.string_value
        )
    else:
        display_value = (
            value.number_value
        )

    print(
        f"{name}: "
        f"{display_value}"
    )


# =========================================================
# 17. Create pipeline submission request
# =========================================================
request = (
    aiplatform_v1.CreatePipelineJobRequest(
        parent=parent,
        pipeline_job=pipeline_job,
        pipeline_job_id=job_id,
    )
)


# =========================================================
# 18. Submit pipeline
# =========================================================
print("\n" + "=" * 70)
print("Submitting pipeline to Vertex AI...")
print("=" * 70)

response = client.create_pipeline_job(
    request=request
)


print("\nPipeline submitted successfully.")

print(
    f"Resource name: "
    f"{response.name}"
)

print(
    f"Display name: "
    f"{response.display_name}"
)


# =========================================================
# 19. Wait for entire pipeline execution
# =========================================================
terminal_states = {
    aiplatform_v1.types.PipelineState.PIPELINE_STATE_SUCCEEDED,
    aiplatform_v1.types.PipelineState.PIPELINE_STATE_FAILED,
    aiplatform_v1.types.PipelineState.PIPELINE_STATE_CANCELLED,
    aiplatform_v1.types.PipelineState.PIPELINE_STATE_PAUSED,
}

print("\n" + "=" * 70)
print("Waiting for full pipeline execution")
print("=" * 70)

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

    # Print only if state changes,
    # or every loop while RUNNING.
    if (
        state != last_state
        or state
        == aiplatform_v1.types.PipelineState.PIPELINE_STATE_RUNNING
    ):

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


# =========================================================
# 20. Final pipeline result
# =========================================================
print("\n" + "=" * 70)

if (
    pipeline_job_status.state
    == aiplatform_v1.types.PipelineState.PIPELINE_STATE_SUCCEEDED
):

    print(
        "FULL PIPELINE COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

else:

    print(
        "FULL PIPELINE DID NOT COMPLETE SUCCESSFULLY"
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

    print("=" * 70)

    raise RuntimeError(
        "Vertex AI pipeline failed."
    )
