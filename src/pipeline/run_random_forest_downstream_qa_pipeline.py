import json
import yaml
from datetime import datetime, timezone

from google.cloud import aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct, Value

from src.config.load_config import load_config


# ---------------------------------------------------------
# 1. Load configuration
# ---------------------------------------------------------
config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]

FEATURE_TABLE = config["data"]["feature_table"]
FEATURE_COLUMNS = config["features"]

EXPERIMENT_RUNS_TABLE = (
    config["data"]["experiment_runs_table"]
)

PROVIDER_PREDICTION_TABLE = (
    config["data"]["provider_prediction_table"]
)

BATCH_INPUT_TABLE = (
    config["data"]["batch_input_table"]
)

CURRENT_DATA_MONTH = (
    config["data"]["current_data_month"]
)

MONITORING = config["monitoring"]

DATA_QUALITY_TABLE = (
    MONITORING["data_quality_table"]
)

FEATURE_DRIFT_TABLE = (
    MONITORING["feature_drift_table"]
)

PREDICTION_DRIFT_TABLE = (
    MONITORING["prediction_drift_table"]
)

TRAINING_STATE_TABLE = (
    MONITORING["training_state_table"]
)

FEATURE_SHIFT_THRESHOLD = float(
    MONITORING["feature_shift_threshold"]
)

PREDICTION_SHIFT_THRESHOLD = float(
    MONITORING["prediction_shift_threshold"]
)

BASELINE_POSITIVE_RATE = float(
    MONITORING["baseline_positive_rate"]
)


# ---------------------------------------------------------
# 2. QA pipeline template
# ---------------------------------------------------------
PIPELINE_TEMPLATE = (
    "random_forest_downstream_qa_pipeline.yaml"
)


# ---------------------------------------------------------
# 3. QA pipeline artifact location
# ---------------------------------------------------------
PIPELINE_ARTIFACT_BUCKET = (
    "fwa-mlops-accelerator-demo-pipeline-artifacts"
)

PIPELINE_ROOT = (
    f"gs://{PIPELINE_ARTIFACT_BUCKET}/"
    "pipeline_root/random_forest_downstream_qa"
)


# ---------------------------------------------------------
# 4. Existing raw Vertex prediction table
#
# IMPORTANT:
# Replace this with a successful existing prediction table.
# It must be fully qualified:
#
# project.dataset.table
# ---------------------------------------------------------
RAW_PREDICTION_TABLE = (
    "fwa-mlops-accelerator-demo."
    "fraud_experiments."
    "predictions_2026_10_04T19_24_14_315Z_883"
)


# ---------------------------------------------------------
# 5. Validate raw prediction table format
# ---------------------------------------------------------
raw_parts = RAW_PREDICTION_TABLE.split(".")

if len(raw_parts) != 3:
    raise ValueError(
        "RAW_PREDICTION_TABLE must be fully qualified "
        "as project.dataset.table. "
        f"Received: {RAW_PREDICTION_TABLE}"
    )


# ---------------------------------------------------------
# 6. Runtime parameter values
# ---------------------------------------------------------
runtime_parameter_values = {
    "project_id": Value(
        string_value=PROJECT_ID
    ),

    "raw_prediction_table": Value(
        string_value=RAW_PREDICTION_TABLE
    ),

    "feature_table": Value(
        string_value=FEATURE_TABLE
    ),

    "feature_columns_json": Value(
        string_value=json.dumps(
            FEATURE_COLUMNS
        )
    ),

    "experiment_runs_table": Value(
        string_value=EXPERIMENT_RUNS_TABLE
    ),

    "provider_prediction_table": Value(
        string_value=PROVIDER_PREDICTION_TABLE
    ),

    "batch_input_table": Value(
        string_value=BATCH_INPUT_TABLE
    ),

    "data_quality_table": Value(
        string_value=DATA_QUALITY_TABLE
    ),

    "feature_drift_table": Value(
        string_value=FEATURE_DRIFT_TABLE
    ),

    "prediction_drift_table": Value(
        string_value=PREDICTION_DRIFT_TABLE
    ),

    "feature_shift_threshold": Value(
        number_value=FEATURE_SHIFT_THRESHOLD
    ),

    "prediction_shift_threshold": Value(
        number_value=PREDICTION_SHIFT_THRESHOLD
    ),

    "baseline_positive_rate": Value(
        number_value=BASELINE_POSITIVE_RATE
    ),

    "current_data_month": Value(
        string_value=CURRENT_DATA_MONTH
    ),

    "training_state_table": Value(
        string_value=TRAINING_STATE_TABLE
    ),
}


# ---------------------------------------------------------
# 7. Validate required runtime parameters
# ---------------------------------------------------------
required_parameters = [
    "project_id",
    "raw_prediction_table",
    "feature_table",
    "feature_columns_json",
    "experiment_runs_table",
    "provider_prediction_table",
    "batch_input_table",
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
        "Missing QA runtime parameters: "
        f"{missing_parameters}"
    )


# ---------------------------------------------------------
# 8. Load compiled QA pipeline specification
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 9. Create Vertex runtime configuration
# ---------------------------------------------------------
runtime_config = (
    aiplatform_v1.types.PipelineJob.RuntimeConfig(
        gcs_output_directory=PIPELINE_ROOT,
        parameter_values=runtime_parameter_values,
    )
)


# ---------------------------------------------------------
# 10. Create QA PipelineJob
# ---------------------------------------------------------
pipeline_job = (
    aiplatform_v1.types.PipelineJob(
        display_name=(
            "random-forest-downstream-qa"
        ),
        pipeline_spec=pipeline_spec,
        runtime_config=runtime_config,
    )
)


# ---------------------------------------------------------
# 11. Vertex Pipeline service client
# ---------------------------------------------------------
client = (
    aiplatform_v1.PipelineServiceClient(
        client_options={
            "api_endpoint":
                f"{REGION}-aiplatform.googleapis.com"
        }
    )
)


# ---------------------------------------------------------
# 12. Unique QA job ID
# ---------------------------------------------------------
timestamp = (
    datetime.now(timezone.utc)
    .strftime("%Y%m%d%H%M%S")
)

job_id = (
    f"random-forest-downstream-qa-"
    f"{timestamp}"
)

parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)


# ---------------------------------------------------------
# 13. Print QA configuration
# ---------------------------------------------------------
print("=" * 70)
print("Submitting Random Forest Downstream QA Pipeline")
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
    f"Pipeline root: "
    f"{PIPELINE_ROOT}"
)


print(
    "\nExisting raw prediction table:"
)

print(
    RAW_PREDICTION_TABLE
)


print(
    "\nPost-processing configuration:"
)

print(
    f"Feature table: "
    f"{FEATURE_TABLE}"
)

print(
    f"Experiment runs table: "
    f"{EXPERIMENT_RUNS_TABLE}"
)

print(
    f"Provider prediction table: "
    f"{PROVIDER_PREDICTION_TABLE}"
)


print(
    "\nMonitoring configuration:"
)

print(
    f"Batch input table: "
    f"{BATCH_INPUT_TABLE}"
)

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


print(
    "\nContinuous Training configuration:"
)

print(
    f"Current data month: "
    f"{CURRENT_DATA_MONTH}"
)

print(
    f"Training state table: "
    f"{TRAINING_STATE_TABLE}"
)


print(
    "\nFeatures:"
)

for feature in FEATURE_COLUMNS:
    print(
        f"  - {feature}"
    )


# ---------------------------------------------------------
# 14. Print runtime parameters
# ---------------------------------------------------------
print(
    "\nRuntime parameters submitted to Vertex:"
)

for name, value in runtime_parameter_values.items():

    if value.string_value:
        display_value = value.string_value

    else:
        display_value = value.number_value

    print(
        f"  {name}: "
        f"{display_value}"
    )


# ---------------------------------------------------------
# 15. Submit QA pipeline
# ---------------------------------------------------------
request = (
    aiplatform_v1.CreatePipelineJobRequest(
        parent=parent,
        pipeline_job=pipeline_job,
        pipeline_job_id=job_id,
    )
)

response = client.create_pipeline_job(
    request=request
)


# ---------------------------------------------------------
# 16. Confirmation
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("QA pipeline submitted successfully")
print("=" * 70)

print(
    f"Resource name: "
    f"{response.name}"
)

print(
    f"Display name: "
    f"{response.display_name}"
)


