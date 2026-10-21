import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone

import yaml
from dateutil.relativedelta import relativedelta
from google.cloud import aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct, Value

from src.config.load_config import load_config


# Load project configuration
config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]
BUCKET = config["bucket"]

data_config = config["data"]
model_config = config["model"]
monitoring_config = config["monitoring"]
promotion_config = config["promotion"]


# ------------------------------------------------------------------
# Active model
# ------------------------------------------------------------------

MODEL_TYPE = model_config["model_type"]

if MODEL_TYPE not in model_config:
    raise ValueError(
        f"Unsupported model_type: {MODEL_TYPE}"
    )

active_model = model_config[MODEL_TYPE]

MODEL_NAME = active_model["model_name"]
EXPERIMENT_NAME = active_model["experiment_name"]
MODEL_PARAMETERS = active_model["parameters"]

SERVING_CONTAINER_IMAGE_URI = (
    active_model["serving_container_image_uri"]
    .strip()
)


# ------------------------------------------------------------------
# Artifact format
# ------------------------------------------------------------------

if MODEL_TYPE == "random_forest":
    MODEL_ARTIFACT_FORMAT = "joblib"
    MODEL_FILENAME = "model.joblib"

elif MODEL_TYPE == "xgboost":
    MODEL_ARTIFACT_FORMAT = (
        "native_xgboost_booster"
    )
    MODEL_FILENAME = "model.bst"

else:
    raise ValueError(
        f"Unsupported model_type: {MODEL_TYPE}"
    )


# ------------------------------------------------------------------
# Feature configuration
# ------------------------------------------------------------------

FEATURE_COLUMNS = config["features"]
FEATURE_SQL = config["feature_sql"]
DERIVED_FEATURE_SQL = config[
    "derived_feature_sql"
]


def build_signature(payload):
    """Create a stable SHA-256 signature for configuration changes."""

    serialized = json.dumps(
        payload,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


CURRENT_FEATURE_SIGNATURE = build_signature(
    {
        "feature_sql": FEATURE_SQL,
        "derived_feature_sql": (
            DERIVED_FEATURE_SQL
        ),
        "features": FEATURE_COLUMNS,
    }
)


# The model signature is used by the retraining decision.
# Changes to the algorithm, parameters, artifact format, or
# serving configuration should create a new candidate.
CURRENT_MODEL_SIGNATURE = build_signature(
    {
        "model_type": MODEL_TYPE,
        "model_name": MODEL_NAME,
        "parameters": MODEL_PARAMETERS,
        "artifact_format": (
            MODEL_ARTIFACT_FORMAT
        ),
        "serving_container_image_uri": (
            SERVING_CONTAINER_IMAGE_URI
        ),
    }
)


# ------------------------------------------------------------------
# Data configuration
# ------------------------------------------------------------------

CURRENT_DATA_MONTH = str(
    data_config["current_data_month"]
)

rolling_window = data_config[
    "rolling_window"
]

TRAINING_MONTHS = int(
    rolling_window["training_months"]
)

TESTING_MONTHS = int(
    rolling_window["testing_months"]
)

CLAIMS_TABLE = data_config[
    "claims_table"
]

FEATURE_TABLE = data_config[
    "feature_table"
]

EXPERIMENT_RUNS_TABLE = data_config[
    "experiment_runs_table"
]

EXPERIMENT_METRICS_TABLE = data_config[
    "experiment_metrics_table"
]


# ------------------------------------------------------------------
# Training-window metadata
#
# The pipeline calculates the real rolling window again for
# feature generation. These values are passed separately only
# for experiment metadata.
# ------------------------------------------------------------------

current_month = datetime.strptime(
    CURRENT_DATA_MONTH,
    "%Y-%m-%d",
)

test_start_month = (
    current_month
    - relativedelta(
        months=TESTING_MONTHS - 1
    )
)

training_end_month = (
    test_start_month
    - relativedelta(months=1)
)

training_start_month = (
    training_end_month
    - relativedelta(
        months=TRAINING_MONTHS - 1
    )
)

EXPERIMENT_TRAINING_START = (
    training_start_month
    .replace(day=1)
    .strftime("%Y-%m-%d")
)

EXPERIMENT_TRAINING_END = (
    (
        test_start_month
        - relativedelta(days=1)
    )
    .strftime("%Y-%m-%d")
)


# ------------------------------------------------------------------
# Experiment metadata
# ------------------------------------------------------------------

EXPERIMENT_OWNER = (
    config
    .get("experiment_tracking", {})
    .get(
        "owner",
        "data_science_team",
    )
)

EXPERIMENT_ID = (
    hashlib.sha256(
        EXPERIMENT_NAME.encode(
            "utf-8"
        )
    )
    .hexdigest()[:16]
)

RUN_TIMESTAMP = (
    datetime.now(timezone.utc)
    .strftime("%Y%m%d%H%M%S")
)

RUN_ID = (
    f"{MODEL_TYPE}-"
    f"{CURRENT_DATA_MONTH.replace('-', '')}-"
    f"{RUN_TIMESTAMP}"
)

FEATURE_VERSION = (
    CURRENT_FEATURE_SIGNATURE
)

DATASET_VERSION = (
    CURRENT_DATA_MONTH
)


try:
    GIT_COMMIT = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

except Exception:
    GIT_COMMIT = "unknown"


# ------------------------------------------------------------------
# Monitoring tables
# ------------------------------------------------------------------

DATA_QUALITY_TABLE = (
    monitoring_config[
        "data_quality_table"
    ]
)

FEATURE_DRIFT_TABLE = (
    monitoring_config[
        "feature_drift_table"
    ]
)

PREDICTION_DRIFT_TABLE = (
    monitoring_config[
        "prediction_drift_table"
    ]
)

PERFORMANCE_DEGRADATION_TABLE = (
    monitoring_config[
        "performance_degradation_table"
    ]
)

TRAINING_STATE_TABLE = (
    monitoring_config[
        "training_state_table"
    ]
)


# ------------------------------------------------------------------
# Candidate and production artifact locations
# ------------------------------------------------------------------

if MODEL_TYPE not in promotion_config:
    raise ValueError(
        "No promotion configuration found "
        f"for model_type={MODEL_TYPE}"
    )

active_promotion = (
    promotion_config[MODEL_TYPE]
)

CANDIDATE_ARTIFACT_PATH = (
    active_promotion[
        "candidate_artifact_path"
    ]
)

PRODUCTION_ARTIFACT_PATH = (
    active_promotion[
        "production_artifact_path"
    ]
)

MODEL_OUTPUT_URI = (
    f"gs://{BUCKET}/"
    f"{CANDIDATE_ARTIFACT_PATH.rstrip('/')}/"
    f"{MODEL_FILENAME}"
)

MODEL_ARTIFACT_URI = (
    f"gs://{BUCKET}/"
    f"{CANDIDATE_ARTIFACT_PATH.rstrip('/')}/"
)

MODEL_DISPLAY_NAME = (
    f"fwa-"
    f"{MODEL_TYPE.replace('_', '-')}"
    f"-candidate"
)


# ------------------------------------------------------------------
# Serialize configuration passed to KFP
# ------------------------------------------------------------------

FEATURE_SQL_JSON = json.dumps(
    FEATURE_SQL
)

DERIVED_FEATURE_SQL_JSON = (
    json.dumps(
        DERIVED_FEATURE_SQL
    )
)

FEATURE_COLUMNS_JSON = (
    json.dumps(
        FEATURE_COLUMNS
    )
)

MODEL_PARAMETERS_JSON = (
    json.dumps(
        MODEL_PARAMETERS
    )
)


# ------------------------------------------------------------------
# Pipeline configuration
# ------------------------------------------------------------------

PIPELINE_TEMPLATE = (
    "model_training_pipeline.yaml"
)

PIPELINE_ARTIFACT_BUCKET = (
    "fwa-mlops-accelerator-demo-"
    "pipeline-artifacts"
)

PIPELINE_ROOT = (
    f"gs://{PIPELINE_ARTIFACT_BUCKET}/"
    "pipeline_root/model_training"
)


# ------------------------------------------------------------------
# Vertex runtime parameters
# ------------------------------------------------------------------

runtime_parameter_values = {
    "project_id": Value(
        string_value=PROJECT_ID
    ),

    "region": Value(
        string_value=REGION
    ),

    "model_type": Value(
        string_value=MODEL_TYPE
    ),

    "model_name": Value(
        string_value=MODEL_NAME
    ),

    "model_parameters_json": Value(
        string_value=(
            MODEL_PARAMETERS_JSON
        )
    ),

    "current_model_signature": Value(
        string_value=(
            CURRENT_MODEL_SIGNATURE
        )
    ),

    "claims_table": Value(
        string_value=CLAIMS_TABLE
    ),

    "current_data_month": Value(
        string_value=(
            CURRENT_DATA_MONTH
        )
    ),

    "current_feature_signature": Value(
        string_value=(
            CURRENT_FEATURE_SIGNATURE
        )
    ),

    "training_months": Value(
        number_value=TRAINING_MONTHS
    ),

    "testing_months": Value(
        number_value=TESTING_MONTHS
    ),

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

    "model_output_uri": Value(
        string_value=MODEL_OUTPUT_URI
    ),

    "model_artifact_uri": Value(
        string_value=MODEL_ARTIFACT_URI
    ),

    "model_display_name": Value(
        string_value=MODEL_DISPLAY_NAME
    ),

    "serving_container_image_uri": Value(
        string_value=(
            SERVING_CONTAINER_IMAGE_URI
        )
    ),

    "experiment_runs_table": Value(
        string_value=(
            EXPERIMENT_RUNS_TABLE
        )
    ),

    "experiment_metrics_table": Value(
        string_value=(
            EXPERIMENT_METRICS_TABLE
        )
    ),

    "experiment_id": Value(
        string_value=EXPERIMENT_ID
    ),

    "run_id": Value(
        string_value=RUN_ID
    ),

    "experiment_name": Value(
        string_value=EXPERIMENT_NAME
    ),

    "experiment_owner": Value(
        string_value=EXPERIMENT_OWNER
    ),

    "feature_version": Value(
        string_value=FEATURE_VERSION
    ),

    "dataset_version": Value(
        string_value=DATASET_VERSION
    ),

    "git_commit": Value(
        string_value=GIT_COMMIT
    ),

    "training_start": Value(
        string_value=(
            EXPERIMENT_TRAINING_START
        )
    ),

    "training_end": Value(
        string_value=(
            EXPERIMENT_TRAINING_END
        )
    ),

    "data_quality_table": Value(
        string_value=(
            DATA_QUALITY_TABLE
        )
    ),

    "feature_drift_table": Value(
        string_value=(
            FEATURE_DRIFT_TABLE
        )
    ),

    "prediction_drift_table": Value(
        string_value=(
            PREDICTION_DRIFT_TABLE
        )
    ),

    "performance_degradation_table": Value(
        string_value=(
            PERFORMANCE_DEGRADATION_TABLE
        )
    ),

    "training_state_table": Value(
        string_value=(
            TRAINING_STATE_TABLE
        )
    ),
}


# ------------------------------------------------------------------
# Load the compiled KFP spec
# ------------------------------------------------------------------

with open(
    PIPELINE_TEMPLATE,
    "r",
) as file:
    compiled_pipeline = (
        yaml.safe_load(file)
    )

pipeline_spec_dict = (
    compiled_pipeline.get(
        "pipelineSpec",
        compiled_pipeline,
    )
)

pipeline_spec = Struct()

json_format.ParseDict(
    pipeline_spec_dict,
    pipeline_spec,
)


# ------------------------------------------------------------------
# Build the Vertex PipelineJob
# ------------------------------------------------------------------

runtime_config = (
    aiplatform_v1.types
    .PipelineJob
    .RuntimeConfig(
        gcs_output_directory=(
            PIPELINE_ROOT
        ),
        parameter_values=(
            runtime_parameter_values
        ),
    )
)

data_month_label = (
    CURRENT_DATA_MONTH.replace(
        "-",
        "",
    )
)

model_label = (
    MODEL_TYPE.replace(
        "_",
        "-",
    )
)

job_id = (
    f"{model_label}-"
    f"candidate-"
    f"{data_month_label}-"
    f"{RUN_TIMESTAMP}"
)

pipeline_job = (
    aiplatform_v1.types
    .PipelineJob(
        display_name=(
            f"{model_label}-"
            f"candidate-"
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

client = (
    aiplatform_v1
    .PipelineServiceClient(
        client_options={
            "api_endpoint": (
                f"{REGION}-"
                "aiplatform.googleapis.com"
            )
        }
    )
)

parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)


# ------------------------------------------------------------------
# Print configuration before submission
# ------------------------------------------------------------------

print("=" * 70)
print("FWA Continuous Training Pipeline")
print("=" * 70)

print(f"Model type: {MODEL_TYPE}")
print(f"Model name: {MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Experiment ID: {EXPERIMENT_ID}")
print(f"Run ID: {RUN_ID}")
print(
    f"Current data month: "
    f"{CURRENT_DATA_MONTH}"
)
print(
    f"Feature count: "
    f"{len(FEATURE_COLUMNS)}"
)
print(
    f"Feature signature: "
    f"{CURRENT_FEATURE_SIGNATURE[:12]}"
)
print(
    f"Model signature: "
    f"{CURRENT_MODEL_SIGNATURE[:12]}"
)
print(
    f"Artifact format: "
    f"{MODEL_ARTIFACT_FORMAT}"
)
print(
    f"Git commit: "
    f"{GIT_COMMIT[:12]}"
)

print("\nModel parameters")
print("-" * 70)

for name, value in (
    MODEL_PARAMETERS.items()
):
    print(
        f"{name}: {value}"
    )

print("\nExperiment")
print("-" * 70)

print(
    f"Training window: "
    f"{EXPERIMENT_TRAINING_START} "
    f"-> "
    f"{EXPERIMENT_TRAINING_END}"
)

print(
    f"Feature version: "
    f"{FEATURE_VERSION[:12]}"
)

print(
    f"Dataset version: "
    f"{DATASET_VERSION}"
)

print("\nMonitoring")
print("-" * 70)

print(
    f"Data quality: "
    f"{DATA_QUALITY_TABLE}"
)

print(
    f"Feature drift: "
    f"{FEATURE_DRIFT_TABLE}"
)

print(
    f"Prediction drift: "
    f"{PREDICTION_DRIFT_TABLE}"
)

print(
    f"Performance degradation: "
    f"{PERFORMANCE_DEGRADATION_TABLE}"
)

print(
    f"Training state: "
    f"{TRAINING_STATE_TABLE}"
)

print("\nArtifacts")
print("-" * 70)

print(
    f"Candidate model: "
    f"{MODEL_OUTPUT_URI}"
)

print(
    f"Candidate directory: "
    f"{MODEL_ARTIFACT_URI}"
)

print(
    f"Production path: "
    f"gs://{BUCKET}/"
    f"{PRODUCTION_ARTIFACT_PATH}"
)

print(
    f"Registry serving container: "
    f"{SERVING_CONTAINER_IMAGE_URI}"
)


# ------------------------------------------------------------------
# Submit
# ------------------------------------------------------------------

request = (
    aiplatform_v1.types
    .CreatePipelineJobRequest(
        parent=parent,
        pipeline_job=pipeline_job,
        pipeline_job_id=job_id,
    )
)

print(
    "\nSubmitting candidate pipeline..."
)

response = (
    client.create_pipeline_job(
        request=request
    )
)

print(
    f"Pipeline submitted: "
    f"{response.name}"
)


# ------------------------------------------------------------------
# Wait for completion
# ------------------------------------------------------------------

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

last_state = None

while True:
    pipeline_status = (
        client.get_pipeline_job(
            name=response.name
        )
    )

    state = pipeline_status.state

    if state != last_state:
        current_time = (
            datetime.now()
            .strftime("%H:%M:%S")
        )

        print(
            f"[{current_time}] "
            f"{state.name}"
        )

        last_state = state

    if state in terminal_states:
        break

    time.sleep(30)


# ------------------------------------------------------------------
# Final result
# ------------------------------------------------------------------

print("\n" + "=" * 70)

if (
    pipeline_status.state
    ==
    aiplatform_v1.types
    .PipelineState
    .PIPELINE_STATE_SUCCEEDED
):
    print("CANDIDATE PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Model type: "
        f"{MODEL_TYPE}"
    )

    print(
        f"Artifact format: "
        f"{MODEL_ARTIFACT_FORMAT}"
    )

    print(
        f"Candidate artifact: "
        f"{MODEL_OUTPUT_URI}"
    )

    print(
        f"Experiment ID: "
        f"{EXPERIMENT_ID}"
    )

    print(
        f"Run ID: "
        f"{RUN_ID}"
    )

    print(
        f"Data month: "
        f"{CURRENT_DATA_MONTH}"
    )

    print(
        f"Training window: "
        f"{EXPERIMENT_TRAINING_START} "
        f"-> "
        f"{EXPERIMENT_TRAINING_END}"
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
        f"Model signature: "
        f"{CURRENT_MODEL_SIGNATURE[:12]}"
    )

    print(
        "\nProduction remains unchanged "
        "until manual approval."
    )

else:
    print("CANDIDATE PIPELINE FAILED")
    print("=" * 70)

    print(
        f"Experiment ID: "
        f"{EXPERIMENT_ID}"
    )

    print(
        f"Run ID: "
        f"{RUN_ID}"
    )

    print(
        f"State: "
        f"{pipeline_status.state.name}"
    )

    if pipeline_status.error:
        print(
            f"Error: "
            f"{pipeline_status.error}"
        )

    raise RuntimeError(
        "Vertex AI candidate pipeline failed."
    )
