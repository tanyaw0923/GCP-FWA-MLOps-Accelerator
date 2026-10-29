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
from src.utils.observability import (
    configure_observability,
    log_event,
)


# STEP 1 — Load shared pipeline settings.
#
# WHY:
# Keep environment-specific settings outside Python so the same orchestrator
# can launch different models, data months, and monitoring configurations.
# load_config() resolves the YAML-backed configuration; subsequent sections
# select the values to pass to the compiled Vertex AI pipeline.
#
# NOTE:
# This file submits and observes the pipeline. The actual feature engineering,
# training, evaluation, and promotion logic runs inside its KFP components.
# Load project configuration
config = load_config()

PROJECT_ID = config["project_id"]
REGION = config["region"]
BUCKET = config["bucket"]

logger = configure_observability(
    project_id=PROJECT_ID,
    service_name="fwa-training-orchestrator",
)

data_config = config["data"]
model_config = config["model"]
monitoring_config = config["monitoring"]
promotion_config = config["promotion"]
promotion_gate_config = promotion_config["gate"]


# ------------------------------------------------------------------
# STEP 2 — Select the model configuration.
#
# WHY:
# Switching model.model_type in configuration chooses the active algorithm
# without rewriting this orchestrator. Each model provides its own name,
# experiment, hyperparameters, training image, and serving image.
#
# This also enables Use Case 3: testing a different model or parameters
# against the same data window and features.
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

TRAINING_IMAGE_URI = (
    active_model["training_image"]
    .strip()
)

SERVING_CONTAINER_IMAGE_URI = (
    active_model["serving_container_image_uri"]
    .strip()
)


# ------------------------------------------------------------------
# STEP 3 — Match each algorithm to its serialized model artifact.
#
# Random Forest is serialized with joblib; XGBoost uses its native Booster
# file. Downstream registration and serving components must load the
# matching format. Reject unsupported model types instead of silently
# producing an artifact that serving cannot read.
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
# STEP 4 — Validate and select the feature definition.
#
# Each model explicitly lists its training features. The shared feature SQL
# and derived-feature SQL describe how provider-level inputs are produced.
# Fail early if the selected model has no feature list.
#
# Use Case 2: changing SQL or selected feature columns changes the feature
# signature, making a distinct experiment candidate possible.
# Feature configuration
# ------------------------------------------------------------------

if "features" not in active_model:
    raise ValueError(
        "Each model must define its own features list in "
        "pipeline_config.yaml under model.<model_type>.features. "
        f"Missing features for model_type={MODEL_TYPE}."
    )

FEATURE_COLUMNS = active_model["features"]

if not FEATURE_COLUMNS:
    raise ValueError(
        f"Feature list is empty for model_type={MODEL_TYPE}."
    )
FEATURE_SQL = config["feature_sql"]
DERIVED_FEATURE_SQL = config[
    "derived_feature_sql"
]


def build_signature(payload):
    """Hash selected settings so equivalent configuration has one signature.

    Sorting JSON keys makes the hash insensitive to dictionary insertion
    order. This is configuration identity, not a hash of the underlying
    training rows or a guarantee that external source data is unchanged.
    """

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


# STEP 5 — Track model-setting changes independently of feature changes.
#
# The pipeline can distinguish a changed algorithm/hyperparameter/serving
# configuration from a changed feature definition. This is useful when
# deciding whether a new candidate is warranted.
#
# The model signature is used by the retraining decision.
# Changes to the algorithm, parameters, training image,
# artifact format, or serving configuration should create
# a new candidate.
CURRENT_MODEL_SIGNATURE = build_signature(
    {
        "model_type": MODEL_TYPE,
        "model_name": MODEL_NAME,
        "parameters": MODEL_PARAMETERS,
        "training_image_uri": (
            TRAINING_IMAGE_URI
        ),
        "artifact_format": (
            MODEL_ARTIFACT_FORMAT
        ),
        "serving_container_image_uri": (
            SERVING_CONTAINER_IMAGE_URI
        ),
    }
)


# ------------------------------------------------------------------
# STEP 6 — Read data windows and BigQuery table destinations.
#
# Use Case 1: moving current_data_month forward lets the pipeline train
# on newly available monthly claims while reusing model and feature logic.
# training_months and testing_months control the rolling temporal split.
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
# STEP 7 — Calculate human-readable experiment date boundaries.
#
# The latest testing_months end with current_data_month. Training ends
# immediately before testing starts, so the reported windows do not overlap.
# These dates describe experiment metadata; feature generation calculates
# its operational rolling window separately within pipeline components.
#
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
# STEP 8 — Stamp the run with lineage identifiers.
#
# experiment_id is stable for an experiment name; run_id additionally embeds
# model type, data month, and a UTC timestamp to distinguish executions.
# Feature/model signatures, dataset month, and Git commit help explain which
# code and settings produced a candidate during audits or comparisons.
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


# Git may be unavailable in some deployment environments; keeping "unknown"
# allows submission while making the missing code provenance explicit.
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
# STEP 9 — Provide downstream observability and promotion settings.
#
# KFP components write data-quality/drift/performance/training-state results
# to these tables. Promotion thresholds are supplied to the gate that
# compares candidate metrics; declaring them here does not itself promote
# or deploy a model.
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

MIN_PR_AUC_IMPROVEMENT = float(
    promotion_gate_config[
        "min_pr_auc_improvement"
    ]
)

MIN_RECALL_CHANGE = float(
    promotion_gate_config[
        "min_recall_change"
    ]
)

MIN_F1_CHANGE = float(
    promotion_gate_config[
        "min_f1_change"
    ]
)

MIN_PRECISION_CHANGE = float(
    promotion_gate_config[
        "min_precision_change"
    ]
)


# ------------------------------------------------------------------
# STEP 10 — Separate candidate artifacts from production artifacts.
#
# Training writes to a configured candidate GCS location. The production
# path is tracked separately so an experiment cannot overwrite production
# merely by running this launcher. Promotion requires the downstream gate
# and any configured manual-approval process.
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
# STEP 11 — Serialize structured inputs for the pipeline boundary.
#
# Feature SQL, feature lists, and estimator settings are nested structures.
# JSON strings preserve them when passing through string-valued KFP/Vertex
# runtime parameters; pipeline components must deserialize them.
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
# STEP 12 — Identify the compiled pipeline and its artifact root.
#
# The YAML template represents the compiled KFP graph. Vertex AI writes
# task outputs and metadata under PIPELINE_ROOT in Cloud Storage.
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
# STEP 13 — Build typed runtime arguments for every pipeline component.
#
# Vertex's low-level API expects protobuf Value objects, not raw Python
# values. Strings carry IDs, table names, SQL/config JSON, image URIs, and
# lineage; number_value carries integer windows and numeric thresholds.
# This explicit mapping makes launches reproducible and inspectable.
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

    "min_pr_auc_improvement": Value(
        number_value=(
            MIN_PR_AUC_IMPROVEMENT
        )
    ),

    "min_recall_change": Value(
        number_value=(
            MIN_RECALL_CHANGE
        )
    ),

    "min_f1_change": Value(
        number_value=(
            MIN_F1_CHANGE
        )
    ),

    "min_precision_change": Value(
        number_value=(
            MIN_PRECISION_CHANGE
        )
    ),
}


# ------------------------------------------------------------------
# STEP 14 — Parse the compiled KFP YAML into a protobuf pipeline spec.
#
# Some compiler outputs wrap the graph in 'pipelineSpec'; others contain
# the spec directly. Parsing to Struct produces the object expected by
# the Vertex PipelineJob API. This does not execute the graph locally.
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
# STEP 15 — Construct the Vertex AI job request.
#
# RuntimeConfig supplies parameters and GCS output location. The job ID
# includes model, data month, and timestamp for traceability; display name
# is shorter for readability. The regional client directs requests to
# the intended Vertex AI location.
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
# STEP 16 — Print a preflight summary for debugging and audit review.
#
# Log active features, version signatures, windows, parameters, candidate
# paths, monitoring sinks, and gate thresholds before making an API call.
# This is informational; it is not a validation of source data quality.
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
    f"Training image: "
    f"{TRAINING_IMAGE_URI}"
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

print("\nModel promotion gate")
print("-" * 70)

print(
    f"Minimum PR-AUC improvement: "
    f"{MIN_PR_AUC_IMPROVEMENT:+.4f}"
)

print(
    f"Minimum recall change: "
    f"{MIN_RECALL_CHANGE:+.4f}"
)

print(
    f"Minimum F1 change: "
    f"{MIN_F1_CHANGE:+.4f}"
)

print(
    f"Minimum precision change: "
    f"{MIN_PRECISION_CHANGE:+.4f}"
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
# STEP 17 — Submit the compiled pipeline to Vertex AI.
#
# create_pipeline_job starts the remote candidate workflow and returns a
# resource name. Structured logging records the run and Vertex resource
# so operators can correlate CLI output with Cloud Logging.
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

log_event(
    logger,
    "training_pipeline_submitted",
    fields={
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "pipeline_job": response.name,
        "model_type": MODEL_TYPE,
    },
)


# ------------------------------------------------------------------
# STEP 18 — Poll the remote job until a terminal state is observed.
#
# Query Vertex every 30 seconds and log state transitions only when they
# change, avoiding noisy duplicate events. Success, failure, cancellation,
# and pause all end this polling loop; only success passes the final check.
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

        log_event(
            logger,
            "training_pipeline_state_changed",
            fields={
                "run_id": RUN_ID,
                "pipeline_job": response.name,
                "state": state.name,
            },
        )

        last_state = state

    if state in terminal_states:
        break

    time.sleep(30)


# ------------------------------------------------------------------
# STEP 19 — Report execution outcome and preserve failure visibility.
#
# Success means the Vertex pipeline completed, not necessarily that its
# candidate passed the promotion gate or reached production. Metrics are
# recorded independently of promotion eligibility. A non-success state
# logs diagnostic information and raises an error for callers/CI.
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
    print("TRAINING PIPELINE COMPLETED")
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
        "\nExperiment results were recorded "
        "regardless of promotion eligibility."
    )

    print(
        "Only models that pass the Model Promotion Gate "
        "are registered as candidates and moved to "
        "PENDING_APPROVAL."
    )

    print(
        "Production remains unchanged until manual approval."
    )

    log_event(
        logger,
        "training_pipeline_completed",
        fields={
            "run_id": RUN_ID,
            "experiment_id": EXPERIMENT_ID,
            "pipeline_job": response.name,
            "state": pipeline_status.state.name,
            "model_type": MODEL_TYPE,
            "artifact_uri": MODEL_OUTPUT_URI,
            "data_month": CURRENT_DATA_MONTH,
        },
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

    log_event(
        logger,
        "training_pipeline_failed",
        severity="ERROR",
        fields={
            "run_id": RUN_ID,
            "experiment_id": EXPERIMENT_ID,
            "pipeline_job": response.name,
            "state": pipeline_status.state.name,
            "error": str(
                pipeline_status.error
            ),
        },
    )

    raise RuntimeError(
        "Vertex AI candidate pipeline failed."
    )
