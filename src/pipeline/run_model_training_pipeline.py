import hashlib
import json
import subprocess
import time
import yaml

from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta

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
MODEL_CONFIG = config["model"]
MONITORING_CONFIG = config["monitoring"]
PROMOTION_CONFIG = config["promotion"]


# ============================================================
# 2. SELECT ACTIVE MODEL
#
# Use Case 3:
#
# Change only:
#
#   model_type: random_forest
#
# to:
#
#   model_type: xgboost
#
# The rest of the workflow stays reusable.
# ============================================================

MODEL_TYPE = MODEL_CONFIG[
    "model_type"
]


if MODEL_TYPE not in MODEL_CONFIG:

    raise ValueError(
        f"Unsupported model_type: "
        f"{MODEL_TYPE}"
    )


ACTIVE_MODEL_CONFIG = (
    MODEL_CONFIG[
        MODEL_TYPE
    ]
)


MODEL_NAME = (
    ACTIVE_MODEL_CONFIG[
        "model_name"
    ]
)


EXPERIMENT_NAME = (
    ACTIVE_MODEL_CONFIG[
        "experiment_name"
    ]
)


MODEL_PARAMETERS = (
    ACTIVE_MODEL_CONFIG[
        "parameters"
    ]
)


SERVING_CONTAINER_IMAGE_URI = (
    ACTIVE_MODEL_CONFIG[
        "serving_container_image_uri"
    ]
    .strip()
)


# ============================================================
# 3. FEATURES
# ============================================================

FEATURE_COLUMNS = (
    config[
        "features"
    ]
)


FEATURE_SQL = (
    config[
        "feature_sql"
    ]
)


DERIVED_FEATURE_SQL = (
    config[
        "derived_feature_sql"
    ]
)


# ============================================================
# 4. FEATURE SIGNATURE
#
# Used for Use Case 2.
#
# Any change to:
#   feature SQL
#   derived feature SQL
#   feature list
#
# generates a new feature signature.
# ============================================================

feature_signature_payload = {
    "feature_sql": (
        FEATURE_SQL
    ),
    "derived_feature_sql": (
        DERIVED_FEATURE_SQL
    ),
    "features": (
        FEATURE_COLUMNS
    ),
}


feature_signature_json = (
    json.dumps(
        feature_signature_payload,
        sort_keys=True,
    )
)


CURRENT_FEATURE_SIGNATURE = (
    hashlib.sha256(
        feature_signature_json.encode(
            "utf-8"
        )
    )
    .hexdigest()
)


# ============================================================
# 5. MODEL SIGNATURE
#
# Used for Use Case 3.
#
# Any change to:
#   model type
#   model name
#   hyperparameters
#
# generates a new model signature.
# ============================================================

model_signature_payload = {
    "model_type": (
        MODEL_TYPE
    ),
    "model_name": (
        MODEL_NAME
    ),
    "parameters": (
        MODEL_PARAMETERS
    ),
}


model_signature_json = (
    json.dumps(
        model_signature_payload,
        sort_keys=True,
    )
)


CURRENT_MODEL_SIGNATURE = (
    hashlib.sha256(
        model_signature_json.encode(
            "utf-8"
        )
    )
    .hexdigest()
)


# ============================================================
# 6. DATA CONFIGURATION
# ============================================================

CURRENT_DATA_MONTH = str(
    DATA_CONFIG[
        "current_data_month"
    ]
)


ROLLING_WINDOW = (
    DATA_CONFIG[
        "rolling_window"
    ]
)


TRAINING_MONTHS = int(
    ROLLING_WINDOW[
        "training_months"
    ]
)


TESTING_MONTHS = int(
    ROLLING_WINDOW[
        "testing_months"
    ]
)


CLAIMS_TABLE = (
    DATA_CONFIG[
        "claims_table"
    ]
)


FEATURE_TABLE = (
    DATA_CONFIG[
        "feature_table"
    ]
)


EXPERIMENT_RUNS_TABLE = (
    DATA_CONFIG[
        "experiment_runs_table"
    ]
)


EXPERIMENT_METRICS_TABLE = (
    DATA_CONFIG[
        "experiment_metrics_table"
    ]
)


# ============================================================
# 7. EXPERIMENT TRAINING WINDOW
#
# These dates are calculated here only for experiment metadata.
#
# The pipeline still uses Calculate Rolling Window for the
# actual feature-engineering workflow.
#
# Keeping these as pipeline parameters prevents extra DAG
# dependencies from Rolling Window directly to model training.
# ============================================================

current_month_date = (
    datetime.strptime(
        CURRENT_DATA_MONTH,
        "%Y-%m-%d",
    )
)


# Example:
#
# current month = 2026-09-01
# testing months = 3
#
# TEST:
#   2026-07-01 -> 2026-09-30
#
# TRAIN:
#   2026-01-01 -> 2026-06-30

test_start_month = (
    current_month_date
    - relativedelta(
        months=(
            TESTING_MONTHS - 1
        )
    )
)


training_end_month = (
    test_start_month
    - relativedelta(
        months=1
    )
)


training_start_month = (
    training_end_month
    - relativedelta(
        months=(
            TRAINING_MONTHS - 1
        )
    )
)


EXPERIMENT_TRAINING_START = (
    training_start_month
    .replace(
        day=1
    )
    .strftime(
        "%Y-%m-%d"
    )
)


EXPERIMENT_TRAINING_END = (
    (
        test_start_month
        - relativedelta(
            days=1
        )
    )
    .strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# 8. EXPERIMENT TRACKING
# ============================================================

EXPERIMENT_OWNER = (
    config.get(
        "experiment_tracking",
        {},
    )
    .get(
        "owner",
        "data_science_team",
    )
)


# Stable experiment ID.
#
# The same experiment_name always produces the same
# experiment_id.
EXPERIMENT_ID = (
    hashlib.sha256(
        EXPERIMENT_NAME.encode(
            "utf-8"
        )
    )
    .hexdigest()[:16]
)


# One timestamp per pipeline execution.
RUN_TIMESTAMP = (
    datetime.now(
        timezone.utc
    )
    .strftime(
        "%Y%m%d%H%M%S"
    )
)


# Unique run ID.
RUN_ID = (
    f"{MODEL_TYPE}-"
    f"{CURRENT_DATA_MONTH.replace('-', '')}-"
    f"{RUN_TIMESTAMP}"
)


# Feature signature becomes the feature version.
FEATURE_VERSION = (
    CURRENT_FEATURE_SIGNATURE
)


# Current data month represents the dataset version.
DATASET_VERSION = (
    CURRENT_DATA_MONTH
)


# Capture Git commit for reproducibility.
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

    GIT_COMMIT = (
        "unknown"
    )


# ============================================================
# 9. MONITORING / TRAINING STATE
# ============================================================

DATA_QUALITY_TABLE = (
    MONITORING_CONFIG[
        "data_quality_table"
    ]
)


FEATURE_DRIFT_TABLE = (
    MONITORING_CONFIG[
        "feature_drift_table"
    ]
)


PREDICTION_DRIFT_TABLE = (
    MONITORING_CONFIG[
        "prediction_drift_table"
    ]
)


TRAINING_STATE_TABLE = (
    MONITORING_CONFIG[
        "training_state_table"
    ]
)


# ============================================================
# 10. MODEL-SPECIFIC PROMOTION CONFIG
# ============================================================

if MODEL_TYPE not in PROMOTION_CONFIG:

    raise ValueError(
        "No promotion configuration found "
        f"for model_type={MODEL_TYPE}"
    )


MODEL_PROMOTION_CONFIG = (
    PROMOTION_CONFIG[
        MODEL_TYPE
    ]
)


CANDIDATE_ARTIFACT_PATH = (
    MODEL_PROMOTION_CONFIG[
        "candidate_artifact_path"
    ]
)


PRODUCTION_ARTIFACT_PATH = (
    MODEL_PROMOTION_CONFIG[
        "production_artifact_path"
    ]
)


# ============================================================
# 11. MODEL ARTIFACT FORMAT
# ============================================================

if MODEL_TYPE == "random_forest":

    MODEL_FILENAME = (
        "model.joblib"
    )


elif MODEL_TYPE == "xgboost":

    MODEL_FILENAME = (
        "model.bst"
    )


else:

    raise ValueError(
        f"Unsupported model_type: "
        f"{MODEL_TYPE}"
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


# ============================================================
# 12. VERTEX MODEL REGISTRY CONFIGURATION
# ============================================================

MODEL_DISPLAY_NAME = (
    f"fwa-"
    f"{MODEL_TYPE.replace('_', '-')}"
    f"-candidate"
)


# ============================================================
# 13. SERIALIZE CONFIGURATION
#
# Complex Python objects are passed into KFP as JSON strings.
# ============================================================

FEATURE_SQL_JSON = (
    json.dumps(
        FEATURE_SQL
    )
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


# ============================================================
# 14. COMPILED PIPELINE
# ============================================================

PIPELINE_TEMPLATE = (
    "model_training_pipeline.yaml"
)


PIPELINE_ARTIFACT_BUCKET = (
    "fwa-mlops-accelerator-demo-"
    "pipeline-artifacts"
)


PIPELINE_ROOT = (
    f"gs://{PIPELINE_ARTIFACT_BUCKET}/"
    "pipeline_root/"
    "model_training"
)


# ============================================================
# 15. RUNTIME PARAMETERS
#
# These values are passed into the compiled Vertex pipeline.
# ============================================================

runtime_parameter_values = {

    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    "project_id": Value(
        string_value=(
            PROJECT_ID
        )
    ),

    "region": Value(
        string_value=(
            REGION
        )
    ),

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    "model_type": Value(
        string_value=(
            MODEL_TYPE
        )
    ),

    "model_name": Value(
        string_value=(
            MODEL_NAME
        )
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

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    "claims_table": Value(
        string_value=(
            CLAIMS_TABLE
        )
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
        number_value=(
            TRAINING_MONTHS
        )
    ),

    "testing_months": Value(
        number_value=(
            TESTING_MONTHS
        )
    ),

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    "feature_table": Value(
        string_value=(
            FEATURE_TABLE
        )
    ),

    "feature_sql_json": Value(
        string_value=(
            FEATURE_SQL_JSON
        )
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
    # Candidate model artifacts
    # --------------------------------------------------------

    "model_output_uri": Value(
        string_value=(
            MODEL_OUTPUT_URI
        )
    ),

    "model_artifact_uri": Value(
        string_value=(
            MODEL_ARTIFACT_URI
        )
    ),

    # --------------------------------------------------------
    # Vertex Model Registry
    # --------------------------------------------------------

    "model_display_name": Value(
        string_value=(
            MODEL_DISPLAY_NAME
        )
    ),

    "serving_container_image_uri": Value(
        string_value=(
            SERVING_CONTAINER_IMAGE_URI
        )
    ),

    # --------------------------------------------------------
    # Experiment tracking
    # --------------------------------------------------------

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
        string_value=(
            EXPERIMENT_ID
        )
    ),

    "run_id": Value(
        string_value=(
            RUN_ID
        )
    ),

    "experiment_name": Value(
        string_value=(
            EXPERIMENT_NAME
        )
    ),

    "experiment_owner": Value(
        string_value=(
            EXPERIMENT_OWNER
        )
    ),

    "feature_version": Value(
        string_value=(
            FEATURE_VERSION
        )
    ),

    "dataset_version": Value(
        string_value=(
            DATASET_VERSION
        )
    ),

    "git_commit": Value(
        string_value=(
            GIT_COMMIT
        )
    ),

    # --------------------------------------------------------
    # Experiment training window
    #
    # These are ordinary pipeline parameters rather than
    # outputs from Calculate Rolling Window. This keeps the
    # Vertex DAG visually clean.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Monitoring
    # --------------------------------------------------------

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

    "training_state_table": Value(
        string_value=(
            TRAINING_STATE_TABLE
        )
    ),
}


# ============================================================
# 16. LOAD COMPILED KFP PIPELINE
# ============================================================

with open(
    PIPELINE_TEMPLATE,
    "r",
) as f:

    compiled_pipeline = (
        yaml.safe_load(
            f
        )
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


pipeline_spec = (
    Struct()
)


json_format.ParseDict(
    pipeline_spec_dict,
    pipeline_spec,
)


# ============================================================
# 17. VERTEX RUNTIME CONFIG
# ============================================================

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


# ============================================================
# 18. PIPELINE JOB METADATA
# ============================================================

data_month_label = (
    CURRENT_DATA_MONTH
    .replace(
        "-",
        "",
    )
)


model_label = (
    MODEL_TYPE
    .replace(
        "_",
        "-",
    )
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


# ============================================================
# 19. VERTEX AI CLIENT
# ============================================================

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


# ============================================================
# 20. UNIQUE PIPELINE JOB ID
#
# Reuse RUN_TIMESTAMP so the Vertex job and experiment run
# clearly belong to the same execution.
# ============================================================

job_id = (
    f"{model_label}-"
    f"candidate-"
    f"{data_month_label}-"
    f"{RUN_TIMESTAMP}"
)


parent = (
    f"projects/{PROJECT_ID}/"
    f"locations/{REGION}"
)


# ============================================================
# 21. PRINT PIPELINE CONFIGURATION
# ============================================================

print("=" * 70)

print(
    "FWA Continuous Training Pipeline"
)

print("=" * 70)


print(
    f"Model type: "
    f"{MODEL_TYPE}"
)


print(
    f"Model name: "
    f"{MODEL_NAME}"
)


print(
    f"Experiment: "
    f"{EXPERIMENT_NAME}"
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
    f"Git commit: "
    f"{GIT_COMMIT[:12]}"
)


# ============================================================
# 22. MODEL PARAMETERS
# ============================================================

print(
    "\nMODEL PARAMETERS"
)

print("-" * 70)


for parameter, value in (
    MODEL_PARAMETERS.items()
):

    print(
        f"{parameter}: "
        f"{value}"
    )


# ============================================================
# 23. EXPERIMENT TRACKING
# ============================================================

print(
    "\nEXPERIMENT TRACKING"
)

print("-" * 70)


print(
    f"Experiment runs table: "
    f"{EXPERIMENT_RUNS_TABLE}"
)


print(
    f"Experiment metrics table: "
    f"{EXPERIMENT_METRICS_TABLE}"
)


print(
    f"Experiment owner: "
    f"{EXPERIMENT_OWNER}"
)


print(
    f"Feature version: "
    f"{FEATURE_VERSION[:12]}"
)


print(
    f"Dataset version: "
    f"{DATASET_VERSION}"
)


print(
    f"Training window: "
    f"{EXPERIMENT_TRAINING_START} "
    f"-> "
    f"{EXPERIMENT_TRAINING_END}"
)


# ============================================================
# 24. MODEL ARTIFACTS
# ============================================================

print(
    "\nMODEL ARTIFACTS"
)

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
    f"Serving container: "
    f"{SERVING_CONTAINER_IMAGE_URI}"
)


# ============================================================
# 25. RETRAINING USE CASES
# ============================================================

print(
    "\nRETRAINING TRIGGERS"
)

print("-" * 70)


print(
    "Use Case 1: "
    "New monthly data"
)


print(
    "Use Case 2: "
    "Feature configuration change"
)


print(
    "Use Case 3: "
    "Model configuration change"
)


# ============================================================
# 26. SUBMIT PIPELINE
# ============================================================

request = (
    aiplatform_v1.types
    .CreatePipelineJobRequest(
        parent=(
            parent
        ),
        pipeline_job=(
            pipeline_job
        ),
        pipeline_job_id=(
            job_id
        ),
    )
)


print(
    "\n" + "=" * 70
)

print(
    "Submitting candidate pipeline..."
)

print("=" * 70)


response = (
    client.create_pipeline_job(
        request=(
            request
        )
    )
)


print(
    f"\nPipeline submitted: "
    f"{response.name}"
)


# ============================================================
# 27. WAIT FOR PIPELINE COMPLETION
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


last_state = (
    None
)


while True:

    pipeline_job_status = (
        client.get_pipeline_job(
            name=(
                response.name
            )
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
            f"{state.name}"
        )

        last_state = (
            state
        )


    if state in terminal_states:

        break


    time.sleep(
        30
    )


# ============================================================
# 28. FINAL RESULT
# ============================================================

print(
    "\n" + "=" * 70
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

    print("=" * 70)


    print(
        f"Model type: "
        f"{MODEL_TYPE}"
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

    print(
        "CANDIDATE PIPELINE FAILED"
    )

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
        f"{pipeline_job_status.state.name}"
    )


    if pipeline_job_status.error:

        print(
            f"Error: "
            f"{pipeline_job_status.error}"
        )


    raise RuntimeError(
        "Vertex AI candidate "
        "pipeline failed."
    )
