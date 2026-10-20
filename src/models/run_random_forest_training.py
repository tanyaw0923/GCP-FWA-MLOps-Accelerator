#!/usr/bin/env python

import argparse
import json

from src.models.train_random_forest import (
    train_random_forest,
)


# ============================================================
# RANDOM FOREST TRAINING ENTRY POINT
#
# Vertex Pipeline
#       ↓
# train_random_forest_component.py
#       ↓
# Container arguments
#       ↓
# This script
#       ↓
# train_random_forest()
#
# This script is the CLI / container entry-point layer.
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a Random Forest fraud "
            "candidate model."
        )
    )

    # ========================================================
    # 1. CORE TRAINING INPUTS
    # ========================================================

    parser.add_argument(
        "--project-id",
        required=True,
        help="Google Cloud project ID.",
    )

    parser.add_argument(
        "--feature-table",
        required=True,
        help=(
            "Fully qualified BigQuery "
            "provider feature table."
        ),
    )

    parser.add_argument(
        "--feature-columns",
        required=True,
        help=(
            "JSON encoded list of "
            "model feature columns."
        ),
    )

    parser.add_argument(
        "--model-parameters",
        required=True,
        help=(
            "JSON encoded Random Forest "
            "hyperparameters."
        ),
    )

    parser.add_argument(
        "--model-output-uri",
        required=True,
        help=(
            "GCS URI for the candidate "
            "model artifact."
        ),
    )

    # ========================================================
    # 2. EXPERIMENT TRACKING
    # ========================================================

    parser.add_argument(
        "--experiment-runs-table",
        required=True,
        help=(
            "BigQuery table used to record "
            "one row per training run."
        ),
    )

    parser.add_argument(
        "--experiment-metrics-table",
        required=True,
        help=(
            "BigQuery table used to record "
            "evaluation metrics."
        ),
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
        help=(
            "Stable identifier for the "
            "experiment."
        ),
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Unique identifier for this "
            "training execution."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        required=True,
        help=(
            "Human-readable experiment name."
        ),
    )

    parser.add_argument(
        "--experiment-owner",
        required=True,
        help=(
            "Owner or team responsible for "
            "the experiment."
        ),
    )

    parser.add_argument(
        "--model-type",
        required=True,
        help=(
            "Model type used for this run."
        ),
    )

    parser.add_argument(
        "--feature-version",
        required=True,
        help=(
            "Feature signature used as the "
            "feature version."
        ),
    )

    parser.add_argument(
        "--dataset-version",
        required=True,
        help=(
            "Dataset version used for "
            "this training run."
        ),
    )

    parser.add_argument(
        "--git-commit",
        required=True,
        help=(
            "Git commit associated with "
            "this training run."
        ),
    )

    # ========================================================
    # 3. TRAINING WINDOW
    # ========================================================

    parser.add_argument(
        "--training-start",
        required=True,
        help=(
            "Start date of the training "
            "data window."
        ),
    )

    parser.add_argument(
        "--training-end",
        required=True,
        help=(
            "End date of the training "
            "data window."
        ),
    )


    parser.add_argument(
        "--result-output-path",
        required=True,
        help=(
            "KFP-generated output path used "
            "to return the serialized training result."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # 4. PARSE FEATURE CONFIGURATION
    # ========================================================

    try:
        feature_columns = json.loads(
            args.feature_columns
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "--feature-columns must "
            "contain valid JSON."
        ) from exc

    if not isinstance(
        feature_columns,
        list,
    ):
        raise ValueError(
            "--feature-columns must "
            "decode to a list."
        )

    if not feature_columns:
        raise ValueError(
            "Feature list cannot be empty."
        )

    # ========================================================
    # 5. PARSE MODEL PARAMETERS
    # ========================================================

    try:
        model_parameters = json.loads(
            args.model_parameters
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "--model-parameters must "
            "contain valid JSON."
        ) from exc

    if not isinstance(
        model_parameters,
        dict,
    ):
        raise ValueError(
            "--model-parameters must "
            "decode to a dictionary."
        )

    # ========================================================
    # 6. PRINT CONTAINER INPUT
    # ========================================================

    print("=" * 70)
    print("Random Forest Training Container")
    print("=" * 70)

    print(
        f"Project: "
        f"{args.project_id}"
    )

    print(
        f"Feature table: "
        f"{args.feature_table}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Model output URI: "
        f"{args.model_output_uri}"
    )

    print(
        f"Experiment ID: "
        f"{args.experiment_id}"
    )

    print(
        f"Run ID: "
        f"{args.run_id}"
    )

    print(
        f"Experiment name: "
        f"{args.experiment_name}"
    )

    print(
        f"Experiment owner: "
        f"{args.experiment_owner}"
    )

    print(
        f"Model type: "
        f"{args.model_type}"
    )

    print(
        f"Feature version: "
        f"{args.feature_version[:12]}"
    )

    print(
        f"Dataset version: "
        f"{args.dataset_version}"
    )

    print(
        f"Git commit: "
        f"{args.git_commit[:12]}"
    )

    print(
        f"Training window: "
        f"{args.training_start} "
        f"-> "
        f"{args.training_end}"
    )

    # ========================================================
    # 7. TRAIN CANDIDATE
    # ========================================================

    result = train_random_forest(
        project_id=(
            args.project_id
        ),

        feature_table=(
            args.feature_table
        ),

        feature_columns=(
            feature_columns
        ),

        model_parameters=(
            model_parameters
        ),

        model_output_uri=(
            args.model_output_uri
        ),

        experiment_runs_table=(
            args.experiment_runs_table
        ),

        experiment_metrics_table=(
            args.experiment_metrics_table
        ),

        experiment_id=(
            args.experiment_id
        ),

        run_id=(
            args.run_id
        ),

        experiment_name=(
            args.experiment_name
        ),

        experiment_owner=(
            args.experiment_owner
        ),

        model_type=(
            args.model_type
        ),

        feature_version=(
            args.feature_version
        ),

        dataset_version=(
            args.dataset_version
        ),

        git_commit=(
            args.git_commit
        ),

        training_start=(
            args.training_start
        ),

        training_end=(
            args.training_end
        ),
    )

    # ========================================================
    # 8. PRINT RESULT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOM FOREST TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


    # ========================================================
    # 9. RETURN TRAINING RESULT TO KFP
    #
    # The downstream component expects one JSON object encoded
    # as a string. Write that serialized object directly so it
    # is decoded exactly once by json.loads downstream.
    # ========================================================

    from pathlib import Path

    result_output_path = Path(
        args.result_output_path
    )

    result_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_output_path.open(
        "w"
    ) as output_file:
        output_file.write(
            json.dumps(
                result
            )
        )

    print(
        "\nKFP training result written to:"
    )

    print(
        str(
            result_output_path
        )
    )


if __name__ == "__main__":
    main()
