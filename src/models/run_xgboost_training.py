#!/usr/bin/env python

import argparse
import json

from src.models.train_xgboost import (
    train_xgboost,
)


# ============================================================
# XGBOOST TRAINING ENTRY POINT
#
# Vertex Pipeline
#       ↓
# Container arguments
#       ↓
# This script
#       ↓
# train_xgboost()
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train an XGBoost fraud "
            "candidate model."
        )
    )

    parser.add_argument(
        "--project-id",
        required=True,
        help=(
            "Google Cloud project ID."
        ),
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
            "JSON encoded XGBoost "
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

    args = parser.parse_args()

    # ========================================================
    # PARSE FEATURES
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
    # PARSE MODEL PARAMETERS
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
    # PRINT CONTAINER INPUT
    # ========================================================

    print("=" * 70)
    print("XGBoost Training Container")
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

    # ========================================================
    # TRAIN CANDIDATE
    # ========================================================

    result = train_xgboost(
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
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "XGBOOST TRAINING COMPLETED"
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


if __name__ == "__main__":
    main()
