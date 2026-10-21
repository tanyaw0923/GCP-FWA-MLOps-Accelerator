from datetime import datetime, timezone

import pandas as pd
import xgboost as xgb

from google.cloud import bigquery, storage
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_xgboost(
    project_id,
    feature_table,
    feature_columns,
    model_parameters,
    model_output_uri,

    # Experiment tracking
    experiment_runs_table,
    experiment_metrics_table,
    experiment_id,
    run_id,
    experiment_name,
    experiment_owner,
    model_type,
    feature_version,
    dataset_version,
    git_commit,

    # Rolling training window
    training_start,
    training_end,
):
    print("=" * 70)
    print("XGBoost Candidate Training")
    print("=" * 70)

    print(f"Project: {project_id}")
    print(f"Feature table: {feature_table}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Features: {feature_columns}")

    print(f"Experiment ID: {experiment_id}")
    print(f"Run ID: {run_id}")
    print(f"Experiment name: {experiment_name}")
    print(f"Model type: {model_type}")

    print(
        f"Training window: "
        f"{training_start} -> {training_end}"
    )

    print("\nModel parameters:")

    for parameter, value in model_parameters.items():
        print(
            f"{parameter}: "
            f"{value}"
        )

    # ========================================================
    # 1. VALIDATE MODEL TYPE
    # ========================================================

    if model_type != "xgboost":
        raise ValueError(
            "train_xgboost received unexpected "
            f"model_type={model_type}"
        )

    # ========================================================
    # 2. VALIDATE ARTIFACT URI
    #
    # Vertex's XGBoost prebuilt serving container expects
    # the native Booster artifact to be named exactly:
    #
    #   model.bst
    # ========================================================

    if not model_output_uri.startswith(
        "gs://"
    ):
        raise ValueError(
            "model_output_uri must start with gs://"
        )

    if not model_output_uri.endswith(
        "/model.bst"
    ):
        raise ValueError(
            "Native XGBoost artifact must be named "
            "model.bst for Vertex prebuilt serving."
        )

    # ========================================================
    # 3. BIGQUERY CLIENT
    # ========================================================

    bq = bigquery.Client(
        project=project_id
    )

    # ========================================================
    # 4. LOAD TRAIN / TEST DATA
    # ========================================================

    feature_select = ",\n      ".join(
        feature_columns
    )

    query = f"""
    SELECT
      provider_id,
      dataset_split,
      CAST(fraud_label AS INT64) AS fraud_label,
      {feature_select}

    FROM `{feature_table}`

    WHERE
      dataset_split IN ('TRAIN', 'TEST')
      AND fraud_label IS NOT NULL
    """

    print(
        "\nLoading provider-level TRAIN / TEST data..."
    )

    df = (
        bq.query(
            query
        )
        .to_dataframe()
    )

    if df.empty:
        raise ValueError(
            "Provider feature dataset is empty."
        )

    # ========================================================
    # 5. VALIDATE FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing configured feature columns: "
            f"{missing_features}"
        )

    # ========================================================
    # 6. SPLIT TRAIN / TEST
    # ========================================================

    train_df = (
        df[
            df["dataset_split"]
            == "TRAIN"
        ]
        .copy()
    )

    test_df = (
        df[
            df["dataset_split"]
            == "TEST"
        ]
        .copy()
    )

    if train_df.empty:
        raise ValueError(
            "TRAIN dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            "TEST dataset is empty."
        )

    # ========================================================
    # 7. VALIDATE LABELS
    # ========================================================

    if (
        train_df[
            "fraud_label"
        ]
        .nunique()
        < 2
    ):
        raise ValueError(
            "TRAIN dataset must contain both fraud classes."
        )

    if (
        test_df[
            "fraud_label"
        ]
        .nunique()
        < 2
    ):
        raise ValueError(
            "TEST dataset must contain both fraud classes."
        )

    # ========================================================
    # 8. VALIDATE NULLS
    # ========================================================

    if (
        train_df[
            feature_columns
        ]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "Null values detected in TRAIN features."
        )

    if (
        test_df[
            feature_columns
        ]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "Null values detected in TEST features."
        )

    print(
        f"\nTRAIN providers: "
        f"{len(train_df)}"
    )

    print(
        f"TEST providers: "
        f"{len(test_df)}"
    )

    print(
        "\nTRAIN label distribution:"
    )

    print(
        train_df[
            "fraud_label"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTEST label distribution:"
    )

    print(
        test_df[
            "fraud_label"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # 9. PREPARE MODEL INPUTS
    #
    # Preserve configured feature order exactly.
    # ========================================================

    X_train = (
        train_df[
            feature_columns
        ]
        .copy()
    )

    y_train = (
        train_df[
            "fraud_label"
        ]
        .copy()
    )

    X_test = (
        test_df[
            feature_columns
        ]
        .copy()
    )

    y_test = (
        test_df[
            "fraud_label"
        ]
        .copy()
    )

    # ========================================================
    # 10. VALIDATE XGBOOST PARAMETERS
    # ========================================================

    supported_parameters = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "scale_pos_weight",
        "eval_metric",
        "random_state",
        "n_jobs",
    }

    unsupported_parameters = (
        set(
            model_parameters.keys()
        )
        - supported_parameters
    )

    if unsupported_parameters:
        raise ValueError(
            "Unsupported XGBoost parameter(s): "
            f"{sorted(unsupported_parameters)}"
        )

    # ========================================================
    # 11. TRAIN XGBClassifier
    #
    # We still use XGBClassifier for convenient training and
    # offline probability evaluation.
    #
    # For serving, however, we export the underlying Booster.
    # ========================================================

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        **model_parameters,
    )

    print(
        "\nTraining XGBoost candidate..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # 12. OFFLINE TEST EVALUATION
    # ========================================================

    print(
        "\nEvaluating XGBoost candidate..."
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    probability_summary = (
        pd.Series(
            test_probabilities
        )
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print(
        "\nTEST probability distribution:"
    )

    print(
        probability_summary
    )

    print(
        "\nMaximum predicted fraud probability: "
        f"{test_probabilities.max():.4f}"
    )

    print(
        "Actual fraud providers in TEST: "
        f"{int(y_test.sum())}"
    )

    # ========================================================
    # 13. THRESHOLD ANALYSIS
    # ========================================================

    threshold_results = []

    print(
        "\nThreshold analysis:"
    )

    print(
        "-" * 70
    )

    for threshold in [
        0.20,
        0.30,
        0.40,
        0.50,
    ]:

        threshold_predictions = (
            test_probabilities
            >= threshold
        ).astype(int)

        threshold_precision = float(
            precision_score(
                y_test,
                threshold_predictions,
                zero_division=0,
            )
        )

        threshold_recall = float(
            recall_score(
                y_test,
                threshold_predictions,
                zero_division=0,
            )
        )

        threshold_f1 = float(
            f1_score(
                y_test,
                threshold_predictions,
                zero_division=0,
            )
        )

        predicted_positive_count = int(
            threshold_predictions.sum()
        )

        threshold_results.append(
            {
                "threshold":
                    threshold,

                "predicted_fraud":
                    predicted_positive_count,

                "precision":
                    threshold_precision,

                "recall":
                    threshold_recall,

                "f1_score":
                    threshold_f1,
            }
        )

        print(
            f"threshold={threshold:.2f} | "
            f"predicted_fraud="
            f"{predicted_positive_count} | "
            f"precision="
            f"{threshold_precision:.4f} | "
            f"recall="
            f"{threshold_recall:.4f} | "
            f"f1="
            f"{threshold_f1:.4f}"
        )

    # ========================================================
    # 14. STANDARD METRICS @ 0.50
    # ========================================================

    classification_threshold = 0.50

    test_predictions = (
        test_probabilities
        >= classification_threshold
    ).astype(int)

    metrics = {
        "pr_auc": float(
            average_precision_score(
                y_test,
                test_probabilities,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities,
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
    }

    print(
        "\nCandidate TEST metrics "
        f"@ threshold "
        f"{classification_threshold:.2f}:"
    )

    for (
        metric_name,
        metric_value,
    ) in metrics.items():

        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    # ========================================================
    # 15. FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        pd.DataFrame(
            {
                "feature":
                    feature_columns,

                "importance":
                    model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "XGBoost Feature Importance"
    )

    print(
        "=" * 70
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # ========================================================
    # 16. EXTRACT NATIVE BOOSTER
    # ========================================================

    booster = (
        model.get_booster()
    )

    # ========================================================
    # 17. SAVE NATIVE BOOSTER
    #
    # Vertex XGBoost prebuilt prediction container expects:
    #
    #   model.bst
    #
    # This is intentionally NOT joblib and NOT the sklearn
    # XGBClassifier wrapper.
    # ========================================================

    local_model_path = (
        "/tmp/model.bst"
    )

    booster.save_model(
        local_model_path
    )

    print(
        f"\nNative Booster saved: "
        f"{local_model_path}"
    )

    # ========================================================
    # 18. RELOAD EXACT SAVED BOOSTER
    #
    # This validates the artifact itself before upload.
    # ========================================================

    reloaded_booster = (
        xgb.Booster()
    )

    reloaded_booster.load_model(
        local_model_path
    )

    # ========================================================
    # 19. SERVING-STYLE DMatrix CONSISTENCY TEST
    #
    # The Vertex XGBoost serving container passes a DMatrix
    # into a native Booster. Reproduce that behavior locally.
    # ========================================================

    test_dmatrix = (
        xgb.DMatrix(
            X_test,
            feature_names=(
                feature_columns
            ),
        )
    )

    booster_probabilities = (
        reloaded_booster.predict(
            test_dmatrix
        )
    )

    booster_predictions = (
        booster_probabilities
        >= classification_threshold
    ).astype(int)

    booster_precision = float(
        precision_score(
            y_test,
            booster_predictions,
            zero_division=0,
        )
    )

    booster_recall = float(
        recall_score(
            y_test,
            booster_predictions,
            zero_division=0,
        )
    )

    booster_f1 = float(
        f1_score(
            y_test,
            booster_predictions,
            zero_division=0,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Native Booster Consistency Check"
    )

    print(
        "=" * 70
    )

    print(
        f"Predicted fraud: "
        f"{int(booster_predictions.sum())}"
    )

    print(
        f"Predicted normal: "
        f"{len(booster_predictions) - int(booster_predictions.sum())}"
    )

    print(
        f"Precision: "
        f"{booster_precision:.4f}"
    )

    print(
        f"Recall: "
        f"{booster_recall:.4f}"
    )

    print(
        f"F1: "
        f"{booster_f1:.4f}"
    )

    print(
        f"Min probability: "
        f"{booster_probabilities.min():.6f}"
    )

    print(
        f"Max probability: "
        f"{booster_probabilities.max():.6f}"
    )

    # ========================================================
    # 20. COMPARE CLASSIFIER VS BOOSTER
    #
    # They should produce effectively identical probabilities.
    # ========================================================

    max_probability_difference = float(
        abs(
            test_probabilities
            - booster_probabilities
        )
        .max()
    )

    print(
        "\nMaximum classifier vs Booster "
        "probability difference:"
    )

    print(
        f"{max_probability_difference:.12f}"
    )

    tolerance = 1e-6

    if (
        max_probability_difference
        > tolerance
    ):
        raise RuntimeError(
            "Saved native Booster predictions do not "
            "match XGBClassifier predictions."
        )

    # ========================================================
    # 21. VERIFY METRICS MATCH
    # ========================================================

    metric_tolerance = 1e-9

    if (
        abs(
            booster_precision
            - metrics["precision"]
        )
        > metric_tolerance
        or abs(
            booster_recall
            - metrics["recall"]
        )
        > metric_tolerance
        or abs(
            booster_f1
            - metrics["f1_score"]
        )
        > metric_tolerance
    ):
        raise RuntimeError(
            "Native Booster artifact does not reproduce "
            "offline TEST metrics."
        )

    print(
        "\nNative Booster artifact verified successfully."
    )

    # ========================================================
    # 22. UPLOAD model.bst TO GCS
    # ========================================================

    uri_without_prefix = (
        model_output_uri.replace(
            "gs://",
            "",
            1,
        )
    )

    bucket_name, blob_path = (
        uri_without_prefix.split(
            "/",
            1,
        )
    )

    storage_client = (
        storage.Client(
            project=project_id
        )
    )

    bucket = (
        storage_client.bucket(
            bucket_name
        )
    )

    blob = (
        bucket.blob(
            blob_path
        )
    )

    blob.upload_from_filename(
        local_model_path
    )

    print(
        "\nCandidate model uploaded to:"
    )

    print(
        model_output_uri
    )

    # ========================================================
    # 23. RECORD EXPERIMENT RUN
    # ========================================================

    created_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    experiment_run_row = {
        "experiment_id":
            experiment_id,

        "run_id":
            run_id,

        "experiment_name":
            experiment_name,

        "owner":
            experiment_owner,

        "model_type":
            model_type,

        "feature_version":
            feature_version,

        "dataset_version":
            dataset_version,

        "git_commit":
            git_commit,

        "training_start":
            training_start,

        "training_end":
            training_end,

        "status":
            "SUCCEEDED",

        "created_at":
            created_at,
    }

    print(
        "\nRecording experiment run:"
    )

    print(
        experiment_runs_table
    )

    run_errors = (
        bq.insert_rows_json(
            experiment_runs_table,
            [
                experiment_run_row
            ],
        )
    )

    if run_errors:
        raise RuntimeError(
            "Failed to write experiment run: "
            f"{run_errors}"
        )

    print(
        "Experiment run recorded successfully."
    )

    # ========================================================
    # 24. RECORD EXPERIMENT METRICS
    # ========================================================

    metric_rows = [
        {
            "experiment_id":
                experiment_id,

            "run_id":
                run_id,

            "metric_name":
                metric_name,

            "metric_value":
                metric_value,

            "created_at":
                created_at,
        }

        for (
            metric_name,
            metric_value,
        )
        in metrics.items()
    ]

    print(
        "\nRecording experiment metrics:"
    )

    print(
        experiment_metrics_table
    )

    metric_errors = (
        bq.insert_rows_json(
            experiment_metrics_table,
            metric_rows,
        )
    )

    if metric_errors:
        raise RuntimeError(
            "Failed to write experiment metrics: "
            f"{metric_errors}"
        )

    print(
        "Experiment metrics recorded successfully."
    )

    # ========================================================
    # 25. RETURN CANDIDATE METADATA
    # ========================================================

    result = {
        "model_type":
            model_type,

        "experiment_id":
            experiment_id,

        "run_id":
            run_id,

        "experiment_name":
            experiment_name,

        "metrics":
            metrics,

        "model_output_uri":
            model_output_uri,

        "artifact_format":
            "native_xgboost_booster",

        "artifact_filename":
            "model.bst",

        "training_rows":
            int(
                len(train_df)
            ),

        "test_rows":
            int(
                len(test_df)
            ),

        "training_start":
            training_start,

        "training_end":
            training_end,

        "feature_version":
            feature_version,

        "dataset_version":
            dataset_version,

        "git_commit":
            git_commit,

        "classification_threshold":
            classification_threshold,

        "threshold_analysis":
            threshold_results,

        "features":
            feature_columns,

        "model_parameters":
            model_parameters,

        "feature_importance":
            (
                feature_importance
                .to_dict(
                    orient="records"
                )
            ),

        "booster_consistency": {
            "predicted_fraud":
                int(
                    booster_predictions.sum()
                ),

            "precision":
                booster_precision,

            "recall":
                booster_recall,

            "f1_score":
                booster_f1,

            "max_probability_difference":
                max_probability_difference,
        },
    }

    return result
