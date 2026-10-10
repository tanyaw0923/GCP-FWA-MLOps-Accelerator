import pandas as pd
import xgboost as xgb

from google.cloud import bigquery, storage
from sklearn.metrics import (
    average_precision_score,
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
):
    print("=" * 70)
    print("XGBoost Candidate Training")
    print("=" * 70)

    print(f"Project: {project_id}")
    print(f"Feature table: {feature_table}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Features: {feature_columns}")

    print("\nModel parameters:")

    for parameter, value in model_parameters.items():
        print(
            f"{parameter}: "
            f"{value}"
        )

    # ========================================================
    # 1. LOAD PROVIDER-LEVEL TRAIN / TEST DATA
    # ========================================================

    bq = bigquery.Client(
        project=project_id
    )

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
    WHERE dataset_split IN ('TRAIN', 'TEST')
      AND fraud_label IS NOT NULL
    """

    print(
        "\nLoading provider-level "
        "TRAIN / TEST data..."
    )

    df = (
        bq.query(query)
        .to_dataframe()
    )

    if df.empty:
        raise ValueError(
            "Provider feature dataset is empty."
        )

    # ========================================================
    # 2. VALIDATE FEATURES
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
    # 3. SPLIT TRAIN / TEST
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
    # 4. VALIDATE LABELS
    # ========================================================

    if (
        train_df["fraud_label"]
        .nunique()
        < 2
    ):
        raise ValueError(
            "TRAIN dataset must contain "
            "both fraud classes."
        )

    if (
        test_df["fraud_label"]
        .nunique()
        < 2
    ):
        raise ValueError(
            "TEST dataset must contain "
            "both fraud classes."
        )

    # ========================================================
    # 5. VALIDATE NULL VALUES
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
            "Null values detected in "
            "TRAIN features."
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
            "Null values detected in "
            "TEST features."
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
    # 6. PREPARE MODEL INPUTS
    # ========================================================

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "fraud_label"
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "fraud_label"
    ]

    # ========================================================
    # 7. VALIDATE XGBOOST PARAMETERS
    # ========================================================

    supported_parameters = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "eval_metric",
        "random_state",
        "n_jobs",
    }

    unsupported_parameters = (
        set(model_parameters.keys())
        - supported_parameters
    )

    if unsupported_parameters:
        raise ValueError(
            "Unsupported XGBoost "
            "parameter(s): "
            f"{sorted(unsupported_parameters)}"
        )

    # ========================================================
    # 8. BUILD XGBOOST MODEL FROM CONFIG
    # ========================================================

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        **model_parameters,
    )

    # ========================================================
    # 9. TRAIN
    # ========================================================

    print(
        "\nTraining XGBoost candidate..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # 10. EVALUATE ON TEST
    # ========================================================

    print(
        "\nEvaluating XGBoost candidate..."
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= 0.5
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
    }

    print(
        "\nCandidate TEST metrics:"
    )

    for metric_name, metric_value in (
        metrics.items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    # ========================================================
    # 11. FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        pd.DataFrame(
            {
                "feature": (
                    feature_columns
                ),
                "importance": (
                    model.feature_importances_
                ),
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
    # 12. SAVE MODEL LOCALLY
    #
    # Save the Booster artifact because the launcher expects:
    #
    #   model.bst
    # ========================================================

    local_model_path = (
        "/tmp/model.bst"
    )

    model.get_booster().save_model(
        local_model_path
    )

    print(
        f"\nLocal model saved: "
        f"{local_model_path}"
    )

    # ========================================================
    # 13. UPLOAD CANDIDATE TO GCS
    # ========================================================

    if not model_output_uri.startswith(
        "gs://"
    ):
        raise ValueError(
            "model_output_uri must "
            "start with gs://"
        )

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

    bucket = storage_client.bucket(
        bucket_name
    )

    blob = bucket.blob(
        blob_path
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
    # 14. RETURN CANDIDATE METADATA
    # ========================================================

    result = {
        "model_type": (
            "xgboost"
        ),

        "metrics": (
            metrics
        ),

        "model_output_uri": (
            model_output_uri
        ),

        "training_rows": int(
            len(train_df)
        ),

        "test_rows": int(
            len(test_df)
        ),

        "features": (
            feature_columns
        ),

        "model_parameters": (
            model_parameters
        ),

        "feature_importance": (
            feature_importance
            .to_dict(
                orient="records"
            )
        ),
    }

    return result
