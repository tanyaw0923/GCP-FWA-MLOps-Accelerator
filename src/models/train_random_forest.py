import joblib
import pandas as pd

from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.load_config import load_config


def train_random_forest(
    project_id,
    feature_table,
    feature_columns,
    model_output_uri,
):
    print("=" * 60)
    print("Random Forest Candidate Training")
    print("=" * 60)

    print(f"Project: {project_id}")
    print(f"Feature table: {feature_table}")
    print(f"Features: {feature_columns}")

    # ---------------------------------------------------------
    # Load provider-level TRAIN and TEST data.
    #
    # The feature table is created by the rolling-window feature
    # component, so dataset_split is already assigned dynamically.
    #
    # TRAIN:
    # used to fit the model
    #
    # TEST:
    # held out for candidate evaluation
    # ---------------------------------------------------------
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

    print("\nLoading rolling TRAIN / TEST data...")

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

    # ---------------------------------------------------------
    # Validate configured feature columns.
    # ---------------------------------------------------------
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: "
            f"{missing_features}"
        )

    # ---------------------------------------------------------
    # Split TRAIN and TEST.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Validate fraud labels.
    #
    # Random Forest training and evaluation both need fraud and
    # normal providers to be represented.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Validate feature quality.
    # ---------------------------------------------------------
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
        f"TEST providers:  "
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

    # ---------------------------------------------------------
    # Prepare model inputs.
    # ---------------------------------------------------------
    X_train = (
        train_df[
            feature_columns
        ]
    )

    y_train = (
        train_df[
            "fraud_label"
        ]
    )

    X_test = (
        test_df[
            feature_columns
        ]
    )

    y_test = (
        test_df[
            "fraud_label"
        ]
    )

    # ---------------------------------------------------------
    # Random Forest candidate.
    #
    # Model parameters currently match pipeline_config.yaml.
    # We can make these fully config-driven in a later cleanup.
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # ---------------------------------------------------------
    # Fit only on TRAIN.
    # ---------------------------------------------------------
    print(
        "\nTraining Random Forest "
        "candidate on TRAIN split..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Evaluate only on TEST.
    #
    # TEST is never used to fit the candidate model.
    # ---------------------------------------------------------
    print(
        "\nEvaluating candidate "
        "on TEST split..."
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

    # ---------------------------------------------------------
    # Candidate metrics.
    #
    # PR-AUC is especially useful for fraud because the positive
    # class is typically much smaller than the normal class.
    # ---------------------------------------------------------
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

    for (
        metric_name,
        metric_value,
    ) in metrics.items():

        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    # ---------------------------------------------------------
    # Feature importance.
    #
    # This will be especially useful in Use Case 2 when
    # avg_member_provider_distance is introduced.
    # ---------------------------------------------------------
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
        + "=" * 50
    )

    print(
        "Feature Importance"
    )

    print(
        "=" * 50
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save candidate locally.
    # ---------------------------------------------------------
    local_model_path = (
        "/tmp/model.joblib"
    )

    joblib.dump(
        model,
        local_model_path,
    )

    # ---------------------------------------------------------
    # Upload candidate artifact to GCS.
    #
    # model_output_uri should point to the candidate path rather
    # than the currently approved production model.
    # ---------------------------------------------------------
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

    (
        bucket_name,
        blob_path,
    ) = uri_without_prefix.split(
        "/",
        1,
    )

    storage_client = storage.Client(
        project=project_id
    )

    bucket = (
        storage_client.bucket(
            bucket_name
        )
    )

    blob = bucket.blob(
        blob_path
    )

    blob.upload_from_filename(
        local_model_path
    )

    print(
        f"\nCandidate model uploaded to: "
        f"{model_output_uri}"
    )

    # ---------------------------------------------------------
    # Return candidate metadata.
    #
    # These metrics can later be persisted and displayed before
    # the manual production promotion decision.
    # ---------------------------------------------------------
    result = {
        "metrics": metrics,
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
        "feature_importance": (
            feature_importance
            .to_dict(
                orient="records"
            )
        ),
    }

    return result


# =============================================================
# Standalone execution
# =============================================================

if __name__ == "__main__":

    config = load_config()

    model_output_uri = (
        f"gs://{config['bucket']}/"
        f"{config['promotion']['candidate_artifact_path']}"
        "model.joblib"
    )

    result = train_random_forest(
        project_id=(
            config["project_id"]
        ),
        feature_table=(
            config["data"][
                "feature_table"
            ]
        ),
        feature_columns=(
            config["features"]
        ),
        model_output_uri=(
            model_output_uri
        ),
    )

    print(
        "\nTraining completed successfully."
    )

    print(
        result
    )
