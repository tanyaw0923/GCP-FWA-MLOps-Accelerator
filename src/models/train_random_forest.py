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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

from src.config.load_config import load_config


def train_random_forest(
    project_id,
    feature_table,
    feedback_table,
    feature_columns,
    model_output_uri,
):
    print("=" * 60)
    print("Random Forest Training")
    print("=" * 60)

    print(f"Project: {project_id}")
    print(f"Feature table: {feature_table}")
    print(f"Feedback table: {feedback_table}")
    print(f"Features: {feature_columns}")

    # ---------------------------------------------------------
    # Load investigator feedback + provider features
    # ---------------------------------------------------------
    bq = bigquery.Client(project=project_id)

    feature_select = ",\n      ".join(
        [f"p.{feature}" for feature in feature_columns]
    )

    query = f"""
    WITH latest_feedback AS (
      SELECT *
      FROM `{feedback_table}`
      WHERE confirmed_fraud IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY provider_id
        ORDER BY investigation_date DESC, created_at DESC
      ) = 1
    )

    SELECT
      f.provider_id,
      {feature_select},
      CAST(f.confirmed_fraud AS INT64) AS fraud_label
    FROM latest_feedback f
    JOIN `{feature_table}` p
      ON f.provider_id = p.provider_id
    WHERE p.dataset_split = 'TEST'
    """

    print("\nLoading training data...")

    df = bq.query(query).to_dataframe()

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df["fraud_label"].nunique() < 2:
        raise ValueError(
            "Training data must contain both fraud classes."
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if df[feature_columns].isnull().any().any():
        raise ValueError(
            "Null values detected in training features."
        )

    print(f"Training rows: {len(df)}")
    print("\nLabel distribution:")
    print(df["fraud_label"].value_counts().sort_index())

    X = df[feature_columns]
    y = df["fraud_label"]

    # ---------------------------------------------------------
    # Random Forest
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
    # Cross-validation
    # ---------------------------------------------------------
    min_class_count = int(
        y.value_counts().min()
    )

    n_splits = min(
        5,
        min_class_count,
    )

    if n_splits < 2:
        raise ValueError(
            "Not enough minority-class samples "
            "for stratified cross-validation."
        )

    print(
        f"\nRunning {n_splits}-fold "
        "stratified cross-validation..."
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    cv_probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    cv_predictions = (
        cv_probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    metrics = {
        "pr_auc": float(
            average_precision_score(
                y,
                cv_probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                cv_probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y,
                cv_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                cv_predictions,
                zero_division=0,
            )
        ),
    }

    print("\nCross-validation metrics:")

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    # ---------------------------------------------------------
    # Final model
    # ---------------------------------------------------------
    print("\nTraining final model...")

    model.fit(X, y)

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------
    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\n" + "=" * 50)
    print("Feature Importance")
    print("=" * 50)

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save locally
    # ---------------------------------------------------------
    local_model_path = "/tmp/model.joblib"

    joblib.dump(
        model,
        local_model_path,
    )

    # ---------------------------------------------------------
    # Upload to GCS
    # ---------------------------------------------------------
    if not model_output_uri.startswith("gs://"):
        raise ValueError(
            "model_output_uri must start with gs://"
        )

    uri_without_prefix = model_output_uri.replace(
        "gs://",
        "",
        1,
    )

    bucket_name, blob_path = uri_without_prefix.split(
        "/",
        1,
    )

    storage_client = storage.Client(
        project=project_id
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
        f"\nModel uploaded to: "
        f"{model_output_uri}"
    )

    # ---------------------------------------------------------
    # Return outputs
    # ---------------------------------------------------------
    result = {
        "metrics": metrics,
        "model_output_uri": model_output_uri,
        "training_rows": int(len(df)),
        "features": feature_columns,
    }

    return result


# -------------------------------------------------------------
# Standalone execution
# -------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()

    model_output_uri = (
        f"gs://{config['bucket']}/"
        "models/random_forest/model.joblib"
    )

    result = train_random_forest(
        project_id=config["project_id"],
        feature_table=config["data"]["feature_table"],
        feedback_table=config["data"]["feedback_table"],
        feature_columns=config["features"],
        model_output_uri=model_output_uri,
    )

    print("\nTraining completed successfully.")
    print(result)

