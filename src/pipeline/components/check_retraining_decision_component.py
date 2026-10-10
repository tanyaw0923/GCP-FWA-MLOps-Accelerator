from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def check_retraining_decision_component(
    project_id: str,
    current_data_month: str,
    current_feature_signature: str,
    current_model_signature: str,
    current_model_type: str,
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    training_state_table: str,
    state_key: str = "provider_fraud_model",
) -> bool:
    from datetime import date

    from google.cloud import bigquery

    bq = bigquery.Client(
        project=project_id
    )

    # ---------------------------------------------------------
    # 1. Data quality trigger
    # ---------------------------------------------------------

    data_quality_query = f"""
    SELECT
      monitoring_status
    FROM `{data_quality_table}`
    ORDER BY monitoring_timestamp DESC
    LIMIT 1
    """

    data_quality_rows = list(
        bq.query(
            data_quality_query
        ).result()
    )

    data_quality_status = (
        data_quality_rows[0].monitoring_status
        if data_quality_rows
        else "PASS"
    )

    # ---------------------------------------------------------
    # 2. Feature drift trigger
    #
    # Evaluate only the latest monitoring run.
    # ---------------------------------------------------------

    feature_drift_query = f"""
    SELECT
      COUNTIF(
        monitoring_status = 'ALERT'
      ) AS alert_count
    FROM `{feature_drift_table}`
    WHERE monitoring_timestamp = (
      SELECT
        MAX(monitoring_timestamp)
      FROM `{feature_drift_table}`
    )
    """

    feature_drift_rows = list(
        bq.query(
            feature_drift_query
        ).result()
    )

    feature_alert_count = (
        feature_drift_rows[0].alert_count
        if feature_drift_rows
        else 0
    )

    # ---------------------------------------------------------
    # 3. Prediction drift trigger
    #
    # prediction_drift uses:
    #   status
    #   check_date
    # ---------------------------------------------------------

    prediction_drift_query = f"""
    SELECT
      status
    FROM `{prediction_drift_table}`
    ORDER BY check_date DESC
    LIMIT 1
    """

    prediction_drift_rows = list(
        bq.query(
            prediction_drift_query
        ).result()
    )

    prediction_drift_status = (
        prediction_drift_rows[0].status
        if prediction_drift_rows
        else "PASS"
    )

    # ---------------------------------------------------------
    # 4. Read approved production state
    #
    # state_key identifies the fraud workflow, not an algorithm.
    #
    # Random Forest and XGBoost therefore compare against the
    # same approved production baseline.
    # ---------------------------------------------------------

    training_state_query = f"""
    SELECT
      last_training_data_month,
      approved_feature_signature,
      approved_model_signature,
      approved_model_type
    FROM `{training_state_table}`
    WHERE model_name = @state_key
    LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "state_key",
                "STRING",
                state_key,
            )
        ]
    )

    training_state_rows = list(
        bq.query(
            training_state_query,
            job_config=job_config,
        ).result()
    )

    if training_state_rows:
        state = training_state_rows[0]

        last_training_data_month = (
            state.last_training_data_month
        )

        approved_feature_signature = (
            state.approved_feature_signature
        )

        approved_model_signature = (
            state.approved_model_signature
        )

        approved_model_type = (
            state.approved_model_type
        )

    else:
        last_training_data_month = None
        approved_feature_signature = None
        approved_model_signature = None
        approved_model_type = None

    # ---------------------------------------------------------
    # 5. Use Case 1
    #
    # New monthly data becomes available.
    # ---------------------------------------------------------

    current_data_month_date = (
        date.fromisoformat(
            current_data_month
        )
    )

    new_month_available = (
        last_training_data_month is None
        or current_data_month_date
        > last_training_data_month
    )

    # ---------------------------------------------------------
    # 6. Use Case 2
    #
    # Feature configuration changed.
    # ---------------------------------------------------------

    feature_set_changed = (
        approved_feature_signature is not None
        and current_feature_signature
        != approved_feature_signature
    )

    # ---------------------------------------------------------
    # 7. Use Case 3
    #
    # Model type or model parameters changed.
    #
    # The model signature contains:
    #   model_type
    #   model_name
    #   hyperparameters
    #
    # Example:
    #
    # random_forest
    #       ↓
    # xgboost
    #
    # produces a different signature.
    # ---------------------------------------------------------

    model_configuration_changed = (
        approved_model_signature is not None
        and current_model_signature
        != approved_model_signature
    )

    # ---------------------------------------------------------
    # 8. Final retraining decision
    # ---------------------------------------------------------

    should_retrain = (
        data_quality_status == "ALERT"
        or feature_alert_count > 0
        or prediction_drift_status == "ALERT"
        or new_month_available
        or feature_set_changed
        or model_configuration_changed
    )

    # ---------------------------------------------------------
    # 9. Collect retraining reasons
    # ---------------------------------------------------------

    reasons = []

    if data_quality_status == "ALERT":
        reasons.append(
            "data_quality_alert"
        )

    if feature_alert_count > 0:
        reasons.append(
            "feature_drift_alert"
        )

    if prediction_drift_status == "ALERT":
        reasons.append(
            "prediction_drift_alert"
        )

    if new_month_available:
        reasons.append(
            "new_month_data"
        )

    if feature_set_changed:
        reasons.append(
            "feature_set_changed"
        )

    if model_configuration_changed:
        reasons.append(
            "model_configuration_changed"
        )

    # ---------------------------------------------------------
    # 10. Decision summary
    # ---------------------------------------------------------

    print("=" * 70)
    print("FWA Retraining Decision")
    print("=" * 70)

    print(
        f"State key: "
        f"{state_key}"
    )

    print("\nMONITORING")
    print("-" * 70)

    print(
        f"Data Quality Status: "
        f"{data_quality_status}"
    )

    print(
        f"Feature Drift Alert Count: "
        f"{feature_alert_count}"
    )

    print(
        f"Prediction Drift Status: "
        f"{prediction_drift_status}"
    )

    print("\nUSE CASE 1 - DATA")
    print("-" * 70)

    print(
        f"Current Data Month: "
        f"{current_data_month_date}"
    )

    print(
        f"Last Training Data Month: "
        f"{last_training_data_month}"
    )

    print(
        f"New Month Available: "
        f"{new_month_available}"
    )

    print("\nUSE CASE 2 - FEATURES")
    print("-" * 70)

    print(
        "Approved Feature Signature: "
        f"{approved_feature_signature}"
    )

    print(
        "Current Feature Signature: "
        f"{current_feature_signature}"
    )

    print(
        f"Feature Set Changed: "
        f"{feature_set_changed}"
    )

    print("\nUSE CASE 3 - MODEL")
    print("-" * 70)

    print(
        f"Approved Model Type: "
        f"{approved_model_type}"
    )

    print(
        f"Current Model Type: "
        f"{current_model_type}"
    )

    print(
        "Approved Model Signature: "
        f"{approved_model_signature}"
    )

    print(
        "Current Model Signature: "
        f"{current_model_signature}"
    )

    print(
        f"Model Configuration Changed: "
        f"{model_configuration_changed}"
    )

    print("\n" + "=" * 70)

    print(
        f"Should Retrain: "
        f"{should_retrain}"
    )

    if reasons:
        print(
            "\nRetraining trigger(s):"
        )

        for reason in reasons:
            print(
                f"  - {reason}"
            )

    else:
        print(
            "\nNo retraining triggers detected."
        )

    return should_retrain

