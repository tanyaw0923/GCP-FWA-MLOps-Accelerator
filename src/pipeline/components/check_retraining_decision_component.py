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
    performance_degradation_table: str,
    training_state_table: str,
    state_key: str = "provider_fraud_model",
) -> bool:
    from datetime import date

    from google.cloud import bigquery

    bq = bigquery.Client(project=project_id)

    def normalize_status(value, default="PASS"):
        if value is None:
            return default

        return str(value).strip().upper()

    # Latest data quality result
    data_quality_rows = list(
        bq.query(
            f"""
            SELECT monitoring_status
            FROM `{data_quality_table}`
            ORDER BY monitoring_timestamp DESC
            LIMIT 1
            """
        ).result()
    )

    data_quality_status = normalize_status(
        data_quality_rows[0].monitoring_status
        if data_quality_rows
        else None
    )

    # Feature drift writes one row per feature, so check whether
    # any feature was flagged in the latest monitoring run.
    feature_drift_rows = list(
        bq.query(
            f"""
            SELECT
              COUNTIF(monitoring_status = 'ALERT') AS alert_count
            FROM `{feature_drift_table}`
            WHERE monitoring_timestamp = (
              SELECT MAX(monitoring_timestamp)
              FROM `{feature_drift_table}`
            )
            """
        ).result()
    )

    feature_alert_count = (
        int(feature_drift_rows[0].alert_count or 0)
        if feature_drift_rows
        else 0
    )

    # Latest prediction drift result
    prediction_drift_rows = list(
        bq.query(
            f"""
            SELECT status
            FROM `{prediction_drift_table}`
            ORDER BY
              COALESCE(
                monitoring_timestamp,
                TIMESTAMP(check_date)
              ) DESC
            LIMIT 1
            """
        ).result()
    )

    prediction_drift_status = normalize_status(
        prediction_drift_rows[0].status
        if prediction_drift_rows
        else None
    )

    # Latest performance monitoring result based on completed
    # investigator feedback.
    performance_rows = list(
        bq.query(
            f"""
            SELECT
              monitoring_status,
              model_name,
              model_version,
              run_id,
              alert_reason,
              evaluation_timestamp
            FROM `{performance_degradation_table}`
            ORDER BY evaluation_timestamp DESC
            LIMIT 1
            """
        ).result()
    )

    if performance_rows:
        performance = performance_rows[0]

        performance_status = normalize_status(
            performance.monitoring_status
        )

        performance_model_name = (
            performance.model_name
        )

        performance_model_version = (
            performance.model_version
        )

        performance_run_id = (
            performance.run_id
        )

        performance_alert_reason = (
            performance.alert_reason
        )

        performance_timestamp = (
            performance.evaluation_timestamp
        )

    else:
        performance_status = "PASS"
        performance_model_name = None
        performance_model_version = None
        performance_run_id = None
        performance_alert_reason = None
        performance_timestamp = None

    # Read the currently approved production state.
    state_query = f"""
    SELECT
      last_training_data_month,
      approved_feature_signature,
      approved_model_signature,
      approved_model_type
    FROM `{training_state_table}`
    WHERE model_name = @state_key
    LIMIT 1
    """

    state_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "state_key",
                "STRING",
                state_key,
            )
        ]
    )

    state_rows = list(
        bq.query(
            state_query,
            job_config=state_config,
        ).result()
    )

    if state_rows:
        state = state_rows[0]

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

    # New data month
    current_data_month_date = date.fromisoformat(
        current_data_month
    )

    new_month_available = (
        last_training_data_month is None
        or current_data_month_date
        > last_training_data_month
    )

    # Feature definition changed
    feature_set_changed = (
        approved_feature_signature is not None
        and current_feature_signature
        != approved_feature_signature
    )

    # Model type, hyperparameters, serving configuration, etc.
    # changed compared with the approved model.
    model_configuration_changed = (
        approved_model_signature is not None
        and current_model_signature
        != approved_model_signature
    )

    data_quality_alert = (
        data_quality_status == "ALERT"
    )

    feature_drift_alert = (
        feature_alert_count > 0
    )

    prediction_drift_alert = (
        prediction_drift_status == "ALERT"
    )

    performance_degradation_alert = (
        performance_status == "ALERT"
    )

    reasons = []

    if data_quality_alert:
        reasons.append(
            "data_quality_alert"
        )

    if feature_drift_alert:
        reasons.append(
            "feature_drift_alert"
        )

    if prediction_drift_alert:
        reasons.append(
            "prediction_drift_alert"
        )

    if performance_degradation_alert:
        reasons.append(
            "performance_degradation_alert"
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

    should_retrain = bool(reasons)

    print("=" * 60)
    print("FWA Retraining Decision")
    print("=" * 60)

    print(f"State key: {state_key}")

    print("\nMonitoring")
    print("-" * 60)
    print(
        f"Data quality: "
        f"{data_quality_status}"
    )
    print(
        f"Feature drift alerts: "
        f"{feature_alert_count}"
    )
    print(
        f"Prediction drift: "
        f"{prediction_drift_status}"
    )
    print(
        f"Performance degradation: "
        f"{performance_status}"
    )

    if performance_rows:
        print(
            f"Performance model: "
            f"{performance_model_name}"
        )
        print(
            f"Performance version: "
            f"{performance_model_version}"
        )
        print(
            f"Performance run: "
            f"{performance_run_id}"
        )
        print(
            f"Performance evaluated at: "
            f"{performance_timestamp}"
        )
        print(
            f"Performance alert reason: "
            f"{performance_alert_reason}"
        )

    print("\nCurrent vs approved state")
    print("-" * 60)
    print(
        f"Current data month: "
        f"{current_data_month_date}"
    )
    print(
        f"Last training data month: "
        f"{last_training_data_month}"
    )
    print(
        f"New month available: "
        f"{new_month_available}"
    )

    print(
        f"Feature set changed: "
        f"{feature_set_changed}"
    )

    print(
        f"Approved model type: "
        f"{approved_model_type}"
    )
    print(
        f"Current model type: "
        f"{current_model_type}"
    )
    print(
        f"Model configuration changed: "
        f"{model_configuration_changed}"
    )

    print("\nDecision")
    print("-" * 60)
    print(
        f"Should retrain: "
        f"{should_retrain}"
    )

    if reasons:
        print("Reasons:")

        for reason in reasons:
            print(
                f"  - {reason}"
            )

    else:
        print(
            "No retraining triggers detected."
        )

    return should_retrain
