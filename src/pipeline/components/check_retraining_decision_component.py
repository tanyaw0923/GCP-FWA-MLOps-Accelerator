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
    data_quality_table: str,
    feature_drift_table: str,
    prediction_drift_table: str,
    training_state_table: str,
) -> bool:
    from datetime import date
    from google.cloud import bigquery

    bq = bigquery.Client(project=project_id)

    # ---------------------------------------------------------
    # 1. Data Quality trigger
    # ---------------------------------------------------------
    data_quality_query = f"""
    SELECT
      monitoring_status
    FROM `{data_quality_table}`
    ORDER BY monitoring_timestamp DESC
    LIMIT 1
    """

    data_quality_result = list(
        bq.query(data_quality_query).result()
    )

    data_quality_status = (
        data_quality_result[0].monitoring_status
        if data_quality_result
        else "PASS"
    )

    # ---------------------------------------------------------
    # 2. Feature Drift trigger
    # ---------------------------------------------------------
    feature_drift_query = f"""
    SELECT
      COUNTIF(monitoring_status = 'ALERT') AS alert_count
    FROM `{feature_drift_table}`
    """

    feature_drift_result = list(
        bq.query(feature_drift_query).result()
    )

    feature_alert_count = (
        feature_drift_result[0].alert_count
        if feature_drift_result
        else 0
    )

    # ---------------------------------------------------------
    # 3. Prediction Drift trigger
    # ---------------------------------------------------------
    prediction_drift_query = f"""
    SELECT
      monitoring_status
    FROM `{prediction_drift_table}`
    ORDER BY monitoring_timestamp DESC
    LIMIT 1
    """

    prediction_drift_result = list(
        bq.query(prediction_drift_query).result()
    )

    prediction_drift_status = (
        prediction_drift_result[0].monitoring_status
        if prediction_drift_result
        else "PASS"
    )

    # ---------------------------------------------------------
    # 4. New Month Data trigger
    #
    # current_data_month comes from pipeline_config.yaml.
    # Example:
    #   current_data_month: "2026-09-01"
    #
    # Compare it with the last month already used for training.
    # ---------------------------------------------------------
    current_data_month_date = date.fromisoformat(
        current_data_month
    )

    training_state_query = f"""
    SELECT
      MAX(last_training_data_month)
        AS last_training_data_month
    FROM `{training_state_table}`
    WHERE model_name = 'random_forest'
    """

    training_state_result = list(
        bq.query(training_state_query).result()
    )

    last_training_data_month = (
        training_state_result[0].last_training_data_month
        if training_state_result
        else None
    )

    new_month_available = (
        last_training_data_month is None
        or current_data_month_date
        > last_training_data_month
    )

    # ---------------------------------------------------------
    # 5. Final retraining decision
    # ---------------------------------------------------------
    should_retrain = (
        data_quality_status == "ALERT"
        or feature_alert_count > 0
        or prediction_drift_status == "ALERT"
        or new_month_available
    )

    # ---------------------------------------------------------
    # 6. Collect retraining reasons
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

    # ---------------------------------------------------------
    # 7. Print decision summary
    # ---------------------------------------------------------
    print("=" * 60)
    print("Random Forest Retraining Decision")
    print("=" * 60)

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

    print("-" * 60)

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

