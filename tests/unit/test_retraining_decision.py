from datetime import date
from types import SimpleNamespace

import pytest
from google.cloud import bigquery

from src.pipeline.components.check_retraining_decision_component import (
    check_retraining_decision_component,
)


class FakeQueryJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class FakeBigQueryClient:
    def __init__(
        self,
        *,
        data_quality_status="PASS",
        feature_alert_count=0,
        prediction_drift_status="PASS",
        performance_status="PASS",
        last_training_data_month=date(2026, 10, 1),
        approved_feature_signature="feature-v1",
        approved_model_signature="model-v1",
        approved_model_type="xgboost",
    ):
        self.data_quality_status = data_quality_status
        self.feature_alert_count = feature_alert_count
        self.prediction_drift_status = prediction_drift_status
        self.performance_status = performance_status
        self.last_training_data_month = last_training_data_month
        self.approved_feature_signature = approved_feature_signature
        self.approved_model_signature = approved_model_signature
        self.approved_model_type = approved_model_type

    def query(self, query, job_config=None):
        if "FROM `project.monitoring.data_quality`" in query:
            rows = [
                SimpleNamespace(
                    monitoring_status=self.data_quality_status,
                )
            ]

        elif "FROM `project.monitoring.feature_drift`" in query:
            rows = [
                SimpleNamespace(
                    alert_count=self.feature_alert_count,
                )
            ]

        elif "FROM `project.monitoring.prediction_drift`" in query:
            rows = [
                SimpleNamespace(
                    status=self.prediction_drift_status,
                )
            ]

        elif "FROM `project.monitoring.performance`" in query:
            rows = [
                SimpleNamespace(
                    monitoring_status=self.performance_status,
                    model_name="provider_fraud_model",
                    model_version="v1",
                    run_id="run-approved",
                    alert_reason=None,
                    evaluation_timestamp=None,
                )
            ]

        elif "FROM `project.monitoring.training_state`" in query:
            if self.last_training_data_month is None:
                rows = []
            else:
                rows = [
                    SimpleNamespace(
                        last_training_data_month=(
                            self.last_training_data_month
                        ),
                        approved_feature_signature=(
                            self.approved_feature_signature
                        ),
                        approved_model_signature=(
                            self.approved_model_signature
                        ),
                        approved_model_type=(
                            self.approved_model_type
                        ),
                    )
                ]

        else:
            raise AssertionError(f"Unexpected query: {query}")

        return FakeQueryJob(rows)


def run_decision(monkeypatch, fake_client, **overrides):
    monkeypatch.setattr(
        bigquery,
        "Client",
        lambda project=None: fake_client,
    )

    params = {
        "project_id": "project",
        "current_data_month": "2026-10-01",
        "current_feature_signature": "feature-v1",
        "current_model_signature": "model-v1",
        "current_model_type": "xgboost",
        "data_quality_table": "project.monitoring.data_quality",
        "feature_drift_table": "project.monitoring.feature_drift",
        "prediction_drift_table": "project.monitoring.prediction_drift",
        "performance_degradation_table": "project.monitoring.performance",
        "training_state_table": "project.monitoring.training_state",
        "state_key": "provider_fraud_model",
    }
    params.update(overrides)

    return check_retraining_decision_component.python_func(**params)


def test_no_trigger_does_not_retrain(monkeypatch):
    client = FakeBigQueryClient()

    assert run_decision(monkeypatch, client) is False


def test_new_data_month_triggers_retraining(monkeypatch):
    client = FakeBigQueryClient(
        last_training_data_month=date(2026, 9, 1),
    )

    assert run_decision(monkeypatch, client) is True


def test_feature_signature_change_triggers_retraining(monkeypatch):
    client = FakeBigQueryClient()

    assert (
        run_decision(
            monkeypatch,
            client,
            current_feature_signature="feature-v2",
        )
        is True
    )


def test_model_signature_change_triggers_retraining(monkeypatch):
    client = FakeBigQueryClient()

    assert (
        run_decision(
            monkeypatch,
            client,
            current_model_signature="model-v2",
        )
        is True
    )


@pytest.mark.parametrize(
    "client_kwargs",
    [
        {"data_quality_status": "ALERT"},
        {"feature_alert_count": 1},
        {"prediction_drift_status": "ALERT"},
        {"performance_status": "ALERT"},
    ],
)
def test_monitoring_alert_triggers_retraining(
    monkeypatch,
    client_kwargs,
):
    client = FakeBigQueryClient(**client_kwargs)

    assert run_decision(monkeypatch, client) is True


def test_missing_approved_state_bootstraps_retraining(monkeypatch):
    client = FakeBigQueryClient(
        last_training_data_month=None,
    )

    assert run_decision(monkeypatch, client) is True
