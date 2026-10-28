from types import SimpleNamespace

import pytest
from google.cloud import bigquery

from src.pipeline.components.model_promotion_gate_component import (
    model_promotion_gate_component,
)


class FakeQueryJob:
    def __init__(self, rows=None):
        self._rows = rows or []

    def result(self):
        return self._rows


class FakeBigQueryClient:
    def __init__(
        self,
        *,
        approved_run_id="champion-run",
        champion_metrics=None,
        challenger_metrics=None,
    ):
        self.approved_run_id = approved_run_id
        self.metrics = {
            "champion-run": champion_metrics
            or {
                "pr_auc": 0.70,
                "precision": 0.60,
                "recall": 0.65,
                "f1_score": 0.62,
            },
            "challenger-run": challenger_metrics
            or {
                "pr_auc": 0.72,
                "precision": 0.61,
                "recall": 0.66,
                "f1_score": 0.63,
            },
        }

    def query(self, query, job_config=None):
        if "ALTER TABLE" in query:
            return FakeQueryJob()

        if "SELECT approved_run_id" in query:
            return FakeQueryJob(
                [
                    SimpleNamespace(
                        approved_run_id=self.approved_run_id,
                    )
                ]
            )

        if "metric_name" in query and "metric_value" in query:
            run_id = next(
                parameter.value
                for parameter in job_config.query_parameters
                if parameter.name == "run_id"
            )
            metrics = self.metrics.get(run_id, {})

            return FakeQueryJob(
                [
                    SimpleNamespace(
                        metric_name=name,
                        metric_value=value,
                    )
                    for name, value in metrics.items()
                ]
            )

        raise AssertionError(f"Unexpected query: {query}")


def run_gate(monkeypatch, fake_client, **overrides):
    monkeypatch.setattr(
        bigquery,
        "Client",
        lambda project=None: fake_client,
    )

    params = {
        "project_id": "project",
        "experiment_metrics_table": "project.experiments.metrics",
        "training_state_table": "project.monitoring.training_state",
        "challenger_run_id": "challenger-run",
        "min_pr_auc_improvement": 0.01,
        "min_recall_change": 0.0,
        "min_f1_change": 0.0,
        "min_precision_change": 0.0,
        "state_key": "provider_fraud_model",
    }
    params.update(overrides)

    return model_promotion_gate_component.python_func(**params)


def test_challenger_passes_all_promotion_thresholds(monkeypatch):
    client = FakeBigQueryClient()

    result = run_gate(monkeypatch, client)

    assert result.eligible_for_promotion is True
    assert "satisfied all" in result.decision_reason.lower()


def test_challenger_fails_when_one_metric_regresses(monkeypatch):
    client = FakeBigQueryClient(
        challenger_metrics={
            "pr_auc": 0.72,
            "precision": 0.59,
            "recall": 0.66,
            "f1_score": 0.63,
        }
    )

    result = run_gate(monkeypatch, client)

    assert result.eligible_for_promotion is False
    assert "precision" in result.decision_reason


def test_exact_threshold_is_eligible(monkeypatch):
    client = FakeBigQueryClient(
        challenger_metrics={
            "pr_auc": 0.71,
            "precision": 0.60,
            "recall": 0.65,
            "f1_score": 0.62,
        }
    )

    result = run_gate(monkeypatch, client)

    assert result.eligible_for_promotion is True


def test_no_approved_champion_bootstraps_candidate(monkeypatch):
    client = FakeBigQueryClient(
        approved_run_id=None,
    )

    result = run_gate(monkeypatch, client)

    assert result.eligible_for_promotion is True
    assert "bootstrap" in result.decision_reason.lower()


def test_missing_challenger_metric_raises(monkeypatch):
    client = FakeBigQueryClient(
        challenger_metrics={
            "pr_auc": 0.72,
            "precision": 0.61,
            "recall": 0.66,
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Challenger metrics are missing",
    ):
        run_gate(monkeypatch, client)
