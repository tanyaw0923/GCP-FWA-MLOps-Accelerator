"""Offline validation of the isolated Vertex cloud smoke test.

The cloud script must remain importable without GCP credentials or cloud SDKs.
"""
import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "integration" / "isolated_vertex_smoke.py"
SPEC = importlib.util.spec_from_file_location("isolated_vertex_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def args(**updates):
    defaults = dict(
        project="fwa-isolated-test-123",
        region="us-central1",
        serving_image=(
            "us-central1-docker.pkg.dev/"
            "fwa-isolated-test-123/smoke/rf-serving:smoke"
        ),
        acknowledge_cost=True,
        timeout_minutes=40,
        machine_type="n1-standard-2",
        keep_resources=False,
    )
    defaults.update(updates)
    return argparse.Namespace(**defaults)


def test_dedicated_project_configuration_is_accepted():
    smoke.validate_args(args())


@pytest.mark.parametrize(
    "changes",
    [
        {"project": "fwa-mlops-accelerator-demo"},
        {"project": "bad_project"},
        {"serving_image": "us-central1-docker.pkg.dev/other-project/other/image:tag"},
        {"serving_image": "us-central1-docker.pkg.dev/fwa-mlops-accelerator-demo/smoke/rf:tag"},
        {"region": "not_a_region"},
        {"acknowledge_cost": False},
        {"timeout_minutes": 2},
    ],
)
def test_unsafe_configuration_is_rejected_before_cloud_access(changes):
    with pytest.raises(ValueError):
        smoke.validate_args(args(**changes))


def test_three_synthetic_predictions_preserve_provider_lineage():
    rows = [
        {"provider_id": "SMOKE001", "prediction": "0.1"},
        {"provider_id": "SMOKE002", "prediction": "0.8"},
        {"provider_id": "SMOKE003", "prediction": "0.4"},
    ]
    smoke.verify_prediction_rows(rows)


@pytest.mark.parametrize(
    "changed",
    [
        {"provider_id": "SMOKE001", "prediction": "-0.2"},
        {"provider_id": "SMOKE001", "prediction": "1.2"},
        {"provider_id": "SMOKE001", "prediction": '{"unexpected": 0.5}'},
        {"provider_id": None, "prediction": "0.7"},
    ],
)
def test_prediction_validation_rejects_invalid_probability_or_missing_id(changed):
    rows = [
        changed,
        {"provider_id": "SMOKE002", "prediction": "0.8"},
        {"provider_id": "SMOKE003", "prediction": "0.4"},
    ]
    with pytest.raises(AssertionError):
        smoke.verify_prediction_rows(rows)


def test_duplicate_provider_predictions_rejected():
    rows = [
        {"provider_id": "SMOKE001", "prediction": "0.1"},
        {"provider_id": "SMOKE001", "prediction": "0.8"},
        {"provider_id": "SMOKE003", "prediction": "0.4"},
    ]
    with pytest.raises(AssertionError, match="Duplicate"):
        smoke.verify_prediction_rows(rows)


def test_supports_vertex_key_column():
    rows = [
        {"key": "SMOKE001", "prediction": 0.1},
        {"key": "SMOKE002", "prediction": 0.8},
        {"key": "SMOKE003", "prediction": 0.4},
    ]
    smoke.verify_prediction_rows(rows)
