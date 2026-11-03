"""Offline governance tests for approval history and model rollback."""
import copy
import pytest

from src.config.load_config import load_config
from src.models.approval_ledger import (
    ensure_ledger_sql, ledger_table_from_state, promotion_sql, rollback_sql,
    validate_approval_target, validate_resource, validate_scoring_approval,
)
from src.models.rollback_approved_model import check_configuration, compute_config_signatures

STATE_TABLE = "fwa-mlops-accelerator-demo.fraud_monitoring.model_training_state"
LEDGER = ledger_table_from_state(STATE_TABLE)
ACTIVE = {
    "model_resource": "projects/123456789/locations/us-central1/models/222",
    "run_id": "approved-new",
    "candidate_status": "APPROVED",
}


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


@pytest.fixture
def target(cfg):
    feature_sig, model_sig, _ = compute_config_signatures(cfg, "random_forest")
    return {
        "event_id": "approval-event-1", "model_type": "random_forest",
        "model_resource": "projects/123456789/locations/us-central1/models/111",
        "run_id": "approved-old", "feature_signature": feature_sig,
        "model_signature": model_sig, "data_month": "2026-09-01",
    }


def test_ledger_table_stays_in_monitoring_dataset():
    assert LEDGER == "fwa-mlops-accelerator-demo.fraud_monitoring.model_approval_events"


@pytest.mark.parametrize("bad", [
    "some_table", "project.dataset.table; DROP TABLE x",
    "project.dataset.some-invalid-column", "project.dataset.table\nDELETE", "",
])
def test_bad_table_names_rejected(bad):
    with pytest.raises(ValueError):
        ledger_table_from_state(bad)


def test_ledger_created_non_destructively():
    sql = ensure_ledger_sql(LEDGER).upper()
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "EVENT_ID" in sql
    assert "FROM_MODEL_RESOURCE" in sql
    assert "TO_MODEL_RESOURCE" in sql
    assert "DROP TABLE" not in sql
    assert "CREATE OR REPLACE" not in sql


@pytest.mark.parametrize("builder", [promotion_sql, rollback_sql])
def test_approval_and_rollback_are_atomic_and_audited(builder):
    sql = builder(STATE_TABLE, LEDGER).upper()
    assert "BEGIN TRANSACTION" in sql
    assert "COMMIT TRANSACTION" in sql
    assert "INSERT INTO" in sql
    assert "UPDATE" in sql
    assert sql.count("ASSERT") >= 2
    assert "CREATE OR REPLACE" not in sql
    assert "DELETE FROM" not in sql
    assert "TRUNCATE" not in sql


def test_promotion_requires_pending_candidate():
    sql = promotion_sql(STATE_TABLE, LEDGER)
    assert "candidate_status = 'PENDING_APPROVAL'" in sql
    assert "@candidate_run_id" in sql
    assert "@candidate_resource" in sql
    assert "'PROMOTION'" in sql


def test_rollback_rechecks_history_and_active_version():
    sql = rollback_sql(STATE_TABLE, LEDGER)
    assert "event_type = 'PROMOTION'" in sql
    assert "event_id = @target_event_id" in sql
    assert "approved_model_resource = @current_resource" in sql
    assert "approved_run_id = @current_run_id" in sql
    assert "candidate_status, '') != 'PENDING_APPROVAL'" in sql
    assert "'ROLLBACK'" in sql


def test_valid_target_and_matching_historical_yaml(target, cfg):
    assert validate_approval_target(
        target, ACTIVE, cfg["project_id"], cfg["region"]
    ) == target
    assert check_configuration(cfg, target) == cfg["model"]["random_forest"]["features"]


def test_same_active_model_rejected(target, cfg):
    with pytest.raises(ValueError, match="already the active"):
        validate_approval_target(
            target, {**ACTIVE, "model_resource": target["model_resource"]},
            cfg["project_id"], cfg["region"],
        )


def test_pending_candidate_blocks_rollback(target, cfg):
    with pytest.raises(ValueError, match="pending"):
        validate_approval_target(
            target, {**ACTIVE, "candidate_status": "PENDING_APPROVAL"},
            cfg["project_id"], cfg["region"],
        )


@pytest.mark.parametrize("missing", ["run_id", "model_resource", "feature_signature", "model_signature"])
def test_incomplete_history_rejected(target, cfg, missing):
    target[missing] = None
    with pytest.raises(ValueError):
        validate_approval_target(target, ACTIVE, cfg["project_id"], cfg["region"])


def test_unknown_model_type_rejected(target, cfg):
    target["model_type"] = "custom_model"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_approval_target(target, ACTIVE, cfg["project_id"], cfg["region"])


def test_wrong_region_rejected(cfg):
    with pytest.raises(ValueError, match="wrong region"):
        validate_resource(
            "projects/123456789/locations/europe-west4/models/111",
            cfg["project_id"], cfg["region"],
        )


def test_feature_mismatch_blocks_rollback(target, cfg):
    cfg["model"]["random_forest"]["features"].append("new_signal")
    with pytest.raises(ValueError, match="feature signature"):
        check_configuration(cfg, target)


def test_parameters_mismatch_blocks_rollback(target, cfg):
    cfg["model"]["random_forest"]["parameters"]["n_estimators"] += 1
    with pytest.raises(ValueError, match="model signature"):
        check_configuration(cfg, target)


def test_wrong_configured_model_type_blocks_rollback(target, cfg):
    cfg["model"]["model_type"] = "xgboost"
    with pytest.raises(ValueError, match="Select target model_type"):
        check_configuration(cfg, target)


@pytest.mark.parametrize("requested", [
    "projects/123456789/locations/us-central1/models/111", "111",
])
def test_scoring_allows_only_approved_exact_model(requested, target):
    assert validate_scoring_approval(requested, target["model_resource"]) == target["model_resource"]


@pytest.mark.parametrize("requested", ["222", "1111", "wrong-model"])
def test_scoring_rejects_unapproved_models(requested, target):
    with pytest.raises(ValueError, match="not the active approved"):
        validate_scoring_approval(requested, target["model_resource"])


def test_scoring_fails_closed_without_approved_model():
    with pytest.raises(ValueError, match="No approved model"):
        validate_scoring_approval("111", None)
