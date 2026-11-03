"""Offline governance tests for approval history and model rollback."""
import copy
import pytest

from src.config.load_config import load_config
from src.models.approval_ledger import (
    ensure_ledger_sql, ledger_table_from_state, promotion_sql, rollback_sql,
    validate_approval_target, validate_resource, validate_scoring_approval,
    ensure_previous_columns_sql, select_previous_model, resolve_scoring_model,
)
from src.models.rollback_approved_model import check_configuration, compute_config_signatures, load_previous

STATE_TABLE = "fwa-mlops-accelerator-demo.fraud_monitoring.model_training_state"
LEDGER = ledger_table_from_state(STATE_TABLE)
ACTIVE = {
    "model_resource": "projects/123456789/locations/us-central1/models/222",
    "run_id": "approved-new",
    "candidate_status": "APPROVED",
}


@pytest.fixture
def cfg():
    config = copy.deepcopy(load_config())
    # Independent of the active experiment's model_type in YAML.
    config["model"]["model_type"] = "random_forest"
    return config


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
    assert "@target_direction = 'from'" in sql
    assert "from_model_resource = @target_resource" in sql
    assert "from_feature_signature = @target_feature_signature" in sql
    assert "approved_model_signature = @current_model_signature" in sql
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


def test_new_governance_modules_compile_without_syntax_warnings(tmp_path):
    """Catch invalid Python escapes (including BigQuery backtick escapes)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    paths = [
        "src/models/approval_ledger.py",
        "src/models/rollback_approved_model.py",
        "src/models/promote_model_candidate.py",
        "src/scoring/run_batch_prediction.py",
    ]
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::SyntaxWarning",
            "-m",
            "py_compile",
            *paths,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_existing_approval_table_migration_is_additive():
    sqls = ensure_previous_columns_sql(LEDGER)
    assert len(sqls) == 4
    assert all("ADD COLUMN IF NOT EXISTS" in x for x in sqls)
    assert all("DROP" not in x.upper() for x in sqls)


def test_promotion_captures_full_prior_approved_version():
    sql = promotion_sql(STATE_TABLE, LEDGER)
    for field in (
        "approved_model_resource", "approved_run_id", "approved_model_type",
        "approved_feature_signature", "approved_model_signature",
        "last_training_data_month",
    ):
        assert field in sql
    assert "from_model_type" in sql and "from_model_signature" in sql


def test_rollback_audit_captures_current_and_target_versions():
    sql = rollback_sql(STATE_TABLE, LEDGER)
    for param in (
        "@current_model_type", "@current_feature_signature",
        "@current_model_signature", "@current_data_month",
        "@target_model_type", "@target_feature_signature", "@target_model_signature",
    ):
        assert param in sql


def test_a_b_a_rollback_and_scoring_model_resolution(cfg):
    """Revert B directly to A using B's promotion snapshot, even if A
    was approved before the history table existed.
    """
    a = {
        "model_resource": "projects/123456789/locations/us-central1/models/111",
        "run_id": "A-run", "model_type": "random_forest",
        "feature_signature": "features-A", "model_signature": "model-A",
        "data_month": "2026-08-01",
    }
    b = {
        "model_resource": "projects/123456789/locations/us-central1/models/222",
        "run_id": "B-run", "model_type": "random_forest",
        "feature_signature": "features-B", "model_signature": "model-B",
        "data_month": "2026-09-01", "candidate_status": "APPROVED",
    }
    b_promotion = {
        "event_id": "B-promotion-event", "event_type": "PROMOTION",
        "from_model_resource": a["model_resource"],
        "from_run_id": a["run_id"],
        "from_model_type": a["model_type"],
        "from_feature_signature": a["feature_signature"],
        "from_model_signature": a["model_signature"],
        "from_data_month": a["data_month"],
        "to_model_resource": b["model_resource"], "to_run_id": b["run_id"],
    }
    target = select_previous_model(b_promotion, b["model_resource"], b["run_id"])
    assert target["model_resource"] == a["model_resource"]
    assert target["run_id"] == a["run_id"]
    validate_approval_target(target, b, cfg["project_id"], cfg["region"])
    resource, run = resolve_scoring_model(None, target["model_resource"], target["run_id"])
    assert (resource, run) == (a["model_resource"], a["run_id"])
    # Next batch job must not be able to select B explicitly.
    with pytest.raises(ValueError, match="not the active"):
        resolve_scoring_model("222", resource, run)


def test_previous_release_without_complete_metadata_fails_closed():
    row = {
        "event_id": "B", "to_model_resource": ACTIVE["model_resource"],
        "to_run_id": ACTIVE["run_id"],
        "from_model_resource": "projects/123456789/locations/us-central1/models/111",
        "from_run_id": "A-run", "from_model_type": None,
        "from_feature_signature": "sig", "from_model_signature": "sig",
    }
    with pytest.raises(ValueError, match="lacks a complete approval snapshot"):
        select_previous_model(row, ACTIVE["model_resource"], ACTIVE["run_id"])


@pytest.mark.parametrize("difference", ["to_model_resource", "to_run_id"])
def test_previous_release_must_lead_to_current_approved_model(difference):
    row = {
        "event_id": "B", "to_model_resource": ACTIVE["model_resource"],
        "to_run_id": ACTIVE["run_id"],
    }
    row[difference] = "different"
    with pytest.raises(ValueError, match="does not match"):
        select_previous_model(row, ACTIVE["model_resource"], ACTIVE["run_id"])


def test_scoring_rejects_mismatched_explicit_model_after_rollback():
    with pytest.raises(ValueError, match="not the active approved"):
        resolve_scoring_model(
            "222",
            "projects/123456789/locations/us-central1/models/111",
            "A-run",
        )


def test_scoring_requires_approved_run_id():
    with pytest.raises(ValueError, match="No complete approved"):
        resolve_scoring_model(
            None, "projects/123456789/locations/us-central1/models/111", None,
        )


def test_load_previous_uses_latest_transition_with_exact_active_identity():
    class Row(dict):
        def items(self):
            return super().items()

    event = {
        "event_id": "B-promotion", "to_model_resource": ACTIVE["model_resource"],
        "to_run_id": ACTIVE["run_id"],
        "from_model_resource": "projects/123456789/locations/us-central1/models/111",
        "from_run_id": "A-run", "from_model_type": "random_forest",
        "from_feature_signature": "f", "from_model_signature": "m",
        "from_data_month": "2026-08-01",
    }

    class FakeQuery:
        def result(self):
            return [Row(event)]

    class FakeBQ:
        def query(self, sql, job_config):
            assert "ORDER BY event_time DESC" in sql
            assert "to_model_resource = @resource" in sql
            assert "to_run_id = @run_id" in sql
            return FakeQuery()

    class FakeBigQuery:
        class ScalarQueryParameter:
            def __init__(self, *args):
                self.args = args

        class QueryJobConfig:
            def __init__(self, query_parameters):
                self.query_parameters = query_parameters

    selected = load_previous(FakeBQ(), LEDGER, "provider_fraud_model", ACTIVE, FakeBigQuery)
    assert selected["run_id"] == "A-run"
    assert selected["model_resource"] == event["from_model_resource"]
