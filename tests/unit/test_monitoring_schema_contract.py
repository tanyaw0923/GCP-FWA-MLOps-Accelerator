"""Guard against reintroducing the obsolete Random Forest monitor schema."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_only_generic_monitoring_path_is_retained():
    assert not (ROOT / "src/pipeline/components/run_random_forest_monitoring_component.py").exists()
    assert not (ROOT / "src/monitoring/run_random_forest_monitoring.py").exists()


def test_retraining_reader_agrees_with_active_prediction_drift_schema():
    writer = (ROOT / "sql/monitoring/model_monitoring.sql").read_text()
    reader = (
        ROOT / "src/pipeline/components/check_retraining_decision_component.py"
    ).read_text()

    # The active drift writer appends records with a status and timestamp.
    assert "model_prediction_drift" in writer
    assert "check_date TIMESTAMP" in writer
    assert "status STRING" in writer
    assert "INSERT INTO" in writer

    # Retraining reads the same columns, rather than the legacy names.
    assert "SELECT status" in reader
    assert "ORDER BY check_date DESC" in reader


def test_generic_monitoring_is_model_agnostic():
    pipeline = (ROOT / "src/pipeline/model_training_pipeline.py").read_text()
    assert "run_random_forest_monitoring_component" not in pipeline
