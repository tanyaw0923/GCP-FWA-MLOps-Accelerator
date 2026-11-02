"""Static safeguards for opt-in performance-monitoring test mode."""
from pathlib import Path

SOURCE = Path("src/monitoring/run_model_performance_monitoring.py").read_text()


def test_test_mode_routes_output_to_isolated_dataset():
    assert 'FWA_MONITOR_TEST_MODE' in SOURCE
    assert 'fraud_monitoring_test.model_performance_degradation' in SOURCE
    assert 'if not TEST_MODE:' in SOURCE


def test_test_mode_caps_both_queries_and_marks_notification():
    assert SOURCE.count('ROW_NUMBER() OVER (ORDER BY') >= 2
    assert 'else "")' in SOURCE
    assert '("[TEST] " if TEST_MODE else "")' in SOURCE
