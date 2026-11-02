"""Guard monitoring history against destructive table replacement.

This is a static source check. It does not require BigQuery credentials and
does not inspect or modify production datasets.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MONITORING_SOURCES = (
    "sql/monitoring/model_monitoring.sql",
    "sql/monitoring/model_monitoring.template.sql",
    "sql/monitoring/isolation_forest_model_monitoring.sql",
    "sql/monitoring/model_performance_degradation.sql",
    "src/monitoring/run_model_performance_monitoring.py",
)

DESTRUCTIVE_STATEMENTS = re.compile(
    r"\b(?:CREATE\s+OR\\s+REPLACE\\s+TABLE|"
    r"TRUNCATE\\s+TABLE|DROP\\s+TABLE|DELETE\\s+FROM)\b",
    flags=re.IGNORECASE,
)


def test_monitoring_writers_do_not_replace_or_clear_history():
    for relative_path in MONITORING_SOURCES:
        source = (ROOT / relative_path).read_text()
        # Ignore SQL comments to avoid flagging documentation-only examples.
        without_line_comments = "\n".join(
            line.split("--", 1)[0] for line in source.splitlines()
        )
        assert DESTRUCTIVE_STATEMENTS.search(without_line_comments) is None, (
            f"Destructive BigQuery statement in {relative_path}"
        )


def test_generic_monitoring_appends_results():
    for path in (
        "sql/monitoring/model_monitoring.sql",
        "sql/monitoring/model_monitoring.template.sql",
    ):
        content = (ROOT / path).read_text()
        assert content.count("INSERT INTO") == 3
        assert content.count("CREATE TABLE IF NOT EXISTS") == 3


def test_legacy_random_forest_monitor_is_retired():
    assert not (
        ROOT / "src/pipeline/components/run_random_forest_monitoring_component.py"
    ).exists()
    assert not (ROOT / "src/monitoring/run_random_forest_monitoring.py").exists()
