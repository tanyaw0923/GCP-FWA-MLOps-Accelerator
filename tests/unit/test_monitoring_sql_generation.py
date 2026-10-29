"""Tests for configuration-driven feature monitoring SQL."""
import pytest

from src.monitoring.generate_monitoring_sql import compile_monitoring_sql


TEMPLATE = (
    "SELECT ({{NULL_CHECKS}}) AS missing, "
    "({{INVALID_CHECKS}}) AS invalid; "
    "UNPIVOT (v FOR name IN ({{DRIFT_FEATURE_COLUMNS}}))"
)


def config(features, rules):
    return {
        "model": {"model_type": "random_forest",
                  "random_forest": {"features": features}},
        "monitoring": {"feature_validation_rules": rules},
    }


def test_new_feature_is_covered_in_quality_and_drift():
    sql = compile_monitoring_sql(
        config(["distance"], {"distance": {"min": 0}}), TEMPLATE
    )
    assert "COUNTIF(`distance` IS NULL)" in sql
    assert "COUNTIF(`distance` < 0)" in sql
    assert "IN (`distance`)" in sql


def test_percentage_boundaries():
    sql = compile_monitoring_sql(
        config(["ratio"], {"ratio": {"min": 0, "max": 1}}), TEMPLATE
    )
    assert "`ratio` < 0" in sql
    assert "`ratio` > 1" in sql


def test_missing_rule_fails_closed():
    with pytest.raises(ValueError, match="Missing quality rule"):
        compile_monitoring_sql(config(["new_feature"], {}), TEMPLATE)


def test_unsafe_identifier_fails():
    with pytest.raises(ValueError, match="Unsafe feature identifier"):
        compile_monitoring_sql(
            config(["bad; DROP TABLE x"], {"bad; DROP TABLE x": {"min": 0}}),
            TEMPLATE,
        )


def test_duplicate_features_fail():
    with pytest.raises(ValueError, match="unique"):
        compile_monitoring_sql(
            config(["distance", "distance"], {"distance": {"min": 0}}),
            TEMPLATE,
        )
