"""Generate BigQuery quality/drift SQL from the active model's features.

Run: python -m src.monitoring.generate_monitoring_sql
The generated SQL is intended to be executed by the existing monitoring job.
"""
import re
from pathlib import Path
from src.config.load_config import load_config

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "sql/monitoring/model_monitoring.template.sql"
OUTPUT = ROOT / "sql/monitoring/model_monitoring.sql"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def compile_monitoring_sql(config, template):
    model_type = config["model"]["model_type"]
    features = config["model"][model_type]["features"]
    rules = config["monitoring"]["feature_validation_rules"]
    if not features or len(features) != len(set(features)):
        raise ValueError("Model features must be nonempty and unique.")

    null_checks, invalid_checks = [], []
    q = chr(96)  # BigQuery identifier quoting (backtick)
    for feature in features:
        if not IDENTIFIER.fullmatch(feature):
            raise ValueError(f"Unsafe feature identifier: {feature}")
        if feature not in rules or not isinstance(rules[feature], dict):
            raise ValueError(f"Missing quality rule for feature: {feature}")
        rule = rules[feature]
        if not rule.get("allow_null", False):
            null_checks.append(f"COUNTIF({q}{feature}{q} IS NULL)")
        conditions = []
        for key, op in [("min", "<"), ("max", ">")]:
            if key in rule:
                bound = rule[key]
                if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                    raise ValueError(f"Non-numeric {key} for {feature}")
                conditions.append(f"{q}{feature}{q} {op} {bound}")
        if conditions:
            invalid_checks.append("COUNTIF(" + " OR ".join(conditions) + ")")
    replacements = {
        "{{NULL_CHECKS}}": "\n      + ".join(null_checks) or "0",
        "{{INVALID_CHECKS}}": "\n      + ".join(invalid_checks) or "0",
        "{{DRIFT_FEATURE_COLUMNS}}": ",\n      ".join(q + f + q for f in features),
    }
    for token, value in replacements.items():
        if template.count(token) != 1:
            raise ValueError(f"Expected exactly one {token} in template")
        template = template.replace(token, value)
    return (
        "-- AUTO-GENERATED from pipeline_config.yaml; do not edit directly.\n"
        f"-- Active model: {model_type}\n"
        "-- Regenerate: python -m src.monitoring.generate_monitoring_sql\n\n"
        + template
    )


def main():
    sql = compile_monitoring_sql(load_config(), TEMPLATE.read_text())
    OUTPUT.write_text(sql)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
