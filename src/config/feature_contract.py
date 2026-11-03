"""Pure model feature-contract checks used before cloud training/scoring.

Keep this module independent of GCP clients and model frameworks so CI can
exercise actual production validation without credentials.
"""
import json
import re

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_METADATA = {"provider_id", "dataset_split"}


def validate_features(features):
    if not isinstance(features, list) or not features:
        raise ValueError("Feature list must be a non-empty list of strings")
    if any(not isinstance(f, str) or not _IDENTIFIER.fullmatch(f) for f in features):
        raise ValueError("Invalid feature name in model contract")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate features in model contract")
    forbidden = sorted(_METADATA.intersection(features))
    if forbidden:
        raise ValueError(f"Metadata columns cannot be model features: {forbidden}")
    return list(features)


def resolve_model_features(config, model_type=None):
    model = config.get("model", {})
    selected_type = model_type or model.get("model_type")
    if selected_type not in {"random_forest", "xgboost"}:
        raise ValueError(f"Unsupported model_type: {selected_type}")
    selected = model.get(selected_type)
    if not isinstance(selected, dict):
        raise ValueError(f"Missing model configuration for {selected_type}")
    if "features" not in selected:
        raise ValueError(f"Missing model features for {selected_type}")
    return validate_features(selected["features"])


def validate_registered_features(configured_features, registered_json):
    expected = validate_features(configured_features)
    try:
        registered = json.loads(registered_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid registered FEATURE_COLUMNS_JSON") from exc
    if registered != expected:
        raise ValueError(
            f"Registered features differ in names, count or order: "
            f"registered={registered}; expected={expected}"
        )
    return expected


def validate_source_columns(source_columns, features):
    required = {"provider_id", "dataset_split", *validate_features(features)}
    missing = sorted(required - set(source_columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")


def validate_input_columns(columns, features):
    expected = ["provider_id", *validate_features(features)]
    actual = list(columns)
    if actual != expected:
        raise ValueError(
            f"Batch input schema mismatch: actual={actual}; expected={expected}"
        )
