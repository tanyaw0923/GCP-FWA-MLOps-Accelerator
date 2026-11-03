"""Behavioral tests for the same feature validation used by cloud entry points."""
import copy
import json

import pytest
import yaml

from src.config.load_config import load_config
from src.config.feature_contract import (
    resolve_model_features,
    validate_features,
    validate_registered_features,
    validate_source_columns,
    validate_input_columns,
)


@pytest.fixture
def config():
    return copy.deepcopy(load_config())


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_models_resolve_to_their_own_configured_feature_order(config, model_type):
    expected = config["model"][model_type]["features"]
    assert resolve_model_features(config, model_type) == expected


def test_switching_model_type_changes_active_feature_contract(config):
    # Do not assume pipeline_config.yaml defaults to Random Forest:
    # developers may select XGBoost for experiments.
    config["model"]["model_type"] = "random_forest"
    rf = resolve_model_features(config)
    assert rf == config["model"]["random_forest"]["features"]

    config["model"]["model_type"] = "xgboost"
    xgb = resolve_model_features(config)
    assert xgb == config["model"]["xgboost"]["features"]

    # Distinct configurations should be selected without stale state.
    # Verify switching back, even when the original YAML selected XGBoost.
    config["model"]["model_type"] = "random_forest"
    assert resolve_model_features(config) == rf


def test_new_configured_feature_propagates_without_hardcoded_fallback(config):
    features = config["model"]["random_forest"]["features"]
    features.append("new_provider_signal")
    result = resolve_model_features(config, "random_forest")
    assert result[-1] == "new_provider_signal"
    assert validate_registered_features(result, json.dumps(result)) == result
    validate_input_columns(["provider_id", *result], result)


def test_reordered_registered_features_fail_preflight(config):
    features = resolve_model_features(config, "random_forest")
    with pytest.raises(ValueError, match="names, count or order"):
        validate_registered_features(features, json.dumps(list(reversed(features))))


def test_missing_or_extra_registered_feature_fail_preflight(config):
    features = resolve_model_features(config, "xgboost")
    for registered in [features[:-1], features + ["extra_signal"]]:
        with pytest.raises(ValueError, match="names, count or order"):
            validate_registered_features(features, json.dumps(registered))


@pytest.mark.parametrize("payload", [None, "", "not-json", "{}", '{"features":["a"]}', 'null'])
def test_invalid_registered_feature_json_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_registered_features(["score"], payload)


@pytest.mark.parametrize(
    "features",
    [
        [],
        None,
        "score",
        ["score", "score"],
        ["provider_id", "score"],
        ["score", "dataset_split"],
        ["bad-name"],
        ["x; DROP TABLE"],
        ["has space"],
        ["123name"],
        ["score", 12],
    ],
)
def test_invalid_feature_lists_fail_fast(features):
    with pytest.raises(ValueError):
        validate_features(features)


def test_missing_selected_model_fails_fast(config):
    del config["model"]["xgboost"]
    with pytest.raises(ValueError, match="Missing model configuration"):
        resolve_model_features(config, "xgboost")


def test_unsupported_model_fails_fast(config):
    config["model"]["model_type"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported model_type"):
        resolve_model_features(config)


def test_feature_contract_preserves_order_and_does_not_mutate_input():
    original = ["second_feature", "first_feature"]
    result = validate_features(original)
    assert result == original
    assert result is not original


def test_source_schema_requires_provider_id_split_and_all_features():
    validate_source_columns(
        ["provider_id", "dataset_split", "score", "distance", "extra_column"],
        ["score", "distance"],
    )
    with pytest.raises(ValueError, match="distance"):
        validate_source_columns(["provider_id", "dataset_split", "score"], ["score", "distance"])
    with pytest.raises(ValueError, match="provider_id"):
        validate_source_columns(["dataset_split", "score"], ["score"])


@pytest.mark.parametrize(
    "columns",
    [
        ["score", "provider_id", "distance"],
        ["provider_id", "distance", "score"],
        ["provider_id", "score"],
        ["provider_id", "score", "distance", "extra"],
        ["provider_id", "score", "score"],
    ],
)
def test_batch_input_schema_rejects_wrong_order_missing_extra_or_duplicate(columns):
    with pytest.raises(ValueError, match="Batch input schema mismatch"):
        validate_input_columns(columns, ["score", "distance"])


def test_yaml_file_changes_propagate_through_real_config_loader(tmp_path, config):
    config["model"]["random_forest"]["features"] = ["signal_alpha", "signal_beta"]
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = load_config(config_file)
    assert resolve_model_features(loaded, "random_forest") == [
        "signal_alpha",
        "signal_beta",
    ]
    assert config["model"]["xgboost"]["features"] == loaded["model"]["xgboost"]["features"]


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")
