import copy

import pytest

from src.config.load_config import load_config


REQUIRED_MODEL_KEYS = {
    "model_name",
    "experiment_name",
    "features",
    "parameters",
    "training_image",
    "serving_container_image_uri",
}


def validate_model_contract(config, model_type):
    model_config = config["model"]

    if model_type not in model_config:
        raise ValueError(
            f"Missing model configuration for {model_type}"
        )

    active_model = model_config[model_type]

    missing = REQUIRED_MODEL_KEYS - set(active_model)
    if missing:
        raise ValueError(
            "Missing required model keys: "
            + ", ".join(sorted(missing))
        )

    if not active_model["features"]:
        raise ValueError(
            f"Feature list is empty for {model_type}"
        )


@pytest.mark.parametrize(
    "model_type",
    ["random_forest", "xgboost"],
)
def test_model_specific_contract_is_complete(model_type):
    config = load_config()

    validate_model_contract(config, model_type)


def test_config_has_no_legacy_top_level_feature_fallback():
    config = load_config()

    assert "features" not in config


def test_empty_model_feature_list_is_rejected():
    config = copy.deepcopy(load_config())
    config["model"]["random_forest"]["features"] = []

    with pytest.raises(
        ValueError,
        match="Feature list is empty",
    ):
        validate_model_contract(
            config,
            "random_forest",
        )


def test_missing_model_contract_is_rejected():
    config = copy.deepcopy(load_config())
    del config["model"]["xgboost"]

    with pytest.raises(
        ValueError,
        match="Missing model configuration",
    ):
        validate_model_contract(
            config,
            "xgboost",
        )
