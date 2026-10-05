from pathlib import Path
import yaml

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "pipeline_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()

    print("Project:", config["project_id"])
    print("Region:", config["region"])
    print("Model:", config["model"]["model_type"])
    print("Experiment:", config["model"]["experiment_name"])

    print("\nFeatures:")
    for feature in config["features"]:
        print("-", feature)
