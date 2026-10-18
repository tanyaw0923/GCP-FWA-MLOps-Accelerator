from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(
    base_image="python:3.11",
)
def publish_training_metadata_component(
    feature_dataset: Input[Dataset],
    training_result_json: str,
    model_artifact_uri: str,
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Publish model and evaluation artifacts to KFP / Vertex ML Metadata."""
    import json

    result = json.loads(
        training_result_json
    )

    model_output_uri = result[
        "model_output_uri"
    ]

    model.uri = model_output_uri

    model.metadata[
        "model_type"
    ] = result[
        "model_type"
    ]

    model.metadata[
        "experiment_id"
    ] = result[
        "experiment_id"
    ]

    model.metadata[
        "run_id"
    ] = result[
        "run_id"
    ]

    model.metadata[
        "experiment_name"
    ] = result[
        "experiment_name"
    ]

    model.metadata[
        "feature_version"
    ] = result[
        "feature_version"
    ]

    model.metadata[
        "dataset_version"
    ] = result[
        "dataset_version"
    ]

    model.metadata[
        "git_commit"
    ] = result[
        "git_commit"
    ]

    model.metadata[
        "training_start"
    ] = result[
        "training_start"
    ]

    model.metadata[
        "training_end"
    ] = result[
        "training_end"
    ]

    model.metadata[
        "feature_dataset_uri"
    ] = feature_dataset.uri

    model.metadata[
        "registry_artifact_uri"
    ] = model_artifact_uri

    if (
        "classification_threshold"
        in result
    ):
        model.metadata[
            "classification_threshold"
        ] = float(
            result[
                "classification_threshold"
            ]
        )

    metric_values = result[
        "metrics"
    ]

    for (
        metric_name,
        metric_value,
    ) in metric_values.items():
        metrics.log_metric(
            metric_name,
            float(
                metric_value
            ),
        )

    metrics.metadata[
        "experiment_id"
    ] = result[
        "experiment_id"
    ]

    metrics.metadata[
        "run_id"
    ] = result[
        "run_id"
    ]

    metrics.metadata[
        "model_type"
    ] = result[
        "model_type"
    ]

    metrics.metadata[
        "feature_version"
    ] = result[
        "feature_version"
    ]

    metrics.metadata[
        "dataset_version"
    ] = result[
        "dataset_version"
    ]

    metrics.metadata[
        "git_commit"
    ] = result[
        "git_commit"
    ]

    print(
        "Published KFP / Vertex ML Metadata artifacts."
    )

    print(
        f"Feature dataset URI: "
        f"{feature_dataset.uri}"
    )

    print(
        f"Model URI: "
        f"{model.uri}"
    )

    print(
        f"Metrics: "
        f"{metric_values}"
    )
