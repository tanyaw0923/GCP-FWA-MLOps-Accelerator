from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
    ],
)
def model_promotion_gate_component(
    project_id: str,
    experiment_metrics_table: str,
    training_state_table: str,
    challenger_run_id: str,
    min_pr_auc_improvement: float,
    min_recall_change: float,
    min_f1_change: float,
    min_precision_change: float,
    state_key: str = "provider_fraud_model",
) -> NamedTuple(
    "Outputs",
    [
        ("eligible_for_promotion", bool),
        ("decision_reason", str),
    ],
):
    from collections import namedtuple

    from google.cloud import bigquery

    outputs = namedtuple(
        "Outputs",
        [
            "eligible_for_promotion",
            "decision_reason",
        ],
    )

    bq = bigquery.Client(
        project=project_id
    )

    bq.query(
        f"""
        ALTER TABLE `{training_state_table}`
        ADD COLUMN IF NOT EXISTS approved_run_id STRING
        """
    ).result()

    state_rows = list(
        bq.query(
            f"""
            SELECT approved_run_id
            FROM `{training_state_table}`
            WHERE model_name = @state_key
            LIMIT 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "state_key",
                        "STRING",
                        state_key,
                    )
                ]
            ),
        ).result()
    )

    approved_run_id = (
        state_rows[0].approved_run_id
        if state_rows
        else None
    )

    def load_metrics(run_id):
        rows = list(
            bq.query(
                f"""
                SELECT
                  metric_name,
                  metric_value
                FROM `{experiment_metrics_table}`
                WHERE run_id = @run_id
                  AND metric_name IN (
                    'pr_auc',
                    'precision',
                    'recall',
                    'f1_score'
                  )
                """,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter(
                            "run_id",
                            "STRING",
                            run_id,
                        )
                    ]
                ),
            ).result()
        )

        return {
            row.metric_name: float(
                row.metric_value
            )
            for row in rows
        }

    challenger = load_metrics(
        challenger_run_id
    )

    required_metrics = {
        "pr_auc",
        "precision",
        "recall",
        "f1_score",
    }

    missing_challenger = (
        required_metrics
        - set(challenger)
    )

    if missing_challenger:
        raise RuntimeError(
            "Challenger metrics are missing: "
            + ", ".join(
                sorted(
                    missing_challenger
                )
            )
        )

    if not approved_run_id:
        reason = (
            "No approved_run_id is recorded yet. "
            "Bootstrap promotion-gate pass; "
            "manual approval is still required."
        )

        print("=" * 70)
        print("MODEL PROMOTION GATE")
        print("=" * 70)
        print(
            f"Challenger run: "
            f"{challenger_run_id}"
        )
        print(
            "Approved run: not available"
        )
        print(
            "Eligible for promotion: True"
        )
        print(
            f"Reason: {reason}"
        )

        return outputs(
            True,
            reason,
        )

    champion = load_metrics(
        approved_run_id
    )

    missing_champion = (
        required_metrics
        - set(champion)
    )

    if missing_champion:
        raise RuntimeError(
            "Champion metrics are missing: "
            + ", ".join(
                sorted(
                    missing_champion
                )
            )
        )

    pr_auc_change = (
        challenger["pr_auc"]
        - champion["pr_auc"]
    )

    precision_change = (
        challenger["precision"]
        - champion["precision"]
    )

    recall_change = (
        challenger["recall"]
        - champion["recall"]
    )

    f1_change = (
        challenger["f1_score"]
        - champion["f1_score"]
    )

    checks = {
        "pr_auc": (
            pr_auc_change
            >= min_pr_auc_improvement
        ),
        "precision": (
            precision_change
            >= min_precision_change
        ),
        "recall": (
            recall_change
            >= min_recall_change
        ),
        "f1_score": (
            f1_change
            >= min_f1_change
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    eligible = (
        len(failed_checks) == 0
    )

    if eligible:
        reason = (
            "Challenger satisfied all "
            "promotion criteria."
        )
    else:
        reason = (
            "Challenger failed promotion criteria: "
            + ", ".join(
                failed_checks
            )
        )

    print("=" * 70)
    print("MODEL PROMOTION GATE")
    print("=" * 70)

    print(
        f"Champion run: "
        f"{approved_run_id}"
    )

    print(
        f"Challenger run: "
        f"{challenger_run_id}"
    )

    print("\nChampion metrics")
    print("-" * 70)

    for metric in sorted(
        required_metrics
    ):
        print(
            f"{metric}: "
            f"{champion[metric]:.4f}"
        )

    print("\nChallenger metrics")
    print("-" * 70)

    for metric in sorted(
        required_metrics
    ):
        print(
            f"{metric}: "
            f"{challenger[metric]:.4f}"
        )

    print("\nMetric changes")
    print("-" * 70)

    print(
        f"PR-AUC change: "
        f"{pr_auc_change:+.4f}"
    )
    print(
        f"Precision change: "
        f"{precision_change:+.4f}"
    )
    print(
        f"Recall change: "
        f"{recall_change:+.4f}"
    )
    print(
        f"F1 change: "
        f"{f1_change:+.4f}"
    )

    print("\nDecision")
    print("-" * 70)

    print(
        f"Eligible for promotion: "
        f"{eligible}"
    )

    print(
        f"Reason: {reason}"
    )

    return outputs(
        eligible,
        reason,
    )
