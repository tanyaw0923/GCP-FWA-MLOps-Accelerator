"""Dry-run-first, audited rollback to a previously approved Vertex model.

Never rewrites GCS model artifacts or deletes current model versions.
The BigQuery active-model pointer and audit row update in one transaction.
An operator must run a separate post-rollback smoke check before scoring.
"""
import argparse
import getpass
import hashlib
import json
import uuid

from src.config.load_config import load_config
from src.models.approval_ledger import (
    ledger_table_from_state,
    rollback_sql,
    validate_approval_target,
    validate_resource,
)


def _hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def compute_config_signatures(config, model_type):
    from src.config.feature_contract import resolve_model_features

    active = config["model"][model_type]
    features = resolve_model_features(config, model_type)
    feature_signature = _hash({
        "feature_sql": config["feature_sql"],
        "derived_feature_sql": config["derived_feature_sql"],
        "features": features,
    })
    model_signature = _hash({
        "model_type": model_type,
        "model_name": active["model_name"],
        "parameters": active["parameters"],
        "training_image_uri": active["training_image"].strip(),
        "artifact_format": (
            "joblib" if model_type == "random_forest"
            else "native_xgboost_booster"
        ),
        "serving_container_image_uri": active["serving_container_image_uri"].strip(),
    })
    return feature_signature, model_signature, features


def check_configuration(config, target):
    model_type = target["model_type"]
    if config["model"]["model_type"] != model_type:
        raise ValueError(
            f"Select target model_type={model_type} in the supplied YAML first"
        )
    feature_sig, model_sig, features = compute_config_signatures(config, model_type)
    if feature_sig != target["feature_signature"]:
        raise ValueError("YAML feature signature differs from historically approved version")
    if model_sig != target["model_signature"]:
        raise ValueError("YAML model signature differs from historically approved version")
    return features


def load_active(bq, table, state_key, bigquery):
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("state_key", "STRING", state_key),
    ])
    rows = list(bq.query(
        f"""SELECT approved_model_resource, approved_run_id, candidate_status
            FROM \`{table}\` WHERE model_name = @state_key""",
        job_config=job_config,
    ).result())
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one active model state row")
    row = dict(rows[0].items())
    if not row["approved_model_resource"] or not row["approved_run_id"]:
        raise RuntimeError("An approved model resource and run ID are required")
    return {
        "model_resource": row["approved_model_resource"],
        "run_id": row["approved_run_id"],
        "candidate_status": row["candidate_status"],
    }


def load_target(bq, table, state_key, event_id, bigquery):
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("state_key", "STRING", state_key),
        bigquery.ScalarQueryParameter("event_id", "STRING", event_id),
    ])
    rows = list(bq.query(
        f"""SELECT event_id, to_model_resource AS model_resource,
                to_run_id AS run_id, to_model_type AS model_type,
                to_feature_signature AS feature_signature,
                to_model_signature AS model_signature,
                to_data_month AS data_month
            FROM \`{table}\`
            WHERE state_key = @state_key AND event_id = @event_id
              AND event_type = 'PROMOTION'""",
        job_config=job_config,
    ).result())
    if len(rows) != 1:
        raise RuntimeError(
            "Target approval event not found or ambiguous. Only historical "
            "PROMOTION events are eligible; older approvals require a new "
            "history-backed promotion before they can be rolled back to."
        )
    return dict(rows[0].items())


def list_approvals(bq, ledger_table, state_key, bigquery):
    from google.api_core.exceptions import NotFound

    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("state_key", "STRING", state_key),
    ])
    try:
        rows = list(bq.query(
            f"""SELECT event_id, event_time, to_run_id, to_model_type,
                    to_model_resource
                FROM `{ledger_table}`
                WHERE state_key = @state_key AND event_type = 'PROMOTION'
                ORDER BY event_time DESC LIMIT 50""",
            job_config=job_config,
        ).result())
    except NotFound:
        print("No approval ledger yet. It is created on the next promotion.")
        return

    if not rows:
        print("No historical PROMOTION events recorded for this state key.")
    for row in rows:
        print(dict(row.items()))


def preflight_vertex_model(config, target, features):
    # This is the actual registered artifact and serving contract, not a
    # mutable production-path copy. Do not use a candidate artifact path.
    from google.cloud import aiplatform, storage
    from src.config.feature_contract import validate_registered_features

    project, region = config["project_id"], config["region"]
    validate_resource(target["model_resource"], project, region)
    aiplatform.init(project=project, location=region)
    model = aiplatform.Model(model_name=target["model_resource"])
    if model.resource_name != target["model_resource"]:
        raise RuntimeError("Vertex resource does not match historical approval")
    spec = model.gca_resource.container_spec
    expected_image = config["model"][target["model_type"]]["serving_container_image_uri"].strip()
    if spec.image_uri != expected_image:
        raise RuntimeError("Registered Vertex serving image differs from approved YAML")
    env = {item.name: item.value for item in spec.env}
    validate_registered_features(features, env.get("FEATURE_COLUMNS_JSON"))
    artifact_uri = model.gca_resource.artifact_uri.rstrip("/")
    if not artifact_uri.startswith("gs://"):
        raise RuntimeError("Registered model does not reference a GCS artifact")
    uri = artifact_uri[5:]
    bucket_name, sep, prefix = uri.partition("/")
    if not sep or not bucket_name or not prefix:
        raise RuntimeError("Model artifact location is invalid")
    filename = "model.joblib" if target["model_type"] == "random_forest" else "model.bst"
    if not storage.Client(project=project).bucket(bucket_name).blob(
        f"{prefix}/{filename}"
    ).exists():
        raise RuntimeError("Historically approved model artifact is unavailable")
    return {"artifact_uri": artifact_uri, "serving_image": spec.image_uri}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", help="Approved historical YAML config file")
    parser.add_argument("--state-key", default="provider_fraud_model")
    parser.add_argument("--list", action="store_true", help="List historically approved targets")
    parser.add_argument("--target-event-id", help="Exact prior PROMOTION event UUID")
    parser.add_argument("--reason", help="Required incident/recovery rationale")
    parser.add_argument("--change-ticket", help="Required change/incident tracking ID")
    parser.add_argument("--actor", default=getpass.getuser())
    parser.add_argument("--execute", action="store_true", help="Actually switch active state")
    parser.add_argument("--confirm-resource", help="Exact full Vertex resource name to confirm")
    args = parser.parse_args()

    config = load_config(args.config_path)
    state_table = config["monitoring"]["training_state_table"]
    ledger_table = ledger_table_from_state(state_table)
    from google.cloud import bigquery
    bq = bigquery.Client(project=config["project_id"])

    if args.list:
        list_approvals(bq, ledger_table, args.state_key, bigquery)
        return
    if not args.target_event_id:
        parser.error("--target-event-id is required unless --list")
    active = load_active(bq, state_table, args.state_key, bigquery)
    target = load_target(bq, ledger_table, args.state_key, args.target_event_id, bigquery)
    validate_approval_target(target, active, config["project_id"], config["region"])
    features = check_configuration(config, target)
    validated = preflight_vertex_model(config, target, features)

    print("\nGOVERNED MODEL ROLLBACK PREFLIGHT")
    print(f"Project: {config['project_id']}")
    print(f"Current approved model: {active['model_resource']}")
    print(f"Current approved run: {active['run_id']}")
    print(f"Target approved model: {target['model_resource']}")
    print(f"Target approved run: {target['run_id']}")
    print(f"Artifact: {validated['artifact_uri']}")
    print(f"Serving image: {validated['serving_image']}")
    print(f"Operator: {args.actor}")
    print(f"Reason: {args.reason or '(dry run)'}")
    print(f"Change ticket: {args.change_ticket or '(dry run)'}")
    if not args.execute:
        print("DRY RUN: no active-model state or audit event was changed")
        return

    if not args.reason or not args.reason.strip():
        parser.error("--reason is required with --execute")
    if not args.change_ticket or not args.change_ticket.strip():
        parser.error("--change-ticket is required with --execute")
    if args.confirm_resource != target["model_resource"]:
        parser.error("--confirm-resource must exactly match the target Vertex resource")
    answer = input("Type ROLLBACK to apply the atomic state change: ").strip()
    if answer != "ROLLBACK":
        print("Cancelled; active model unchanged")
        return

    params = {
        "event_id": str(uuid.uuid4()),
        "state_key": args.state_key,
        "actor": args.actor,
        "reason": f"{args.change_ticket.strip()}: {args.reason.strip()}",
        "current_resource": active["model_resource"],
        "current_run_id": active["run_id"],
        "target_event_id": target["event_id"],
        "target_resource": target["model_resource"],
        "target_run_id": target["run_id"],
        "target_model_type": target["model_type"],
        "target_feature_signature": target["feature_signature"],
        "target_model_signature": target["model_signature"],
        "target_data_month": target["data_month"],
    }
    types = {
        "target_data_month": "DATE",
    }
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter(key, types.get(key, "STRING"), value)
        for key, value in params.items()
    ])
    bq.query(rollback_sql(state_table, ledger_table), job_config=job_config).result()
    print("ROLLBACK COMMITTED: active model and audit history updated atomically")
    print("NEXT: run a controlled batch smoke check with --model-id "
          + target["model_resource"].rsplit("/", 1)[-1])
    print("Do not resume scheduled scoring until health checks are reviewed.")


if __name__ == "__main__":
    main()
