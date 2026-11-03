"""Governed model approval ledger and atomic rollback primitives.

The ledger is append-only by application convention; enforce BigQuery
write restrictions through IAM in production (application code cannot make
a BigQuery table tamper-proof).
"""
import re

_TABLE = re.compile(r"^[a-z][a-z0-9-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_RESOURCE = re.compile(r"^projects/([0-9]+|[a-z][a-z0-9-]+)/locations/([a-z0-9-]+)/models/([0-9]+)$")
_MODEL_TYPES = {"random_forest", "xgboost"}


def validate_table_name(table):
    if not isinstance(table, str) or not _TABLE.fullmatch(table):
        raise ValueError("Expected fully qualified project.dataset.table name")
    return table


def ledger_table_from_state(training_state_table):
    state = validate_table_name(training_state_table)
    return state.rsplit(".", 1)[0] + ".model_approval_events"


def validate_resource(resource, project_id, region):
    match = _RESOURCE.fullmatch(resource or "")
    if not match:
        raise ValueError("Expected complete Vertex model resource name")
    # The Vertex API normally returns the project number; the full resource
    # string is checked against the actual model after fetching it.
    if match.group(2) != region:
        raise ValueError("Model is in the wrong region")
    return resource


def validate_approval_target(target, active, project_id, region):
    if target["model_type"] not in _MODEL_TYPES:
        raise ValueError("Unsupported approved model type")
    if not target.get("run_id") or not target.get("model_resource"):
        raise ValueError("Approval history missing model or run identity")
    if not target.get("feature_signature") or not target.get("model_signature"):
        raise ValueError("Approval history missing model signatures")
    if not target.get("training_timestamp"):
        raise ValueError("Approval history missing original training timestamp")
    validate_resource(target["model_resource"], project_id, region)
    if target["model_resource"] == active["model_resource"]:
        raise ValueError("Target is already the active approved model")
    if active.get("candidate_status") == "PENDING_APPROVAL":
        raise ValueError("Resolve pending approval before rollback")
    return target


def ensure_ledger_sql(table):
    table = validate_table_name(table)
    return f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
      event_id STRING NOT NULL,
      state_key STRING NOT NULL,
      event_time TIMESTAMP NOT NULL,
      event_type STRING NOT NULL,
      actor STRING NOT NULL,
      reason STRING NOT NULL,
      from_model_resource STRING,
      from_run_id STRING,
      from_model_type STRING,
      from_feature_signature STRING,
      from_model_signature STRING,
      from_data_month DATE,
      from_training_timestamp TIMESTAMP,
      to_model_resource STRING NOT NULL,
      to_run_id STRING NOT NULL,
      to_model_type STRING NOT NULL,
      to_feature_signature STRING NOT NULL,
      to_model_signature STRING NOT NULL,
      to_data_month DATE,
      to_training_timestamp TIMESTAMP
    )
    """


def ensure_previous_columns_sql(table):
    """Migrate an existing ledger without discarding approval history."""
    table = validate_table_name(table)
    return [
        f"ALTER TABLE `{table}` ADD COLUMN IF NOT EXISTS {name} {type_}"
        for name, type_ in (
            ("from_model_type", "STRING"),
            ("from_feature_signature", "STRING"),
            ("from_model_signature", "STRING"),
            ("from_data_month", "DATE"),
            ("from_training_timestamp", "TIMESTAMP"),
            ("to_training_timestamp", "TIMESTAMP"),
        )
    ]


def select_previous_model(event, current_resource, current_run_id):
    """Resolve the model replaced by the latest committed release transition."""
    if event.get("to_model_resource") != current_resource:
        raise ValueError("Release history does not match the active model")
    if event.get("to_run_id") != current_run_id:
        raise ValueError("Release run ID does not match the active model")
    target = {
        "event_id": event["event_id"],
        "model_resource": event.get("from_model_resource"),
        "run_id": event.get("from_run_id"),
        "model_type": event.get("from_model_type"),
        "feature_signature": event.get("from_feature_signature"),
        "model_signature": event.get("from_model_signature"),
        "data_month": event.get("from_data_month"),
        "training_timestamp": event.get("from_training_timestamp"),
    }
    if not all(target.get(key) for key in (
        "model_resource", "run_id", "model_type",
        "feature_signature", "model_signature", "training_timestamp",
    )):
        raise ValueError(
            "Previous production model lacks a complete approval snapshot; "
            "do not guess a rollback target"
        )
    return target


def resolve_scoring_model(requested, approved_resource, approved_run_id):
    """Choose the active approved version and its associated experiment run."""
    if not approved_resource or not approved_run_id:
        raise ValueError("No complete approved model in training state")
    validate_scoring_approval(requested, approved_resource) if requested else None
    return approved_resource, approved_run_id


def promotion_sql(state_table, ledger_table):
    state_table = validate_table_name(state_table)
    ledger_table = validate_table_name(ledger_table)
    # Append the PROMOTION event *before* updating the state in the same
    # BigQuery transaction. If either statement fails, neither commits.
    return f"""
    BEGIN TRANSACTION;
    ASSERT (
      SELECT COUNT(*) = 1
      FROM `{state_table}`
      WHERE model_name = @state_key
        AND candidate_status = 'PENDING_APPROVAL'
        AND candidate_data_month = @candidate_data_month
        AND candidate_run_id = @candidate_run_id
        AND candidate_model_resource = @candidate_resource
    ) AS 'Pending candidate changed; approval aborted';

    INSERT INTO `{ledger_table}` (
      event_id, state_key, event_time, event_type, actor, reason,
      from_model_resource, from_run_id, from_model_type,
      from_feature_signature, from_model_signature, from_data_month,
      from_training_timestamp, to_model_resource, to_run_id,
      to_model_type, to_feature_signature, to_model_signature,
      to_data_month, to_training_timestamp
    )
    SELECT
      @event_id, model_name, CURRENT_TIMESTAMP(), 'PROMOTION',
      @actor, @reason, approved_model_resource, approved_run_id,
      approved_model_type, approved_feature_signature,
      approved_model_signature, last_training_data_month,
      last_training_timestamp, candidate_model_resource,
      candidate_run_id, candidate_model_type,
      candidate_feature_signature, candidate_model_signature,
      candidate_data_month, CURRENT_TIMESTAMP()
    FROM `{state_table}`
    WHERE model_name = @state_key
      AND candidate_status = 'PENDING_APPROVAL'
      AND candidate_data_month = @candidate_data_month
      AND candidate_run_id = @candidate_run_id
      AND candidate_model_resource = @candidate_resource;

    ASSERT @@row_count = 1 AS 'Approval audit insertion failed';

    UPDATE `{state_table}`
    SET last_training_data_month = candidate_data_month,
        last_training_timestamp = CURRENT_TIMESTAMP(),
        approved_feature_signature = candidate_feature_signature,
        approved_model_signature = candidate_model_signature,
        approved_model_type = candidate_model_type,
        approved_model_resource = candidate_model_resource,
        approved_run_id = candidate_run_id,
        candidate_status = 'APPROVED'
    WHERE model_name = @state_key
      AND candidate_status = 'PENDING_APPROVAL'
      AND candidate_data_month = @candidate_data_month
      AND candidate_run_id = @candidate_run_id
      AND candidate_model_resource = @candidate_resource;

    ASSERT @@row_count = 1 AS 'Approval state update failed';
    COMMIT TRANSACTION;
    """


def rollback_sql(state_table, ledger_table):
    state_table = validate_table_name(state_table)
    ledger_table = validate_table_name(ledger_table)
    # The target must still exist in the promotion ledger at commit time.
    # Optimistic concurrency checks the complete active resource/run pair.
    return f"""
    BEGIN TRANSACTION;
    ASSERT (
      SELECT COUNT(*) = 1
      FROM `{ledger_table}`
      WHERE state_key = @state_key
        AND event_id = @target_event_id
        AND (
          (@target_direction = 'to'
            AND event_type = 'PROMOTION'
            AND to_model_resource = @target_resource
            AND to_run_id = @target_run_id)
          OR
          (@target_direction = 'from'
            AND event_type IN ('PROMOTION', 'ROLLBACK')
            AND from_model_resource = @target_resource
            AND from_run_id = @target_run_id
            AND from_model_type = @target_model_type
            AND from_feature_signature = @target_feature_signature
            AND from_model_signature = @target_model_signature
            AND to_model_resource = @current_resource
            AND to_run_id = @current_run_id)
        )
    ) AS 'Target is not a unique historically promoted model';

    UPDATE `{state_table}`
    SET approved_model_resource = @target_resource,
        approved_run_id = @target_run_id,
        approved_model_type = @target_model_type,
        approved_feature_signature = @target_feature_signature,
        approved_model_signature = @target_model_signature,
        last_training_data_month = @target_data_month,
        last_training_timestamp = @target_training_timestamp,
        candidate_status = 'APPROVED'
    WHERE model_name = @state_key
      AND approved_model_resource = @current_resource
      AND approved_run_id = @current_run_id
      AND approved_model_type = @current_model_type
      AND approved_feature_signature = @current_feature_signature
      AND approved_model_signature = @current_model_signature
      AND COALESCE(candidate_status, '') != 'PENDING_APPROVAL';

    ASSERT @@row_count = 1 AS 'Active model changed or approval pending; rollback aborted';

    INSERT INTO `{ledger_table}` (
      event_id, state_key, event_time, event_type, actor, reason,
      from_model_resource, from_run_id, from_model_type,
      from_feature_signature, from_model_signature, from_data_month,
      from_training_timestamp, to_model_resource, to_run_id,
      to_model_type, to_feature_signature, to_model_signature,
      to_data_month, to_training_timestamp
    ) VALUES (
      @event_id, @state_key, CURRENT_TIMESTAMP(), 'ROLLBACK', @actor, @reason,
      @current_resource, @current_run_id,
      @current_model_type, @current_feature_signature,
      @current_model_signature, @current_data_month,
      @current_training_timestamp, @target_resource, @target_run_id,
      @target_model_type, @target_feature_signature, @target_model_signature,
      @target_data_month, @target_training_timestamp
    );

    ASSERT @@row_count = 1 AS 'Rollback audit insertion failed';
    COMMIT TRANSACTION;
    """


def validate_scoring_approval(requested_model, approved_resource):
    if not approved_resource:
        raise ValueError("No approved model resource; approve a model before scoring")
    # The CLI accepts a numeric model ID or a full resource name.
    model_id = approved_resource.rsplit("/", 1)[-1]
    if requested_model not in {approved_resource, model_id}:
        raise ValueError(
            f"Requested model is not the active approved model: {requested_model}"
        )
    return approved_resource
