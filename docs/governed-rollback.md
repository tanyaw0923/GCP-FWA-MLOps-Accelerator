# Governed model rollback: restore previous production version

The workflow supports **Model A (production) → promote Model B →
governed rollback → Model A (production again)** without retraining A.
This covers **Vertex AI batch scoring**, which reads the approved model
resource and approved experiment run from BigQuery at each invocation.

This is an explicitly authorized, **dry-run-by-default** rollback.
It does not switch Vertex endpoints, cancel in-flight batch jobs, recover
deleted models, or automatically execute smoke tests. Do not use the
command to bypass incident approval or health verification.

## Release and artifact lifecycle

1. Training creates a new unique run ID and saves a candidate artifact in
   `gs://BUCKET/models/ALGORITHM/candidate/RUN_ID/model.joblib`
   (XGBoost uses `model.bst`). This preserves older Vertex Model Registry
   model artifacts from later training runs. Do not overwrite or delete
   these historical paths, registered models, or serving images.
2. The operator manually promotes B. Promotion also copies its model
   into a versioned production snapshot under
   `models/ALGORITHM/production/RUN_ID/`.
   It **does not** overwrite A's snapshot.
3. A single BigQuery transaction makes B active **and** records a
   `PROMOTION` event containing the *complete previous approved A
   identity*: Vertex resource, run, algorithm, feature signature,
   model signature and data month. Thus A can be restored even if A was
   approved before the audit ledger was created, provided its metadata
   is complete and its registered model/artifact remain available.
4. Governed batch inference (the `run_batch_prediction.py` CLI)
   resolves the active approved **model resource and associated run ID**.
   Both `--model-id` and `--run-id` are optional, but explicitly supplied
   values must match the approved state. It uses the registered Vertex
   `FEATURE_COLUMNS_JSON` to select features in the exact historical
   order rather than assuming the current candidate's YAML matches.
5. If rollback is needed, use `--previous`: select the previous A
   from the most recent recorded promotion/rollback transition that
   leads to B. The CLI validates the saved Vertex registry object,
   feature contract and GCS artifact before any state change.
6. Confirm with an incident ticket, reason, target resource and the
   typed prompt. One BigQuery transaction makes A active again and
   records a full `ROLLBACK` event. The next governed batch scoring
   run defaults to A and its historical approved run ID.

## Prerequisites

- PR deployed; at least one **new** promotion performed after deployment.
  Existing rows in `model_training_state` remain intact and become the
  previous-version snapshot when the next model is promoted.
- The previous approved version has non-null
  `approved_model_resource`, `approved_run_id`,
  `approved_model_type`, `approved_feature_signature`,
  `approved_model_signature` and retained model artifacts.
- The candidate and previous versions have their Vertex registry
  objects, serving images and GCS artifacts available. Older candidates
  from before per-run paths were introduced may have shared mutable
  artifacts; **do not trust them for rollback without verifying model
  provenance and artifact integrity**.
- Operators have read access to Vertex Registry and GCS and authorized
  BigQuery DML permissions. Enforce IAM separation and independent
  change approval outside this code. The user-supplied `--actor` is a
  log label and does not authenticate identity.
- Pause scheduled scoring and independently authorize the rollback.

## 1. Inspect recent approvals

```bash
python -m src.models.rollback_approved_model --list
```

This lists promotion events. The immediately previous model is captured
in the latest transition leading to the currently approved version.

## 2. Preview rollback to previous production model

```bash
python -m src.models.rollback_approved_model --previous
```

This is **read-only**: it checks active state, finds previous approved
model A, validates its registered model, original artifact and feature
contract, and prints the B → A change. It does not update BigQuery.

For additional historical config verification, pass `--config-path`
with the historical YAML for A; this checks saved feature and model
configuration signatures and serving-image URI. This parameter does not
deploy or modify any configuration.

## 3. Execute after authorization

```bash
python -m src.models.rollback_approved_model \
  --previous \
  --reason "B degraded precision on reviewed claims" \
  --change-ticket "INC-12345" \
  --actor "authorized-operator" \
  --confirm-resource "projects/PROJECT_NUMBER/locations/us-central1/models/MODEL_A_ID" \
  --execute
```

Use the exact full Model A resource printed during dry-run. Then type
`ROLLBACK` when prompted. If a concurrent actor changes the active
model or a pending candidate appears, the transaction aborts.

The manual `--target-event-id` alternative continues to support
restoring another historical promotion (with historical config
verification), but `--previous` is the standard immediate rollback.

## 4. Verify approved state and audit trail

```sql
SELECT model_name, approved_model_resource, approved_run_id,
       approved_model_type, approved_feature_signature,
       approved_model_signature
FROM `fwa-mlops-accelerator-demo.fraud_monitoring.model_training_state`
WHERE model_name = 'provider_fraud_model';

SELECT event_time, event_type, actor, reason,
       from_model_resource, from_run_id, to_model_resource, to_run_id
FROM `fwa-mlops-accelerator-demo.fraud_monitoring.model_approval_events`
WHERE state_key = 'provider_fraud_model'
ORDER BY event_time DESC LIMIT 10;
```

Use your project ID if testing in an isolated project. Confirm the
approved resource and run are A, and a new ROLLBACK audit event exists.

## 5. Verify actual production model selection

```bash
python src/scoring/run_batch_prediction.py --machine-type n1-standard-2
```

**This command is billable and writes prediction outputs.** Run it
only against controlled test data and after inspecting the active
configuration, permissions, and approved model. The script automatically
loads A's exact Vertex resource and approved experiment run, then selects
A's registered feature list. Confirm model ID and run ID in logs and
the output lineage, and review sample predictions and monitoring health.

Only after review should operators resume production scheduling. No
automatic post-change smoke check is part of this PR. Batch jobs already
submitted before rollback will not change models retroactively.

## Safety limitations

- This provides **governed batch-model rollback**, not Vertex endpoint
  rollback, training pipeline reversion, or automatic deployment
  rollback. Old feature transformations and source columns still need
  to be compatible with A.
- Immutable artifact retention is an operational requirement: versioned
  paths prevent normal overwrites but cannot prevent manual deletion,
  GCS replacement or mutable Docker tag updates. Production systems
  should pin image digests, enforce retention and restricted write IAM.
- The SQL release/audit operation is atomic, but artifact copy
  takes place before the transaction. Failed approval may leave an
  unreferenced versioned copy for later cleanup; never treat it as
  approved merely because the copy exists.
- Unit tests and CI cannot prove live BigQuery transaction semantics
  or model compatibility. Validate the full A → B → A workflow in an
  isolated GCP environment before activating this in production.
