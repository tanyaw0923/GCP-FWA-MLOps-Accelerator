# Governed model rollback — operator runbook

This is an **operator-initiated, dry-run-first** rollback of the approved
model pointer in BigQuery. It does not automatically switch Vertex endpoints,
cancel existing jobs, rewrite GCS artifacts, or change deployed configuration.

## Governance and safeguards

- Every new manual promotion appends a PROMOTION event in
  fraud_monitoring.model_approval_events, in the same BigQuery transaction
  that updates the active approved model.
- A rollback can target **only an earlier PROMOTION event** with an exact
  event ID, model resource, model type, run ID and feature/model signatures.
  Older approvals predating this new ledger **are not backfilled**. Perform
  at least two history-backed promotions to have a prior eligible target.
- Preflight checks the historical YAML signature, registered serving-image,
  registered feature contract, GCS model artifact, current active state,
  and pending-candidate status. This prevents switching to an unverified
  old model. The operator supplies the historical YAML using --config-path.
- Execution requires reason, change ticket, exact Vertex resource confirmation,
  and interactive typing of ROLLBACK.
- The BigQuery transaction changes the approved-model pointer and appends
  a ROLLBACK event atomically; a concurrent state change blocks the operation.
- Batch scoring now refuses a --model-id other than the active approved model.
- The ledger is append-only by **application behavior**, not tamper-proof.
  Configure IAM and audit logging so only authorized service accounts can
  update approved state and ledger. The --actor label is self-reported, not
  authenticated identity; obtain independent change approval externally.
- Preserve immutable versioned artifacts, container image digests, and config
  history for stronger enterprise recovery. Existing mutable tags and model
  artifacts can still be changed/deleted outside this tooling.

## Preparation

1. Merge and deploy the new promotion and batch-scoring code.
2. Approve new candidates via the normal promotion command. New promotions
   initialize and populate the approval-event ledger.
3. Have an authorized approver approve the rollback change ticket, stop
   scheduled scoring, preserve the incident evidence and identify the
   historically approved model to restore.

## List historical approved releases

~~~bash
python -m src.models.rollback_approved_model --list
~~~

Find the event_id of an eligible PROMOTION event. If no rows appear,
the ledger has not yet recorded eligible past releases.

## Dry-run (no state changes)

~~~bash
python -m src.models.rollback_approved_model \
  --config-path path/to/historical-approved-config.yaml \
  --target-event-id "PAST_PROMOTION_EVENT_UUID"
~~~

This validates the target in the Vertex Model Registry and registered
feature/serving contract, confirms its GCS artifact exists, and checks
that the historical configuration reproduces both stored signatures.
It does not perform any BigQuery UPDATE or INSERT.

## Execute (requires interactive confirmation)

~~~bash
python -m src.models.rollback_approved_model \
  --config-path path/to/historical-approved-config.yaml \
  --target-event-id "PAST_PROMOTION_EVENT_UUID" \
  --reason "Revert incorrect high-acuity feature release" \
  --change-ticket "INC-12345" \
  --actor "authorized-operator" \
  --confirm-resource "projects/PROJECT_NUMBER/locations/us-central1/models/OLD_MODEL_ID" \
  --execute
~~~

Type ROLLBACK only after independently verifying the displayed
current and target model/run/resource identities and change ticket.

## Post-change verification

~~~sql
SELECT approved_model_resource, approved_run_id, approved_model_type,
       approved_feature_signature, approved_model_signature
FROM `fwa-mlops-accelerator-demo.fraud_monitoring.model_training_state`
WHERE model_name = 'provider_fraud_model';

SELECT event_time, event_type, actor, reason,
       from_model_resource, to_model_resource, from_run_id, to_run_id
FROM `fwa-mlops-accelerator-demo.fraud_monitoring.model_approval_events`
WHERE state_key = 'provider_fraud_model'
ORDER BY event_time DESC LIMIT 10;
~~~

Verify exactly the intended active model was selected and a new ROLLBACK
event recorded. Next, run a controlled batch smoke test, review
prediction probabilities and monitoring health, then explicitly authorize
resumption of scheduled scoring. Post-rollback health checks are **manual**.

IMPORTANT: Supplying --config-path to the rollback command does NOT
update pipeline_config.yaml on deployed services. Align live configuration
with the restored historical release before scoring.

## Limitations

- This is state-pointer rollback, NOT recovery of deleted GCS artifacts,
  Vertex models, endpoint deployments or active batch jobs.
- The tool does not bypass missing history, skip the approval record,
  automatically backfill prior releases, or roll back pending candidates.
- BigQuery IAM must protect state and ledger write access; this tool does
  not provide two-person cryptographic approval.
- On transaction concurrency failure, verify state and ledger before
  repeating the dry-run. Do not manually rewrite approved state.
