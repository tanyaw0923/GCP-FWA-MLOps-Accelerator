# Governed rollback — staged implementation (not active)

> **Status: offline-tested, integration deferred.** This folder contains prototype
> rollback orchestration, approval-history SQL helpers, and offline regression tests.
> It is deliberately **not connected** to the current production promotion,
> training, or batch-scoring entry points. Cloud credits are exhausted, so it has
> not been validated against live GCP resources. **Do not execute
> `rollback_approved_model.py --execute` against production.**

## Scope merged now

- `src/models/approval_ledger.py`: candidate promotion and rollback audit SQL,
  validation, previous-model selection and scoring identity helper functions.
- `src/models/rollback_approved_model.py`: draft dry-run-first operator command,
  intended for integration after cloud test.
- `tests/unit/test_governed_rollback.py`: offline SQL, safety and CLI regression
  tests. These do not assert that current production scoring uses rollback.
- Existing production entry points **remain exactly as on main**:
  `src/models/promote_model_candidate.py`,
  `src/pipeline/run_model_training_pipeline.py`, and
  `src/scoring/run_batch_prediction.py`.

No BigQuery approval ledger will be created by the existing promotion command.
The new `--previous` command will not work against an empty ledger.

## To activate when budget is available

1. Integrate full immutable, run-versioned candidate and approved artifacts.
   Do not repoint existing registered models at mutable GCS objects.
2. Wire atomic promotion ledger creation/previous-version snapshots into the
   production promotion command, including migrations and guardrails.
3. Wire approved resource, approved run ID and registered feature contract into
   production batch scoring, with explicit-override checks and schema matching.
4. Validate actual BigQuery transaction SQL, table schemas, registry metadata,
   artifact retention and consistency with live model signatures.
5. Run an isolated A → B → A test in a disposable project: register A and B,
   promote both, rollback to A, score a few synthetic records with A, verify
   output lineage and audit history. Include negative tests and resource cleanup.
6. Independently authorize any production enablement after the isolated
   smoke test passes.

## Tests without GCP credit

```bash
python -m pytest tests/unit/test_governed_rollback.py -v
python -m pytest tests/unit -v
```

The test workflows do not promote, rollback, or deploy GCP resources.
Do not confuse green CI with validated live rollback behavior.
