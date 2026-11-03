# Testing

The FWA MLOps Accelerator separates local unit tests, BigQuery SQL checks, and environment-dependent integration tests. Run commands from the repository root.

## Unit tests (no GCP credentials required)

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/unit -v
```

Coverage includes:

- `test_retraining_decision.py`: new-data, feature-signature, model-signature, data-quality, drift and performance alerts, bootstrap behavior, and missing monitoring results. Missing results should be reported as `INSUFFICIENT_DATA`, not silently treated as `PASS`. `INSUFFICIENT_DATA` alone does not trigger retraining.
- `test_promotion_gate.py`: candidate eligibility, thresholds, missing metrics and bootstrap promotion.
- `test_rolling_window.py`: ordinary periods, year boundaries and leap years.
- `test_config_contracts.py`: Random Forest and XGBoost configuration contracts and invalid configurations.
- `test_feature_contract_behavior.py`: executes the same validation used in training orchestration and batch scoring; checks model-specific feature selection, actual YAML changes, duplicates/unsafe columns, registered model feature order, source schema, and batch input ordering.
- `test_monitoring_sql_generation.py`: generated monitoring SQL and configured feature handling.
- `test_monitoring_schema_contract.py`: prevents reintroduction of the retired Random Forest monitor and verifies the active prediction-drift writer/reader column contract (`check_date`, `status`).
- `test_performance_monitoring_test_mode.py`: static safeguards for the opt-in 15-case monitoring test, isolated output table, capped sample and test email prefix. These are **not** a substitute for a live GCP integration test.

## BigQuery SQL regression check

`sql/testing/test_insufficient_investigations.sql` builds temporary synthetic claims and investigation data and checks that 15 completed investigations, below the 20-case threshold, produce `INSUFFICIENT_DATA`. It uses **temporary** tables and does not update production monitoring status.

Run from Cloud Shell (with BigQuery access):

```bash
bq query --use_legacy_sql=false < sql/testing/test_insufficient_investigations.sql
```

## Isolated performance monitoring integration test (GCP)

The actual script can evaluate a capped sample of 15 matched completed investigations from the currently approved model. It reads existing prediction, feedback, and approved-model state, but writes its result to a **separate test dataset**.

1. Inspect the location of `fraud_monitoring` and create `fraud_monitoring_test` in the **same location** (skip creation if it exists):

   ```bash
   bq show --format=prettyjson fwa-mlops-accelerator-demo:fraud_monitoring
   bq mk --dataset --location=US fwa-mlops-accelerator-demo:fraud_monitoring_test
   ```

   Replace `US` with the actual source dataset location.

2. Run from the repository root:

   ```bash
   FWA_MONITOR_TEST_MODE=true python -m src.monitoring.run_model_performance_monitoring
   ```

3. Verify the newest isolated record:

   ```sql
   SELECT evaluation_timestamp, evaluated_cases, monitoring_status,
          alert_reason, precision, recall, f1_score
   FROM `fwa-mlops-accelerator-demo.fraud_monitoring_test.model_performance_degradation`
   ORDER BY evaluation_timestamp DESC LIMIT 1;
   ```

For at least 15 matching completed investigations, expect `evaluated_cases = 15`, `monitoring_status = 'INSUFFICIENT_DATA'`, and null precision/recall/F1. The standard minimum is 20. If email is configured, the script sends a **real** notification with a `[TEST]` subject. If sending fails, inspect the monitoring logs. The 15-case cap is an opt-in **test only**: without `FWA_MONITOR_TEST_MODE=true`, the script uses the normal production monitoring table. Do not point tests at production output or manually overwrite production monitoring records.

## Isolated Vertex AI cloud smoke test (manual, billable)

A synthetic end-to-end test registers a tiny Random Forest model in a **dedicated disposable GCP project**, runs Vertex batch inference on a unique BigQuery dataset, validates provider lineage and output probabilities, and attempts automatic cleanup. It does not use the demo or production datasets. **Not run automatically in GitHub Actions.**

See [docs/cloud-smoke-test.md](../docs/cloud-smoke-test.md) for prerequisites, explicit cost opt-in, commands, and cleanup checks. Offline safety and lineage validation is covered by `test_cloud_smoke_contract.py` in regular unit CI.

## Container dependency integration test

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
bash tests/integration/test_container_dependency_reproducibility.sh
```

This requires Docker and permission to pull the project's Artifact Registry images.

## CI gates

GitHub Actions separately checks unit tests, KFP compilation, Docker build, and CodeQL. Passing CI validates these respective checks; it does **not** prove that live BigQuery, SMTP, or scheduled Cloud Run monitoring has succeeded.
