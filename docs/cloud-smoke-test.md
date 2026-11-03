# Isolated Vertex AI cloud smoke test

This is an **explicitly opt-in, billable integration test**, not part of the normal PR checks. It verifies the real GCP interfaces:

1. Fit a **tiny synthetic Random Forest locally** (2 features, 4 labeled examples).
2. Create a **unique BigQuery dataset/table** containing 3 synthetic providers.
3. Create an isolated **Cloud Storage bucket**, upload the model artifact.
4. Register a Vertex AI model using this project's **existing Random Forest serving application**, rebuilt into the **test project**.
5. Run a real **Vertex AI BatchPredictionJob** using BigQuery input and output.
6. Verify output count, numeric probability bounds, no duplicate IDs, and provider lineage.
7. **Attempt cleanup** of the Vertex job/model, BigQuery dataset, and storage bucket, even after an error.

The smoke test does **not** run the full Kubeflow/Vertex training pipeline, promotion gate, investigator feedback, SMTP, or scheduled production monitoring. Those require separate environment-specific tests. Local synthetic RF training is deliberately inexpensive; **Vertex model registration and batch prediction still incur cloud costs** and may fail if machine capacity is temporarily unavailable.

## Safety / isolation

- Use a **dedicated disposable GCP project**, not the repository's demo project.
- The script **refuses** the configured demo project ID (`fwa-mlops-accelerator-demo`) and requires `--acknowledge-cost`.
- The serving image must belong to the **same test project and region**.
- Unique, prefixed resources are created at every run. The script does **not import pipeline_config.yaml** and never reads or writes production fraud tables.
- The test **does not** delete the Artifact Registry repository or the prebuilt serving image (those are created explicitly in setup).
- Cleanup is best-effort. If a cloud operation refuses deletion (for example a still-cancelling batch job), review resources manually and remove them to avoid ongoing charges.
- Do **not** pass `--keep-resources` unless debugging a failure and prepared to pay for leftovers.

## One-time setup in Cloud Shell

Create/select a separate test project with billing enabled; enable the relevant APIs, and use credentials authorized for BigQuery, Cloud Storage, Vertex AI, Cloud Build/Artifact Registry. Check the Vertex AI service-agent access to the model artifacts and serving image in your dedicated project.

```bash
export SMOKE_PROJECT="YOUR-DEDICATED-TEST-PROJECT"
export REGION="us-central1"
export AR_REPO="fwa-smoke"

gcloud config set project "$SMOKE_PROJECT"
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="$SMOKE_PROJECT"

gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$SMOKE_PROJECT"

gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

Build and push **the existing Random Forest serving Dockerfile** into the isolated project; do not use your production Artifact Registry image. Run from the repository root:

```bash
export SMOKE_IMAGE="$REGION-docker.pkg.dev/$SMOKE_PROJECT/$AR_REPO/random-forest-serving:smoke"

docker build \
  -f docker/random-forest-serving/Dockerfile \
  -t "$SMOKE_IMAGE" .
docker push "$SMOKE_IMAGE"

python -m pip install -r tests/integration/requirements-cloud-smoke.txt
```

If Cloud Shell uses a different Application Default Credential identity, ensure it has access to **the test project**.

## Run the isolated end-to-end smoke test

```bash
python tests/integration/isolated_vertex_smoke.py \
  --project "$SMOKE_PROJECT" \
  --region "$REGION" \
  --serving-image "$SMOKE_IMAGE" \
  --machine-type n1-standard-2 \
  --timeout-minutes 40 \
  --acknowledge-cost
```

Expected result: `[smoke] PASS: model registered, 3 batch predictions and lineage verified`.

A passing run demonstrates **BigQuery data creation -> model artifact -> Vertex registry -> Vertex BatchPredictionJob -> BigQuery predictions and provider lineage**, against the isolated test project.

For a failed run, consult the Vertex BatchPredictionJob error, Cloud Logging for the container, and any `[cleanup]` warnings. Temporary machine unavailability (`code: 14`) is an infrastructure failure; retry later or use a supported alternate machine type. Do not mistake it for a feature-contract failure.

## What runs automatically in CI?

The test file `tests/unit/test_cloud_smoke_contract.py` runs in ordinary PR CI **without GCP credentials or costs**. It checks the isolation and opt-in validation, as well as output count, probabilities, duplicate IDs, and provider lineage, using in-memory rows. The actual cloud integration test runs **only with the explicit Cloud Shell command above**.

## Cleanup verification

The script attempts deletion automatically. If it warns about a cleanup failure, check the dedicated test project:

```bash
gcloud ai models list --region="$REGION" --project="$SMOKE_PROJECT"
gcloud ai batch-prediction-jobs list --region="$REGION" --project="$SMOKE_PROJECT"
bq ls --project_id="$SMOKE_PROJECT"
gcloud storage buckets list --project="$SMOKE_PROJECT"
```

Manually remove only leftover `fwa-smoke` / `fwa_smoke_` resources created by this test after confirming ownership. The Artifact Registry image and repository are persistent until you explicitly remove them.
