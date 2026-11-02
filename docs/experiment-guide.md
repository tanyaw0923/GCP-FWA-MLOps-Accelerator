# Experimentation and configuration guide

> Detailed documentation extracted from the original project README. Run commands from the repository root unless otherwise stated.

## Configuration-Driven Experimentation

The primary configuration lives in:

```text
src/config/pipeline_config.yaml
```

The currently configured dataset version is controlled by:

```yaml
data:
  current_data_month: "2026-09-01"
```

The active model is controlled by:

```yaml
model:
  model_type: random_forest
```

For Use Case 3, changing only:

```yaml
model:
  model_type: xgboost
```

causes the launcher to select the corresponding model name, hyperparameters, artifact locations, serving container, and model signature while reusing the rest of the pipeline.

The shared provider feature set currently includes:

```text
high_acuity_pct
pct_99214
pct_99215
referral_rate
eye_procedure_pct
avg_paid_per_claim
high_acuity_oe_ratio
avg_member_provider_distance
```

## Reuse This MLOps Infrastructure for New Experiments

### Minimum changes for a new experiment using an existing model family

For example, if a second fraud experiment still uses Random Forest but targets a different provider pattern, the intended change path is:

```text
1. pipeline_config.yaml
   - experiment_name
   - active features
   - feature SQL / derived feature SQL
   - validation rules
   - table names if the experiment uses separate datasets
   - hyperparameters / artifact paths if needed

2. Source data
   - load the new claims / events into the configured source table

3. Regenerate monitoring SQL
   python -m src.monitoring.generate_monitoring_sql

4. Recompile the KFP pipeline only if pipeline/component code changed
   python -m src.pipeline.model_training_pipeline

5. Run tests
   python -m pytest tests/unit -v

6. Launch the experiment
   python -m src.pipeline.run_model_training_pipeline
```

The design goal is that a normal feature/model experiment should **not** require edits to the orchestration code, serving code, or monitoring generator.

### Adding a completely new model type

Adding a model family beyond `random_forest` and `xgboost` requires more work because model training and serving are algorithm-specific. In addition to adding a new `model.<model_type>` block in `pipeline_config.yaml`, add or update:

```text
src/models/train_<new_model>.py
src/pipeline/components/train_<new_model>_component.py
src/pipeline/model_training_pipeline.py
docker/<new-model>/
docker/<new-model>-serving/
cloudbuild-<new-model>.yaml
cloudbuild-<new-model>-serving.yaml
tests/
```

The new branch in `model_training_pipeline.py` should still reuse the shared components for:

```text
retraining decision
rolling train/test window
feature refresh
experiment tracking
promotion gate
Vertex Model Registry
candidate state
manual approval
monitoring
```

After changing pipeline topology or component definitions, recompile:

```bash
python -m src.pipeline.model_training_pipeline
```

and commit the updated `model_training_pipeline.yaml`.

### Feature changes

When adding a new model feature, update the experiment in this order:

```text
pipeline_config.yaml
  -> feature_sql or derived_feature_sql
  -> model.<model_type>.features
  -> monitoring.feature_validation_rules
```

Then regenerate monitoring SQL:

```bash
python -m src.monitoring.generate_monitoring_sql
```

The generator expands the data-quality null/invalid checks and feature-drift `UNPIVOT` list from the active model feature contract. Batch scoring and the custom serving container also consume that same configured feature list, reducing duplicated hard-coded feature definitions.

### Moving the accelerator to another GCP project

At minimum, update:

```text
src/config/pipeline_config.yaml
cloudbuild-random-forest.yaml
cloudbuild-random-forest-serving.yaml
cloudbuild-xgboost.yaml
cloudbuild-xgboost-serving.yaml
cloudbuild-performance-monitoring.yaml
```

Replace project IDs, bucket names, BigQuery table paths, Artifact Registry image URIs, and Feature Store resource IDs. Then rebuild the required images, regenerate monitoring SQL, recompile the pipeline if needed, and rerun the unit/integration checks before launching the experiment.

## Container Image Versioning

Custom project images use explicit semantic-style versions rather than floating tags.

```text
Random Forest training
random-forest-training:v1.0.0

Random Forest serving
random-forest-serving:v1.0.1

XGBoost training
xgboost-training:v1.0.0

XGBoost serving
xgboost-serving:v1.0.1

Performance monitoring
performance-monitoring:v1.0.0
```

The version is declared as the Cloud Build substitution `_IMAGE_VERSION`, so a new container release can be built without editing the build steps.

For example:

```bash
gcloud builds submit \
  --config cloudbuild-xgboost.yaml \
  --substitutions=_IMAGE_VERSION=v1.1.0 \
  .
```

For model-training images, the image URI is also part of the model signature. Changing the selected training image therefore changes the candidate signature and triggers the governed retraining workflow.

The KFP container component references a pinned image version at compile time, so when a training-image version changes, update the corresponding image version in `pipeline_config.yaml` and the KFP component, then recompile `model_training_pipeline.yaml`.
