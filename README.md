# GCP FWA MLOps Accelerator

[![Unit Tests](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/unit-tests.yml)
[![KFP Pipeline Compile](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/kfp-pipeline-compile.yml/badge.svg)](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/kfp-pipeline-compile.yml)
[![CodeQL](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/codeql.yml/badge.svg)](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/codeql.yml)
[![Docker Build](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/docker-build.yml/badge.svg)](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/docker-build.yml)
[![Dependency Security](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/dependency-security.yml/badge.svg)](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions/workflows/dependency-security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/blob/main/LICENSE)

A reusable Google Cloud MLOps framework for provider-level healthcare fraud, waste, and abuse (FWA) modeling.\n\n> Governed rollback source code and offline tests are included as **unintegrated, deferred work**. Existing promotion, training, and scoring behavior remains unchanged. See [rollback staging notes](docs/governed-rollback.md).

This project is designed as a shared ML accelerator for a team of data scientists. Data scientists can experiment with new data, features, model configurations, and algorithms while reusing the same governed workflow for retraining decisions, feature refresh, training, evaluation, experiment tracking, candidate registration, approval, and production promotion.

#### ▶▶▶Click the image below to watch the video on YouTube◀◀◀
[![Watch the demo](https://img.youtube.com/vi/9gN_HCLjT6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=9gN_HCLjT6k)

## Business Goal

**Accelerate provider-level fraud detection experiments without rebuilding the ML infrastructure for every model.**

This reusable Google Cloud MLOps accelerator separates **shared platform responsibilities** from **model-specific data science**. A team can change features, model algorithms, and hyperparameters while reusing governed training, evaluation, approval, batch inference, and monitoring workflows.

The project is a runnable technical demonstration, not a claim of a fully hardened enterprise deployment.

## Architecture

![GCP FWA MLOps accelerator architecture](https://github.com/user-attachments/assets/f3647476-e04e-44e6-b4d4-c84a5b64d422)

The workflow uses **BigQuery** for claims, provider features, experiment records, predictions, and monitoring; **Vertex AI Pipelines** for orchestration and training; **Vertex AI Model Registry** for model lifecycle; **Cloud Build / Artifact Registry** for reproducible containers; and **Cloud Run Jobs** for scheduled performance monitoring.

```text
Claims / provider features (BigQuery)
              |
              v
Retraining decision -- no trigger --> stop
              |
              v
Train candidate (Random Forest / XGBoost)
              |
              v
Evaluate + track experiment
              |
              v
Promotion gate --> register candidate --> manual approval
                                            |
                                            v
                                      Batch prediction
                                            |
                                            v
                             Investigation queue + feedback
                                            |
                                            v
                             Model and data monitoring
                                            |
                                            +--> future retraining decision
```

For the detailed architecture, two-speed Feature Store design, lineage, monitoring schema, and observability, see [Architecture and ML lifecycle](docs/architecture.md).

## Key Capabilities

| Capability | Implementation |
| --- | --- |
| **Configurable modeling** | Switch between Random Forest and XGBoost; edit features and hyperparameters via `src/config/pipeline_config.yaml` |
| **Reusable feature engineering** | Provider-level features with rolling training/testing windows and separate historical feature snapshots |
| **Experiment tracking** | Store run metadata, data/feature signatures, and model evaluation metrics in BigQuery |
| **Governed promotion** | Candidate evaluation gate, model registration, and manual approval before production promotion |
| **Batch inference** | Feature-contract validation and provider-level prediction lineage |
| **Fraud operations** | Capacity-based investigation queue, investigator feedback, precision@K / recall@K, and fraud-value analysis |
| **Monitoring** | Data-quality checks, feature drift, fraud-score distribution drift, feedback-driven precision/recall/F1, notifications, and retraining signals |

### Three experimentation scenarios

1. **New data:** A new monthly claims period advances the rolling train/test window.
2. **New features:** A changed feature configuration produces a new feature signature and candidate training run.
3. **New model:** Change the selected model family, such as Random Forest to XGBoost, while keeping the shared pipeline.

See [Experimentation guide](docs/experiment-guide.md) for configuration changes, model extensions, and container versioning.

## Demo and Quick Start

The full execution guide covers container builds, pipeline compilation, training, manual model promotion, batch scoring, investigation feedback, and monitoring.


The full end-to-end demo uses configured GCP resources and permissions. Follow the [13-step demo runbook](docs/demo-runbook.md) for cloud execution; compilation alone does not launch a Vertex AI job.

## Testing and CI

[Unit Tests](.github/workflows/unit-tests.yml), [KFP Pipeline Compile](.github/workflows/kfp-pipeline-compile.yml), [Docker Build](.github/workflows/docker-build.yml), [CodeQL](.github/workflows/codeql.yml), [Test Coverage](.github/workflows/test-coverage.yml), [Dependency Security](.github/workflows/dependency-security.yml), [Container Security](.github/workflows/container-security.yml), and [Secret Scanning](.github/workflows/secret-scanning.yml) run through GitHub Actions for pull requests targeting `main` and pushes to `main`. CodeQL also runs on a schedule.

```bash
python -m pytest tests/unit -v
```

These checks cover code and build validation; live BigQuery, SMTP, and cloud deployment validation are separate. See [Testing reference](tests/README.md) and [Code reference](docs/code-reference.md).

## Documentation

| Guide | What's inside |
| --- | --- |
| [Architecture and ML lifecycle](docs/architecture.md) | Cloud services, Feature Store, retraining, model governance, inference, monitoring, observability |
| [Experimentation and configuration](docs/experiment-guide.md) | Reusing the accelerator, adding models/features, container versioning |
| [End-to-end demo runbook](docs/demo-runbook.md) | Demo datasets, full execution commands, investigations, troubleshooting |
| [Code reference](docs/code-reference.md) | Repository map, module walkthrough, suggested reading order, testing |
| [Testing guide](tests/README.md) | Unit, SQL regression, and environment-dependent test instructions |
| [Isolated cloud smoke test](docs/cloud-smoke-test.md) | Opt-in synthetic BigQuery → Vertex Model Registry → batch scoring and lineage check |
| [Governed model rollback](docs/governed-rollback.md) | Audited, dry-run-first rollback to a previously promoted model |
| [Production readiness](docs/production-readiness.md) | Known limitations and enterprise hardening roadmap |

## Production Considerations

The repository demonstrates **batch-focused MLOps with manual production model approval**. Additional production controls would include automated rollback, stronger IAM segmentation, continuous integration tests against isolated GCP resources, and automated non-production deployment validation.

For details and current gaps, see [Production readiness](docs/production-readiness.md).

> **Design principle:** Standardize the ML lifecycle, not the data science. Teams can change data, features, and algorithms while reusing a governed workflow.
