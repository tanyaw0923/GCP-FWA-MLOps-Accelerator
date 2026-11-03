# Production readiness and roadmap

> Detailed documentation extracted from the original project README. Run commands from the repository root unless otherwise stated.

## Production Hardening Roadmap

This repository demonstrates an end-to-end batch MLOps lifecycle, but it is intentionally not positioned as a complete enterprise production platform. A larger production implementation would add the following controls.

### 1. Governed Rollback and Remaining Production Gaps

The repository includes **offline-only, unintegrated rollback prototype code** and regression tests. The existing production promotion and scoring entry points are unchanged. The rollback feature is not enabled or live-tested; see [governed rollback staging notes](governed-rollback.md).

A stronger rollback design would include:

- retain immutable versions of previously approved production artifacts
- store the exact model version, feature version, dataset version, container image version, and Git commit for each production release
- allow operators to mark a prior approved model as the active production version
- restore both the model artifact and its associated serving / scoring configuration
- verify rollback health with post-deployment validation before traffic or batch scoring resumes
- preserve the rollback event in the model-governance audit trail

Conceptually:

```text
Production model v3
        |
        | monitoring / operational issue
        v
Rollback decision
        |
        v
Previously approved model v2
        |
        v
Restore artifact + config + lineage
        |
        v
Validation
        |
        v
Resume production scoring
```

### 2. CI Coverage and Deployment Gaps

**GitHub Actions CI is implemented.** The repository runs four workflows on pull requests targeting `main` and on pushes to `main`; each also supports manual execution:

| Workflow | Current automated check |
| --- | --- |
| [Unit Tests](.github/workflows/unit-tests.yml) | Runs `python -m pytest tests/unit -v`, including retraining decisions, promotion rules, configuration contracts, monitoring checks, and rolling windows |
| [KFP Pipeline Compile](.github/workflows/kfp-pipeline-compile.yml) | Compiles the pipeline and verifies that the generated YAML exists |
| [Docker Build](.github/workflows/docker-build.yml) | Builds the Random Forest, XGBoost, and performance-monitoring Docker images without publishing or deploying them |
| [CodeQL](.github/workflows/codeql.yml) | Performs Python security analysis; also runs on a weekly schedule |

See the [GitHub Actions dashboard](https://github.com/tanyaw0923/GCP-FWA-MLOps-Accelerator/actions) for current results. The `tests/` directory includes unit coverage and a separate container-dependency integration check; read [tests/README.md](tests/README.md) for execution instructions.

**Remaining production-hardening gaps:** These workflows establish automated CI checks, but they do not constitute a complete deployment pipeline. Live GCP/BigQuery and SMTP integration tests, model-training and batch-scoring smoke tests, authenticated Artifact Registry dependency checks, and deployment validation in a non-production environment are not yet part of these workflows. Repository branch-protection settings must also require the relevant checks before merging if strict enforcement is desired; workflow triggers alone do not guarantee merge blocking.

A more complete production promotion path would be:

```text
Pull request
     |
     v
GitHub Actions CI
  unit tests / KFP compile / Docker build / CodeQL
     |
     v
Required checks pass (when branch protection is configured)
     |
     v
Merge to main
     |
     v
Non-production integration and deployment validation (future)
     |
     v
Governed model release and manual production approval
```

### 3. Large-Scale Feature Governance

The current Feature Store layer demonstrates timestamped provider history, reusable feature definitions, and long-term / short-term feature refreshes. At enterprise scale, feature governance would need stronger controls.

A larger implementation would typically add:

- centralized ownership and stewardship for each feature
- feature descriptions, business definitions, and approved use cases
- versioned feature definitions
- lineage from raw source to derived feature
- freshness and quality SLAs
- deprecation policies
- access controls by team / domain
- feature discovery and reuse across multiple models
- automated validation for training-serving consistency
- point-in-time correctness checks
- audit history for changes to feature logic

This prevents multiple teams from independently creating slightly different versions of the same business concept and improves reproducibility across models.

### 4. IAM and Role-Based Access Control

The demo focuses on the ML lifecycle rather than enterprise security administration. A production deployment should use least-privilege IAM and clearly separate responsibilities between data scientists, platform engineers, investigators, and model approvers.

Example responsibilities could be separated as follows:

```text
Data Scientist
  - read approved training data
  - create experiments
  - submit candidate training jobs
  - cannot directly promote to production

ML Platform / Service Account
  - execute pipelines
  - read / write required BigQuery tables
  - access model artifacts
  - register candidate models

Model Approver
  - review candidate metrics
  - approve / reject production promotion

Fraud Investigator
  - access investigation queue
  - write investigation outcomes
  - no model-training permissions

Production Operator
  - manage deployment / rollback
  - restricted access to production resources
```

Additional production controls would include:

- dedicated service accounts per workload
- least-privilege BigQuery dataset permissions
- restricted Artifact Registry and Cloud Storage access
- Secret Manager access only for required services
- separation of development and production projects
- audit logging for model registration, approval, promotion, and rollback

These capabilities would strengthen the platform for enterprise deployment without changing the core design principle of the accelerator: model-specific data science can evolve while the governed MLOps workflow remains reusable.
