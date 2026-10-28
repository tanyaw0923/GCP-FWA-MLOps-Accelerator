# Tests

This directory contains automated tests for the FWA MLOps Accelerator.

- `integration/` — integration and reproducibility checks that may require Docker, Artifact Registry access, or other external services.
- `unit/` — pure unit tests for decision logic and configuration contracts. This folder will be added as unit-test coverage is expanded.

Integration tests are intentionally kept separate from unit tests so lightweight CI checks can run without requiring GCP credentials.
