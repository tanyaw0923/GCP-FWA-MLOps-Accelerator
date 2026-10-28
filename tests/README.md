# Tests

This directory contains automated tests for the FWA MLOps Accelerator.

- `integration/` — integration and reproducibility checks that may require Docker, Artifact Registry access, or other external services.
- `unit/` — pure unit tests for decision logic and configuration contracts. This folder will be added as unit-test coverage is expanded.

Integration tests are intentionally kept separate from unit tests so lightweight CI checks can run without requiring GCP credentials.

<img width="1356" height="602" alt="image" src="https://github.com/user-attachments/assets/debaf745-07f8-4846-bfce-926bc73daa5f" />

## List of Items being tested
```
Retraining decision
├── no trigger
├── new data
├── feature-signature change
├── model-signature change
├── data-quality alert
├── feature drift
├── prediction drift
├── performance degradation
└── bootstrap state

Promotion gate
├── passes all criteria
├── fails one criterion
├── exact threshold
├── bootstrap promotion
└── missing metrics

Rolling windows
├── normal window
├── year boundary
└── leap year

Configuration
├── RF contract
├── XGBoost contract
├── no legacy feature fallback
├── empty features rejected
└── missing model config rejected
```
