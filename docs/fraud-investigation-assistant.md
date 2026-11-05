# Fraud Investigation AI Assistant (opt-in RAG extension)

This standalone extension adds evidence retrieval and an optional Gemini narrative to the existing **provider-level fraud MLOps accelerator**. It **does not change** the training DAG, production model, promotions, batch scoring, monitoring, or BigQuery schemas.

## Workflow

1. Investigator supplies a provider ID and question.
2. Retrieval reads provider-level predictions and active TEST features in BigQuery (live mode), or fixed synthetic demo records (default).
3. A simple chunk retriever selects relevant sections from a curated investigation guide; citations preserve source labels.
4. A deterministic evidence report is produced by default. Optionally Vertex AI Gemini summarizes the evidence and cites numbered passages.
5. Human investigators verify all findings. Neither risk labels nor AI responses prove fraud.

This is **retrieval-augmented generation** only when Gemini mode is enabled. The default offline mode is a deterministic, retrieval-grounded evidence report. The curated-document retriever is lexical, not a vector database; embeddings and a scalable document index are future work.

## Zero-cost local demonstration

From repository root (Python 3.11+):

```bash
python -m src.investigation_assistant.assistant --provider-id P00175
python -m src.investigation_assistant.assistant --provider-id P00175 --question "Explain high acuity coding signals"
python -m pytest tests/unit/test_investigation_assistant.py -v
```

No credentials, package installs, API calls, or cloud charges are required for the standard demo. Data in this mode is **synthetic, fixed, and illustrative**, not fetched from the GCP project.

## Optional live BigQuery

Install optional dependencies:

```bash
pip install -r src/investigation_assistant/requirements.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=fwa-mlops-accelerator-demo
python -m src.investigation_assistant.assistant --provider-id P00175 --live
```

Windows PowerShell: `$env:GOOGLE_CLOUD_PROJECT="fwa-mlops-accelerator-demo"`.

Live mode uses read-only **parameterized** SQL queries. Table locations and model feature names come from `src/config/pipeline_config.yaml`. It reads `data.provider_prediction_table` and `data.feature_table` (TEST split); it does not read claim-level PHI. BigQuery job creation and table-read permissions are required. If a provider row or expected column is missing, fix the schema/configuration rather than inventing evidence. Query costs may apply.

## Optional Vertex AI Gemini

With Vertex AI access enabled and ADC credentials:

```bash
export GOOGLE_CLOUD_LOCATION=us-central1
python -m src.investigation_assistant.assistant --provider-id P00175 --gemini
# Or read current BigQuery evidence + Gemini:
python -m src.investigation_assistant.assistant --provider-id P00175 --live --gemini
```

Use only approved synthetic/de-identified data until organizational security, access, logging, privacy, retention, and data-handling reviews are complete. The application does not currently implement user authentication, audit logging, prompt-injection defense guarantees, verified citation checking, or row-level security; **do not deploy as a production investigator-facing service**. LLM citations are instructed, not programmatically validated. The project does not claim causality or calibrated probabilities.

## Next steps

- Add a verified evidence/citation validator, testable structured JSON output, and investigator-authenticated UI.
- Retrieve approved billing policies and redacted investigation documents using embeddings and a vector index, with access controls.
- Add controlled BigQuery feedback lookup only after confirming its exact schema and authorization boundaries.
- Add evaluation cases measuring evidence faithfulness, refusal on missing evidence, and policy-document retrieval quality.
