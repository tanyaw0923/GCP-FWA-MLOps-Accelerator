"""Offline-only tests: no Google Cloud credentials or LLM calls required."""
import pytest
from src.investigation_assistant.assistant import (
    build_context, demo_evidence, generate_answer, investigate,
    retrieve_guidance, validate_provider_id, _table_name,
)


def test_provider_validation_blocks_injection():
    for bad in ["", "x' OR TRUE --", "a.b", "a b", "x" * 65]:
        with pytest.raises(ValueError):
            validate_provider_id(bad)
    assert validate_provider_id("P00175") == "P00175"


def test_table_name_validation():
    assert _table_name("project-1.dataset.table") == "`project-1.dataset.table`"
    with pytest.raises(ValueError):
        _table_name("dataset.table; DROP TABLE X")


def test_retrieval_returns_traceable_chunks():
    chunks = retrieve_guidance("high acuity coding")
    assert chunks and all(c["source"].startswith("guidance:section-") for c in chunks)
    assert "acuity" in chunks[0]["text"].lower()


def test_demo_answer_is_grounded_and_explicitly_synthetic():
    answer = investigate("P00175")
    assert "P00175" in answer
    assert "synthetic" in answer
    assert "not a fraud determination" in answer
    assert "[1]" in answer


def test_empty_evidence_fails_closed():
    with pytest.raises(LookupError):
        generate_answer("P00175", "why?", [], [])


def test_context_preserves_sources():
    context = build_context(demo_evidence("P00175")[:1], retrieve_guidance("acuity")[:1])
    assert "[1] demo.prediction" in context
    assert "[2] guidance:" in context


def test_question_length_limit():
    with pytest.raises(ValueError):
        investigate("P00175", question="x" * 1001)
