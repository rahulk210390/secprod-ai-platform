"""The extractor and the traced pipeline, with a scripted OpenAI client.

The fake answers each call according to the schema it was sent, the way vLLM
would under structured outputs, so the whole path runs: contexts → prompts →
generations → assembly → validation → scores → persist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import term_fixtures as fx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from secai.config import Settings, reload_settings
from secai.jobs.term_extraction.context import Target
from secai.jobs.term_extraction.extract import TermExtractor, load_extraction
from secai.jobs.term_extraction.pipeline import JOB_NAME, run_term_extraction
from secai.llm import LLMClient, LLMTransportError
from secai.telemetry import (
    SPAN_LLM_EXTRACT,
    SPAN_PERSIST,
    SPAN_RETRIEVE,
    SPAN_VALIDATE,
    get_telemetry,
    job_span_name,
)


class _Usage:
    prompt_tokens, completion_tokens, total_tokens = 300, 50, 350


class _Completion:
    def __init__(self, content: str) -> None:
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message})()]
        self.usage = _Usage()


class RoutingOpenAI:
    """Answers by the requested schema's name; records every call."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        name = kwargs["response_format"]["json_schema"]["name"]
        answer = self.answers[name]
        if isinstance(answer, BaseException):
            raise answer
        return _Completion(answer if isinstance(answer, str) else json.dumps(answer))

    def close(self) -> None:  # pragma: no cover - LLMClient API
        pass


def _answers() -> dict[str, Any]:
    e = fx.extraction()
    assert e.facts and e.tranches
    return {
        "RawDealFacts": e.facts[0][0].model_dump(),
        "RawTranches": e.tranches.model_dump(),
        "RawTriggers": e.triggers[0][0].model_dump(),
        "RawCoverageTests": {"tests": []},
    }


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("VLLM_MODEL", "test-model")
    monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
    # The working local configuration (JOB-02): room for the excerpt and the answer.
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "1024")
    return reload_settings()


def _client(settings: Settings, answers: dict[str, Any]) -> tuple[LLMClient, RoutingOpenAI]:
    fake = RoutingOpenAI(answers)
    return LLMClient(settings=settings, openai_client=fake), fake  # type: ignore[arg-type]


def _run(settings: Settings, client: LLMClient, tmp_path: Path, **kw: Any):  # type: ignore[no-untyped-def]
    return run_term_extraction(
        Path("synthetic.pdf"),
        document=fx.document(),
        client=client,
        settings=settings,
        output_dir=tmp_path / "out",
        **kw,
    )


# --------------------------------------------------------------------------
# extractor
# --------------------------------------------------------------------------
def test_one_call_per_context_with_structured_output(settings: Settings) -> None:
    client, fake = _client(settings, _answers())
    extractor = TermExtractor.create(client, settings)
    from secai.jobs.term_extraction.context import build_contexts

    contexts = {t: build_contexts(fx.document(), t, 10_000) for t in Target}
    extraction, issues = extractor.run(contexts, "synthetic.pdf")

    assert issues == []
    # facts, tranches, triggers; no coverage sections, so no coverage call.
    assert [c["response_format"]["json_schema"]["name"] for c in fake.calls] == [
        "RawTranches",
        "RawDealFacts",
        "RawTriggers",
    ]
    assert extraction.tranches is not None and len(extraction.tranches.tranches) == 3
    first = fake.calls[0]["messages"]
    assert first[0]["role"] == "system" and "exactly as it is printed" in first[0]["content"]
    assert "$ 250,000,000" in first[1]["content"]


def test_prompt_version_changes_with_any_prompt(settings: Settings) -> None:
    client, _ = _client(settings, _answers())
    extractor = TermExtractor.create(client, settings)
    before = extractor.prompt_version
    tranche_prompt = extractor.prompts[Target.TRANCHES]
    extractor.prompts[Target.TRANCHES] = tranche_prompt.__class__(
        name=tranche_prompt.name,
        template=tranche_prompt.template + " ",
        version="edited",
        source="local",
    )
    assert extractor.prompt_version != before and before.startswith("set-")


def test_schema_failure_on_one_context_is_recorded_not_fatal(
    settings: Settings, tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    answers = _answers() | {"RawTriggers": "not json at all"}
    client, _ = _client(settings, answers)
    result = _run(settings, client, tmp_path)
    assert "llm_schema_error" in result.report.codes()
    assert result.report.needs_review
    assert result.terms.tranches  # the rest of the deal still came through


def test_transport_failure_propagates(
    settings: Settings, tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    request = type("R", (), {"method": "POST", "url": "http://vllm.test"})()
    from openai import APIConnectionError

    answers = _answers() | {"RawTranches": APIConnectionError(request=request)}  # type: ignore[arg-type]
    client, _ = _client(settings, answers)
    with pytest.raises(LLMTransportError):
        _run(settings, client, tmp_path)
    get_telemetry().flush()
    root = next(s for s in exporter.get_finished_spans() if s.name == job_span_name(JOB_NAME))
    assert root.status.status_code is StatusCode.ERROR


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------
def test_pipeline_extracts_validates_scores_and_persists(
    settings: Settings, tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    client, _ = _client(settings, _answers())
    result = _run(settings, client, tmp_path, gold=fx.gold(), deal_id="SYN-2099-1")

    assert result.report.passed, result.report.issues
    assert result.citation_rate == 1.0
    assert result.evaluation is not None and result.evaluation.critical_field_f1 == 1.0
    assert result.llm_calls == 3 and result.tokens == 3 * 350
    assert result.contexts == {"tranches": 1, "facts": 1, "triggers": 1, "coverage": 0}
    written = json.loads((tmp_path / "out" / "synthetic.deal_terms.json").read_text())
    assert written["tranches"][0]["citations"]["original_balance"]["page_label"] == "1"

    get_telemetry().flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans[job_span_name(JOB_NAME)]
    for child in (SPAN_RETRIEVE, SPAN_VALIDATE, SPAN_PERSIST):
        assert (
            spans[child].parent is not None and spans[child].parent.span_id == root.context.span_id
        )
    generations = [s for s in exporter.get_finished_spans() if s.name == SPAN_LLM_EXTRACT]
    assert len(generations) == 3
    attrs = root.attributes or {}
    assert attrs["secai.asset_class"] == "auto_loan"
    assert attrs["secai.prompt_version"].startswith("set-")
    assert attrs["secai.model"] == "test-model"
    assert (spans[SPAN_VALIDATE].attributes or {})["secai.validate.passed"] is True


def test_extraction_cache_skips_the_model_on_rerun(
    settings: Settings, tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    cache = tmp_path / "extractions"
    client, fake = _client(settings, _answers())
    first = _run(settings, client, tmp_path, extraction_cache_dir=cache)
    second = _run(settings, client, tmp_path, extraction_cache_dir=cache)

    assert (first.from_cache, second.from_cache) == (False, True)
    assert len(fake.calls) == 3  # the rerun made no calls
    assert second.llm_calls == 0 and second.tokens == 0
    assert second.terms == first.terms
    (cached,) = cache.iterdir()
    extraction, issues = load_extraction(cached)
    assert extraction.tranches is not None and issues == []


def test_needs_review_is_scored_when_validation_fails(
    settings: Settings, tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    answers = _answers()
    answers["RawTranches"]["stated_total"] = "$650,000,000"
    client, _ = _client(settings, answers)
    result = _run(settings, client, tmp_path)
    assert result.report.needs_review
    assert "tranche_sum_mismatch" in result.report.codes()
