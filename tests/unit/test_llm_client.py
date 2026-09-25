"""LLMClient tests (JOB-02).

Mocking happens at the OpenAI client seam rather than at HTTP: the SDK is built
on httpx2 while the rest of the codebase uses httpx, so respx cannot intercept
it. Mocking the seam also tests what we actually own — retry policy, schema
enforcement, and what lands on the trace.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

from secai.config import Settings, reload_settings
from secai.llm import (
    LLMClient,
    LLMEmptyResponseError,
    LLMSchemaError,
    LLMTransportError,
)
from secai.telemetry import init_telemetry, job_trace


class Tranche(BaseModel):
    """A miniature stand-in for the JOB-04 schema."""

    class_name: str = Field(alias="class")
    original_balance: int
    coupon_margin: float


VALID_JSON = '{"class": "A", "original_balance": 250000000, "coupon_margin": 1.35}'
MESSAGES = [{"role": "user", "content": "extract the senior tranche"}]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


DEFAULT_USAGE = FakeUsage(120, 40)


class FakeCompletion:
    def __init__(
        self,
        content: str | None,
        *,
        usage: FakeUsage | None = DEFAULT_USAGE,
        choices: bool = True,
    ) -> None:
        self.choices = [FakeChoice(content)] if choices else []
        self.usage = usage


class FakeCompletions:
    """Replays a scripted sequence of results, recording each call."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self._results.pop(0) if self._results else self._results
        if isinstance(result, BaseException):
            raise result
        return result


class FakeOpenAI:
    def __init__(self, results: list[Any]) -> None:
        self.completions = FakeCompletions(results)
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def status_error(code: int) -> APIStatusError:
    """An APIStatusError carrying a real status code."""
    request = type("R", (), {"method": "POST", "url": "http://vllm.test"})()
    response = type("Resp", (), {"status_code": code, "headers": {}, "request": request})()
    return APIStatusError(f"status {code}", response=response, body=None)  # type: ignore[arg-type]


def connection_error() -> APIConnectionError:
    request = type("R", (), {"method": "POST", "url": "http://vllm.test"})()
    return APIConnectionError(request=request)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("VLLM_MODEL", "test-model")
    monkeypatch.setenv("VLLM_MAX_RETRIES", "3")
    return reload_settings()


def make_client(results: list[Any], settings: Settings) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI(results)
    client = LLMClient(settings=settings, openai_client=fake)  # type: ignore[arg-type]
    return client, fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestStructuredGeneration:
    def test_returns_a_validated_model(self, settings: Settings) -> None:
        client, _ = make_client([FakeCompletion(VALID_JSON)], settings)
        result = client.generate_structured(MESSAGES, Tranche)
        assert isinstance(result, Tranche)
        assert result.original_balance == 250_000_000
        assert result.coupon_margin == 1.35

    def test_guided_decoding_is_always_sent(self, settings: Settings) -> None:
        client, fake = make_client([FakeCompletion(VALID_JSON)], settings)
        client.generate_structured(MESSAGES, Tranche)

        body = fake.completions.calls[0]["extra_body"]
        # The guardrail: extraction is schema-constrained, never free-text.
        assert "guided_json" in body
        assert body["guided_json"]["properties"]["original_balance"]["type"] == "integer"

    def test_traceparent_header_is_sent(
        self, settings: Settings, exporter: InMemorySpanExporter
    ) -> None:
        client, fake = make_client([FakeCompletion(VALID_JSON)], settings)
        with job_trace("term_extraction"):
            client.generate_structured(MESSAGES, Tranche)

        headers = fake.completions.calls[0]["extra_headers"]
        # This is what makes vLLM's server spans nest under our trace.
        assert "traceparent" in headers

    def test_settings_supply_the_defaults(self, settings: Settings) -> None:
        client, fake = make_client([FakeCompletion(VALID_JSON)], settings)
        client.generate_structured(MESSAGES, Tranche)
        call = fake.completions.calls[0]
        assert call["model"] == "test-model"
        assert call["temperature"] == 0.0  # deterministic extraction by default

    def test_overrides_win(self, settings: Settings) -> None:
        client, fake = make_client([FakeCompletion(VALID_JSON)], settings)
        client.generate_structured(
            MESSAGES, Tranche, model="other-model", temperature=0.7, max_tokens=64
        )
        call = fake.completions.calls[0]
        assert call["model"] == "other-model"
        assert call["temperature"] == 0.7
        assert call["max_tokens"] == 64


class TestSchemaFailures:
    def test_malformed_json_raises(self, settings: Settings) -> None:
        client, _ = make_client([FakeCompletion("not json at all")], settings)
        with pytest.raises(LLMSchemaError, match="not valid JSON"):
            client.generate_structured(MESSAGES, Tranche)

    def test_wrong_shape_raises(self, settings: Settings) -> None:
        client, _ = make_client([FakeCompletion('{"class": "A"}')], settings)
        with pytest.raises(LLMSchemaError, match="did not satisfy Tranche"):
            client.generate_structured(MESSAGES, Tranche)

    def test_empty_response_raises(self, settings: Settings) -> None:
        # Distinct from a schema violation: nothing came back at all.
        client, _ = make_client([FakeCompletion("")], settings)
        with pytest.raises(LLMEmptyResponseError, match="empty"):
            client.generate_structured(MESSAGES, Tranche)

    def test_no_choices_raises(self, settings: Settings) -> None:
        client, _ = make_client([FakeCompletion(None, choices=False)], settings)
        with pytest.raises(LLMEmptyResponseError, match="empty"):
            client.generate_structured(MESSAGES, Tranche)

    def test_schema_failure_is_not_retried(self, settings: Settings) -> None:
        # Retrying a schema failure burns tokens for an identical result.
        client, fake = make_client([FakeCompletion("nonsense")], settings)
        with pytest.raises(LLMSchemaError):
            client.generate_structured(MESSAGES, Tranche)
        assert len(fake.completions.calls) == 1

    def test_raw_output_is_preserved_for_triage(self, settings: Settings) -> None:
        client, _ = make_client([FakeCompletion("nonsense")], settings)
        with pytest.raises(LLMSchemaError) as caught:
            client.generate_structured(MESSAGES, Tranche)
        assert caught.value.raw_output == "nonsense"


class TestRetries:
    def test_connection_error_is_retried_then_succeeds(self, settings: Settings) -> None:
        client, fake = make_client([connection_error(), FakeCompletion(VALID_JSON)], settings)
        result = client.generate_structured(MESSAGES, Tranche)
        assert result.original_balance == 250_000_000
        assert len(fake.completions.calls) == 2

    def test_timeout_is_retried(self, settings: Settings) -> None:
        request = type("R", (), {"method": "POST", "url": "http://vllm.test"})()
        client, fake = make_client(
            [APITimeoutError(request=request), FakeCompletion(VALID_JSON)],  # type: ignore[arg-type]
            settings,
        )
        client.generate_structured(MESSAGES, Tranche)
        assert len(fake.completions.calls) == 2

    def test_503_is_retried(self, settings: Settings) -> None:
        client, fake = make_client([status_error(503), FakeCompletion(VALID_JSON)], settings)
        client.generate_structured(MESSAGES, Tranche)
        assert len(fake.completions.calls) == 2

    def test_429_is_retried(self, settings: Settings) -> None:
        client, fake = make_client([status_error(429), FakeCompletion(VALID_JSON)], settings)
        client.generate_structured(MESSAGES, Tranche)
        assert len(fake.completions.calls) == 2

    def test_400_is_not_retried(self, settings: Settings) -> None:
        # A bad request is our fault; retrying cannot fix it.
        client, fake = make_client([status_error(400)], settings)
        with pytest.raises(LLMTransportError, match="rejected the request"):
            client.generate_structured(MESSAGES, Tranche)
        assert len(fake.completions.calls) == 1

    def test_retries_are_bounded(self, settings: Settings) -> None:
        client, fake = make_client([connection_error() for _ in range(5)], settings)
        with pytest.raises(LLMTransportError, match="after retries"):
            client.generate_structured(MESSAGES, Tranche)
        assert len(fake.completions.calls) == settings.vllm.max_retries


class TestTracing:
    def test_generation_span_records_model_and_usage(
        self, settings: Settings, exporter: InMemorySpanExporter
    ) -> None:
        client, _ = make_client([FakeCompletion(VALID_JSON)], settings)
        with job_trace("term_extraction"):
            client.generate_structured(MESSAGES, Tranche)

        init_telemetry(span_exporter=exporter, force=True).flush()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert "llm.extract" in spans

        attrs = spans["llm.extract"].attributes or {}
        blob = " ".join(str(v) for v in attrs.values())
        assert "test-model" in blob
        # Token usage must reach the trace: it is the cost_tokens KPI.
        assert "120" in blob and "40" in blob

    def test_span_is_recorded_even_when_the_schema_fails(
        self, settings: Settings, exporter: InMemorySpanExporter
    ) -> None:
        client, _ = make_client([FakeCompletion("nonsense")], settings)
        with pytest.raises(LLMSchemaError), job_trace("term_extraction"):
            client.generate_structured(MESSAGES, Tranche)

        init_telemetry(span_exporter=exporter, force=True).flush()
        names = {s.name for s in exporter.get_finished_spans()}
        # A failed call still cost tokens; it must not vanish from the trace.
        assert "llm.extract" in names


class TestLifecycle:
    def test_context_manager_closes_the_client(self, settings: Settings) -> None:
        fake = FakeOpenAI([FakeCompletion(VALID_JSON)])
        with LLMClient(settings=settings, openai_client=fake) as client:  # type: ignore[arg-type]
            client.generate_structured(MESSAGES, Tranche)
        assert fake.closed is True
