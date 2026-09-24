"""Telemetry tests (JOB-01).

Spans are captured with an ``InMemorySpanExporter`` and asserted on directly,
so these cover the real export pipeline (including masking) without a network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from secai.telemetry import (
    ATTR_ASSET_CLASS,
    ATTR_DEAL_ID,
    ATTR_DOC_TYPE,
    ATTR_JOB,
    ATTR_MODEL,
    ATTR_PROMPT_VERSION,
    SPAN_LLM_EXTRACT,
    SPAN_PARSE,
    SPAN_VALIDATE,
    init_telemetry,
    inject_trace_context,
    job_span,
    job_span_name,
    job_trace,
    llm_generation,
)
from secai.telemetry.pii import contains_pii, reset_masker


# OpenTelemetry's global tracer provider can only be installed once per
# process, so every test shares a single exporter instance and clears it
# between cases rather than building a new pipeline each time.
@pytest.fixture(scope="session")
def shared_exporter() -> InMemorySpanExporter:
    memory = InMemorySpanExporter()
    init_telemetry(span_exporter=memory, force=True)
    return memory


@pytest.fixture
def exporter(shared_exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    """A clean, enabled telemetry pipeline writing into the shared exporter."""
    reset_masker()
    telemetry = init_telemetry(span_exporter=shared_exporter, force=True)
    # Langfuse batches spans, so drain anything still in flight from the
    # previous test *before* clearing, or it lands in this test's results.
    telemetry.flush()
    shared_exporter.clear()
    yield shared_exporter
    reset_masker()


def finished(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    """Flush and return the exported spans."""
    from secai.telemetry import get_telemetry

    get_telemetry().flush()
    return list(exporter.get_finished_spans())


def by_name(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f"no span named {name!r}; got {[s.name for s in spans]}"
    return matches[0]


class TestSetup:
    def test_exporter_enables_tracing_without_credentials(
        self, exporter: InMemorySpanExporter
    ) -> None:
        from secai.telemetry import get_telemetry

        assert get_telemetry().enabled is True

    def test_init_is_idempotent(self, exporter: InMemorySpanExporter) -> None:
        from secai.telemetry import get_telemetry
        from secai.telemetry import init_telemetry as init

        first = get_telemetry()
        assert init() is first

    def test_disabled_without_credentials(self, shared_exporter: InMemorySpanExporter) -> None:
        # No credentials and no exporter: tracing must switch itself off rather
        # than silently buffering spans that can never be delivered.
        telemetry = init_telemetry(force=True)
        assert telemetry.enabled is False
        # Restore the shared pipeline for subsequent tests.
        init_telemetry(span_exporter=shared_exporter, force=True)


class TestRootTrace:
    def test_root_span_is_named_job_dot_name(self, exporter: InMemorySpanExporter) -> None:
        with job_trace("term_extraction"):
            pass
        spans = finished(exporter)
        assert by_name(spans, job_span_name("term_extraction")) is not None
        assert by_name(spans, "job.term_extraction").parent is None

    def test_all_secai_attributes_are_set(self, exporter: InMemorySpanExporter) -> None:
        with job_trace(
            "term_extraction",
            deal_id="SYNTH-2024-1",
            asset_class="RMBS",
            doc_type="prospectus",
            prompt_version="v3",
            model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        ):
            pass

        root = by_name(finished(exporter), "job.term_extraction")
        attrs = root.attributes or {}
        assert attrs[ATTR_JOB] == "term_extraction"
        assert attrs[ATTR_DEAL_ID] == "SYNTH-2024-1"
        assert attrs[ATTR_ASSET_CLASS] == "RMBS"
        assert attrs[ATTR_DOC_TYPE] == "prospectus"
        assert attrs[ATTR_PROMPT_VERSION] == "v3"
        assert attrs[ATTR_MODEL] == "Qwen/Qwen2.5-7B-Instruct-AWQ"

    def test_unset_attributes_are_omitted_not_stringified(
        self, exporter: InMemorySpanExporter
    ) -> None:
        with job_trace("term_extraction"):
            pass
        attrs = by_name(finished(exporter), "job.term_extraction").attributes or {}
        # "None" as a string in a dashboard filter is worse than a missing key.
        assert ATTR_DEAL_ID not in attrs
        assert ATTR_MODEL not in attrs


class TestSpanHierarchy:
    def test_child_spans_nest_under_the_root(self, exporter: InMemorySpanExporter) -> None:
        with job_trace("term_extraction"):
            with job_span(SPAN_PARSE):
                pass
            with job_span(SPAN_VALIDATE):
                pass

        spans = finished(exporter)
        root = by_name(spans, "job.term_extraction")
        parse = by_name(spans, SPAN_PARSE)
        validate = by_name(spans, SPAN_VALIDATE)

        assert parse.parent is not None
        assert parse.parent.span_id == root.context.span_id
        assert validate.parent.span_id == root.context.span_id
        # One trace per job run.
        assert parse.context.trace_id == root.context.trace_id
        assert validate.context.trace_id == root.context.trace_id

    def test_generation_span_is_nested_and_typed(self, exporter: InMemorySpanExporter) -> None:
        with (
            job_trace("term_extraction", model="test-model"),
            llm_generation(model="test-model", model_parameters={"temperature": 0.0}),
        ):
            pass

        spans = finished(exporter)
        generation = by_name(spans, SPAN_LLM_EXTRACT)
        root = by_name(spans, "job.term_extraction")
        assert generation.parent.span_id == root.context.span_id
        # Langfuse marks observation type in the span attributes.
        attrs = generation.attributes or {}
        assert any("generation" in str(v).lower() for v in attrs.values())


class TestErrors:
    def test_exception_sets_error_status_and_is_reraised(
        self, exporter: InMemorySpanExporter
    ) -> None:
        with (
            pytest.raises(ValueError, match="boom"),
            job_trace("term_extraction"),
            job_span(SPAN_VALIDATE),
        ):
            raise ValueError("boom")

        spans = finished(exporter)
        validate = by_name(spans, SPAN_VALIDATE)
        assert validate.status.status_code is StatusCode.ERROR
        assert validate.events, "the exception should be recorded as a span event"
        assert any(e.name == "exception" for e in validate.events)

    def test_error_propagates_to_the_root_span(self, exporter: InMemorySpanExporter) -> None:
        with pytest.raises(ValueError), job_trace("term_extraction"):
            raise ValueError("boom")
        root = by_name(finished(exporter), "job.term_extraction")
        assert root.status.status_code is StatusCode.ERROR


class TestExportMasking:
    """The acceptance test: no unmasked PII may reach an exported span."""

    def test_pii_in_span_attributes_is_removed(self, exporter: InMemorySpanExporter) -> None:
        with job_trace("term_extraction", deal_id="SYNTH-2024-1") as span:
            with job_span(SPAN_PARSE) as child:
                child.update(
                    metadata={
                        "borrower_name": "Jane Doe",
                        "loan_number": "L-99887766",
                        "tranche_balance": 250_000_000,
                    }
                )
            span.update(output={"note": "adviser jane.doe@example.com, SSN 123-45-6789"})

        payload = json.dumps(
            [{"name": s.name, "attributes": dict(s.attributes or {})} for s in finished(exporter)]
        )

        assert "Jane Doe" not in payload
        assert "jane.doe@example.com" not in payload
        assert "123-45-6789" not in payload
        assert "L-99887766" not in payload
        assert not contains_pii(json.loads(payload))
        # Deal-level data must still be there.
        assert "SYNTH-2024-1" in payload
        assert "250000000" in payload or "250_000_000" in payload


class TestTraceContextPropagation:
    def test_traceparent_is_injected_inside_a_span(self, exporter: InMemorySpanExporter) -> None:
        with job_trace("term_extraction"):
            headers = inject_trace_context()
        assert "traceparent" in headers
        # W3C format: version-traceid-spanid-flags
        version, trace_id, span_id, _flags = headers["traceparent"].split("-")
        assert version == "00"
        assert len(trace_id) == 32
        assert len(span_id) == 16

    def test_injected_trace_id_matches_the_active_trace(
        self, exporter: InMemorySpanExporter
    ) -> None:
        with job_trace("term_extraction"):
            headers = inject_trace_context()
        root = by_name(finished(exporter), "job.term_extraction")
        assert headers["traceparent"].split("-")[1] == format(root.context.trace_id, "032x")

    def test_existing_headers_are_preserved(self, exporter: InMemorySpanExporter) -> None:
        with job_trace("term_extraction"):
            headers = inject_trace_context({"authorization": "Bearer x"})
        assert headers["authorization"] == "Bearer x"
        assert "traceparent" in headers
