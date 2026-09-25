"""Score emission and trace identifiers (JOB-01).

Scores are what the KPIs in CLAUDE.md §5 are actually recorded as, so the data
type matters: a pass/fail sent as NUMERIC renders as 0/1 in Langfuse instead of
a boolean, and the dashboards in §6.6 group on it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from secai.telemetry import (
    current_trace_id,
    init_telemetry,
    job_trace,
    record_exception,
    score_span,
    score_trace,
    trace_url,
)
from secai.telemetry.attributes import (
    SCORE_CRITICAL_FIELD_F1,
    SCORE_NEEDS_REVIEW,
    SCORE_RECON_BREAKS,
    SCORE_TOUCHLESS,
    SCORE_VALIDATION_PASS,
)


@pytest.fixture(scope="session")
def _pipeline() -> InMemorySpanExporter:
    memory = InMemorySpanExporter()
    init_telemetry(span_exporter=memory, force=True)
    return memory


@pytest.fixture
def pipeline(_pipeline: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    init_telemetry(span_exporter=_pipeline, force=True)
    _pipeline.clear()
    yield _pipeline


class Recorder:
    """Captures the score calls the tracing helpers make."""

    def __init__(self) -> None:
        self.trace_scores: list[dict[str, Any]] = []
        self.span_scores: list[dict[str, Any]] = []

    def score_current_trace(self, **kwargs: Any) -> None:
        self.trace_scores.append(kwargs)

    def score_current_span(self, **kwargs: Any) -> None:
        self.span_scores.append(kwargs)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr("secai.telemetry.tracing.get_client", lambda: rec)
    return rec


class TestScoreDataTypes:
    def test_bool_true_is_sent_as_boolean(self, recorder: Recorder) -> None:
        score_trace(SCORE_VALIDATION_PASS, True)
        (call,) = recorder.trace_scores
        assert call["data_type"] == "BOOLEAN"
        assert call["value"] == 1.0

    def test_bool_false_is_sent_as_boolean(self, recorder: Recorder) -> None:
        score_trace(SCORE_NEEDS_REVIEW, False)
        (call,) = recorder.trace_scores
        assert call["data_type"] == "BOOLEAN"
        assert call["value"] == 0.0

    def test_float_is_numeric(self, recorder: Recorder) -> None:
        score_trace(SCORE_CRITICAL_FIELD_F1, 0.97)
        (call,) = recorder.trace_scores
        assert call["data_type"] == "NUMERIC"
        assert call["value"] == 0.97

    def test_int_is_numeric(self, recorder: Recorder) -> None:
        score_trace(SCORE_RECON_BREAKS, 3)
        (call,) = recorder.trace_scores
        assert call["data_type"] == "NUMERIC"
        assert call["value"] == 3.0

    def test_string_is_categorical(self, recorder: Recorder) -> None:
        score_trace("review_outcome", "accepted")
        (call,) = recorder.trace_scores
        assert call["data_type"] == "CATEGORICAL"
        assert call["value"] == "accepted"

    def test_pass_fail_name_forces_boolean_even_for_a_number(self, recorder: Recorder) -> None:
        # touchless is conceptually pass/fail; 1 must not render as NUMERIC.
        score_trace(SCORE_TOUCHLESS, 1)
        (call,) = recorder.trace_scores
        assert call["data_type"] == "BOOLEAN"

    def test_comment_is_forwarded(self, recorder: Recorder) -> None:
        score_trace(SCORE_VALIDATION_PASS, False, comment="tranche sum mismatch")
        (call,) = recorder.trace_scores
        assert call["comment"] == "tranche sum mismatch"


class TestScoreSpan:
    def test_span_scores_use_the_span_endpoint(self, recorder: Recorder) -> None:
        score_span(SCORE_CRITICAL_FIELD_F1, 0.91)
        assert recorder.span_scores and not recorder.trace_scores
        assert recorder.span_scores[0]["data_type"] == "NUMERIC"

    def test_span_boolean(self, recorder: Recorder) -> None:
        score_span(SCORE_VALIDATION_PASS, True)
        assert recorder.span_scores[0]["data_type"] == "BOOLEAN"

    def test_span_categorical(self, recorder: Recorder) -> None:
        score_span("asset_class", "RMBS")
        assert recorder.span_scores[0]["data_type"] == "CATEGORICAL"


class TestIdentifiers:
    def test_trace_id_is_available_inside_a_trace(self, pipeline: InMemorySpanExporter) -> None:
        with job_trace("term_extraction"):
            trace_id = current_trace_id()
        assert trace_id is not None
        assert len(trace_id) == 32

    def test_trace_url_returns_none_rather_than_raising_when_unreachable(
        self, pipeline: InMemorySpanExporter
    ) -> None:
        # Resolving the link calls the Langfuse API; offline it 401s. A cosmetic
        # link must never take a job run down with it.
        with job_trace("term_extraction"):
            assert trace_url() is None

    def test_trace_url_contains_the_trace_id_when_resolvable(
        self, pipeline: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Stub:
            def get_trace_url(self) -> str:
                return "http://langfuse.test/project/p/traces/abc123"

        monkeypatch.setattr("secai.telemetry.tracing.get_client", Stub)
        assert trace_url() == "http://langfuse.test/project/p/traces/abc123"


class TestRecordException:
    def test_no_op_when_the_span_has_no_otel_span(self) -> None:
        # Defensive path: a disabled client yields wrappers without _otel_span.
        class Bare:
            pass

        record_exception(Bare(), ValueError("boom"))  # must not raise


class TestOfflineSemantics:
    """A tracer_provider alone must not be mistaken for "offline".

    Langfuse attaches its own network exporter to whatever provider it is
    given, so treating a supplied provider as offline once shipped debug spans
    to the real server.
    """

    def test_span_exporter_enables_tracing_without_credentials(self) -> None:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        telemetry = init_telemetry(span_exporter=InMemorySpanExporter(), force=True)
        assert telemetry.enabled is True

    def test_tracer_provider_alone_does_not_enable_tracing(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        # No credentials and no exporter: nothing should be emitted anywhere,
        # rather than being sent to LANGFUSE_HOST with placeholder keys.
        telemetry = init_telemetry(tracer_provider=TracerProvider(), force=True)
        assert telemetry.enabled is False

    def test_nothing_supplied_does_not_enable_tracing(self) -> None:
        assert init_telemetry(force=True).enabled is False
