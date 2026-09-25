"""Span helpers: one root trace per job run, canonical child spans, scores.

CLAUDE.md §6.4 fixes the shape of every trace, so the rules live here rather
than being restated at each call site:

* a job run is exactly one root trace named ``job.<job_name>``;
* it carries the ``secai.*`` attributes;
* ``session_id``, ``user_id``, tags and metadata propagate to every child span;
* child spans use the canonical names (``parse``, ``chunk``, …);
* exceptions are recorded and the span status set to ERROR — never swallowed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from langfuse import propagate_attributes

from secai.telemetry.attributes import (
    ATTR_ASSET_CLASS,
    ATTR_DEAL_ID,
    ATTR_DOC_TYPE,
    ATTR_JOB,
    ATTR_MODEL,
    ATTR_PROMPT_VERSION,
    BOOLEAN_SCORES,
    SPAN_LLM_EXTRACT,
    job_span_name,
)
from secai.telemetry.setup import get_client

logger = logging.getLogger(__name__)


def _job_attributes(
    *,
    job: str,
    deal_id: str | None,
    asset_class: str | None,
    doc_type: str | None,
    prompt_version: str | None,
    model: str | None,
) -> dict[str, str]:
    """Build the secai.* attribute set, omitting what wasn't supplied."""
    candidates = {
        ATTR_JOB: job,
        ATTR_DEAL_ID: deal_id,
        ATTR_ASSET_CLASS: asset_class,
        ATTR_DOC_TYPE: doc_type,
        ATTR_PROMPT_VERSION: prompt_version,
        ATTR_MODEL: model,
    }
    return {key: value for key, value in candidates.items() if value is not None}


@contextmanager
def job_trace(
    job: str,
    *,
    deal_id: str | None = None,
    asset_class: str | None = None,
    doc_type: str | None = None,
    prompt_version: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    input_data: Any = None,
) -> Iterator[Any]:
    """Open the root trace for one job run.

    ``session_id`` is the batch or run ID and ``user_id`` the analyst, if any;
    both propagate to every child span via the SDK helper.

    Yields the Langfuse span so callers can attach output or extra metadata.
    """
    client = get_client()
    attributes = _job_attributes(
        job=job,
        deal_id=deal_id,
        asset_class=asset_class,
        doc_type=doc_type,
        prompt_version=prompt_version,
        model=model,
    )

    with (
        propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=list(tags) if tags is not None else None,
            metadata=dict(metadata) if metadata is not None else None,
        ),
        client.start_as_current_observation(
            name=job_span_name(job),
            as_type="span",
            input=input_data,
            metadata=dict(metadata) if metadata is not None else None,
        ) as span,
    ):
        _set_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            record_exception(span, exc)
            raise


@contextmanager
def job_span(
    name: str,
    *,
    input_data: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a plain child span. Use the canonical names from ``attributes``.

    For LLM calls use :func:`llm_generation` instead, so Langfuse records the
    model, token usage and cost.
    """
    client = get_client()
    with client.start_as_current_observation(
        name=name,
        as_type="span",
        input=input_data,
        metadata=dict(metadata) if metadata is not None else None,
    ) as span:
        try:
            yield span
        except Exception as exc:
            record_exception(span, exc)
            raise


@contextmanager
def llm_generation(
    name: str = SPAN_LLM_EXTRACT,
    *,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    input_data: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a Langfuse *generation* span for an LLM call.

    Token usage and cost are attached by the caller once the response is in,
    via ``span.update(usage_details=..., output=...)``.
    """
    client = get_client()
    with client.start_as_current_observation(
        name=name,
        as_type="generation",
        input=input_data,
        model=model,
        model_parameters=model_parameters,
        metadata=dict(metadata) if metadata is not None else None,
    ) as span:
        try:
            yield span
        except Exception as exc:
            record_exception(span, exc)
            raise


def _set_attributes(span: Any, attributes: Mapping[str, str]) -> None:
    """Set raw OTel attributes on a Langfuse span wrapper."""
    otel_span = getattr(span, "_otel_span", None)
    if otel_span is None:
        return
    for key, value in attributes.items():
        otel_span.set_attribute(key, value)


def record_exception(span: Any, exc: BaseException) -> None:
    """Record an exception and set the span status to ERROR.

    Errors are never swallowed: callers re-raise. This only annotates.
    """
    otel_span = getattr(span, "_otel_span", None)
    if otel_span is None:
        return

    # Imported lazily so this module stays importable without the SDK types
    # being resolved at import time.
    from opentelemetry.trace import Status, StatusCode

    otel_span.record_exception(exc)
    otel_span.set_status(Status(StatusCode.ERROR, str(exc)))


def score_trace(name: str, value: float | bool | str, *, comment: str | None = None) -> None:
    """Attach a score to the current trace.

    Booleans are sent with Langfuse's BOOLEAN data type so the UI renders them
    as pass/fail rather than 0/1.
    """
    client = get_client()
    if isinstance(value, str):
        client.score_current_trace(name=name, value=value, data_type="CATEGORICAL", comment=comment)
        return
    client.score_current_trace(
        name=name,
        value=float(value),
        data_type=_numeric_data_type(name, value),
        comment=comment,
    )


def score_span(name: str, value: float | bool | str, *, comment: str | None = None) -> None:
    """Attach a score to the current span."""
    client = get_client()
    if isinstance(value, str):
        client.score_current_span(name=name, value=value, data_type="CATEGORICAL", comment=comment)
        return
    client.score_current_span(
        name=name,
        value=float(value),
        data_type=_numeric_data_type(name, value),
        comment=comment,
    )


def _numeric_data_type(name: str, value: float | bool) -> Literal["NUMERIC", "BOOLEAN"]:
    """Scores that are conceptually pass/fail are sent as BOOLEAN."""
    if isinstance(value, bool) or name in BOOLEAN_SCORES:
        return "BOOLEAN"
    return "NUMERIC"


def current_trace_id() -> str | None:
    """The active trace ID, for recording in PROGRESS.md or a report."""
    return get_client().get_current_trace_id()


def trace_url() -> str | None:
    """A deep link to the current trace in the Langfuse UI, or None.

    Resolving the link calls the Langfuse API, which can fail (bad credentials,
    server down). This is a convenience for logs and reports, never job logic,
    so a failure returns None rather than taking the run down with it. Span
    errors are still recorded and re-raised — see :func:`record_exception`.
    """
    try:
        return get_client().get_trace_url()
    except Exception:
        logger.debug("could not resolve the Langfuse trace URL", exc_info=True)
        return None
