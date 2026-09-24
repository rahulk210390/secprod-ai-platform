"""OpenTelemetry + Langfuse instrumentation for secai.

Entry points call :func:`init_telemetry` once, then use :func:`job_trace` for
the root span of a job run and :func:`job_span` for the canonical child spans.
"""

from secai.telemetry.attributes import (
    ATTR_ASSET_CLASS,
    ATTR_DEAL_ID,
    ATTR_DOC_TYPE,
    ATTR_JOB,
    ATTR_MODEL,
    ATTR_PROMPT_VERSION,
    SPAN_CHUNK,
    SPAN_LLM_EXTRACT,
    SPAN_PARSE,
    SPAN_PERSIST,
    SPAN_RETRIEVE,
    SPAN_VALIDATE,
    job_span_name,
)
from secai.telemetry.http import (
    inject_trace_context,
    traced_async_http_client,
    traced_http_client,
)
from secai.telemetry.pii import Masker, contains_pii, get_masker, mask, reset_masker
from secai.telemetry.setup import (
    Telemetry,
    get_client,
    get_telemetry,
    init_telemetry,
    reset_telemetry,
    shutdown_telemetry,
)
from secai.telemetry.tracing import (
    current_trace_id,
    job_span,
    job_trace,
    llm_generation,
    record_exception,
    score_span,
    score_trace,
    trace_url,
)

__all__ = [
    "ATTR_ASSET_CLASS",
    "ATTR_DEAL_ID",
    "ATTR_DOC_TYPE",
    "ATTR_JOB",
    "ATTR_MODEL",
    "ATTR_PROMPT_VERSION",
    "SPAN_CHUNK",
    "SPAN_LLM_EXTRACT",
    "SPAN_PARSE",
    "SPAN_PERSIST",
    "SPAN_RETRIEVE",
    "SPAN_VALIDATE",
    "Masker",
    "Telemetry",
    "contains_pii",
    "current_trace_id",
    "get_client",
    "get_masker",
    "get_telemetry",
    "init_telemetry",
    "inject_trace_context",
    "job_span",
    "job_span_name",
    "job_trace",
    "llm_generation",
    "mask",
    "record_exception",
    "reset_masker",
    "reset_telemetry",
    "score_span",
    "score_trace",
    "shutdown_telemetry",
    "trace_url",
    "traced_async_http_client",
    "traced_http_client",
]
