"""Telemetry initialisation — call this once, at every entry point.

The application uses the Langfuse Python SDK v4, which is OTel-native and
exports straight to Langfuse's OTLP/HTTP endpoint. vLLM's own spans travel a
different road: vLLM exports gRPC to the OTel Collector, which forwards them to
the same Langfuse project. The two halves join up because the app propagates
W3C trace context on its vLLM requests (see :mod:`secai.telemetry.http`).

Masking is wired in at both hooks the SDK offers:

* ``mask`` — observation inputs and outputs, before serialisation;
* ``mask_otel_spans`` — raw span attributes, at export time.

The Collector's ``attributes/pii`` processor is the third, independent layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

from langfuse import Langfuse
from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanPatch,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from secai.config import Settings, get_settings
from secai.telemetry.pii import get_masker, mask

logger = logging.getLogger(__name__)

# Placeholder credentials used only when spans go to a caller-supplied
# exporter; they are never sent anywhere.
OFFLINE_KEY: Final = "offline"


def _mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    """Redact sensitive span attributes at export time.

    Runs on the batch processor thread, so it stays allocation-light and never
    touches request locals or the active span.
    """
    masker = get_masker()
    patches: dict[Any, OtelSpanPatch | None] = {}

    for identifier, span in params.spans.items():
        deletions: list[str] = []
        replacements: dict[str, Any] = {}

        for key, value in span.attributes.items():
            if masker.is_sensitive_key(key):
                deletions.append(key)
            elif isinstance(value, str):
                masked = masker.mask_text(value)
                if masked != value:
                    replacements[key] = masked

        if deletions or replacements:
            patches[identifier] = OtelSpanPatch(
                delete_attributes=tuple(deletions),
                set_attributes=replacements,
            )

    return MaskOtelSpansResult(span_patches=patches)


@dataclass
class Telemetry:
    """Handle on the initialised telemetry stack."""

    client: Langfuse
    enabled: bool
    settings: Settings

    def flush(self) -> None:
        """Force-export everything buffered. Call before a process exits."""
        self.client.flush()

    def shutdown(self) -> None:
        self.client.shutdown()


_telemetry: Telemetry | None = None


def init_telemetry(
    *,
    settings: Settings | None = None,
    span_exporter: SpanExporter | None = None,
    tracer_provider: TracerProvider | None = None,
    force: bool = False,
) -> Telemetry:
    """Initialise telemetry once and return the handle.

    Args:
        settings: override the process settings (tests).
        span_exporter: send spans somewhere other than Langfuse.
        tracer_provider: supply an existing provider instead of letting the SDK
            build one. Tests pass a provider carrying an
            ``InMemorySpanExporter``; note that supplying *both* a provider and
            an exporter double-exports, since Langfuse adds its own processor to
            the provider it is given.
        force: re-initialise even if already set up.

    Tracing is enabled only when Langfuse credentials are present and
    ``LANGFUSE_ENABLED`` is true, *or* when an explicit exporter or tracer
    provider is supplied.
    That keeps unit tests and offline runs from trying to reach a server, while
    still exercising the full span pipeline.
    """
    global _telemetry

    if _telemetry is not None and not force:
        return _telemetry

    resolved = settings if settings is not None else get_settings()
    langfuse_settings = resolved.langfuse

    offline = span_exporter is not None or tracer_provider is not None
    enabled = offline or (langfuse_settings.enabled and langfuse_settings.is_configured)

    if not enabled:
        logger.warning(
            "Langfuse is not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY); "
            "tracing is disabled and spans will be dropped."
        )

    # The SDK disables itself outright without a key pair, which would also
    # silence a caller-supplied exporter. An explicit exporter means "send spans
    # here, not to Langfuse", so placeholders keep the pipeline alive offline.
    public_key = langfuse_settings.public_key or (OFFLINE_KEY if offline else None)
    secret_key = langfuse_settings.secret_key or (OFFLINE_KEY if offline else None)

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=langfuse_settings.host,
        environment=str(resolved.environment),
        tracing_enabled=enabled,
        mask=mask,
        mask_otel_spans=_mask_otel_spans,
        span_exporter=span_exporter,
        tracer_provider=tracer_provider,
    )

    _telemetry = Telemetry(client=client, enabled=enabled, settings=resolved)
    return _telemetry


def get_telemetry() -> Telemetry:
    """Return the initialised telemetry handle, initialising on first use."""
    if _telemetry is None:
        return init_telemetry()
    return _telemetry


def get_client() -> Langfuse:
    """Shorthand for the Langfuse client."""
    return get_telemetry().client


def shutdown_telemetry() -> None:
    """Flush and tear down. Safe to call when never initialised."""
    global _telemetry
    if _telemetry is not None:
        _telemetry.flush()
        _telemetry.shutdown()
        _telemetry = None


def reset_telemetry() -> None:
    """Drop the handle without flushing. Tests use this between cases."""
    global _telemetry
    _telemetry = None
