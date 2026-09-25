"""Global test fixtures.

Point the settings loader at a non-existent .env *before* secai.config is
imported, so unit tests see a clean environment rather than the developer's
local secrets.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("SECAI_ENV_FILE", str(Path(__file__).parent / "fixtures" / "absent.env"))

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from secai.config import Settings, get_settings, reload_settings
from secai.telemetry import init_telemetry

# Every env var the settings tree can read; cleared between tests.
_MANAGED_PREFIXES = ("SECAI_", "LANGFUSE_", "OTEL_", "VLLM_")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove secai-related env vars and reset the settings cache.

    The cache is only *cleared*, never repopulated, so a test that sets env
    vars still gets them picked up by the first ``get_settings()`` call.
    """
    for key in list(os.environ):
        if key.startswith(_MANAGED_PREFIXES) and key != "SECAI_ENV_FILE":
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Freshly loaded settings for the current environment."""
    return reload_settings()


@pytest.fixture(scope="session")
def span_exporter() -> InMemorySpanExporter:
    """The one in-memory exporter for the entire test session.

    OpenTelemetry installs its global tracer provider once per process, so the
    first Langfuse client to initialise wins and every later exporter is
    silently ignored. Sharing a single exporter across all telemetry tests is
    the only arrangement that actually works; per-file exporters appear to work
    until a second file starts producing spans.
    """
    memory = InMemorySpanExporter()
    init_telemetry(span_exporter=memory, force=True)
    return memory


@pytest.fixture
def exporter(span_exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    """A clean, enabled telemetry pipeline writing into the session exporter."""
    telemetry = init_telemetry(span_exporter=span_exporter, force=True)
    # Langfuse batches spans: drain anything still in flight from the previous
    # test before clearing, or it lands in this test's results.
    telemetry.flush()
    span_exporter.clear()
    yield span_exporter
