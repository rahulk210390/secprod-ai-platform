"""Trace-context propagation over HTTP (JOB-01).

vLLM's server spans only join the application trace if the request carries W3C
trace context, so these assert on the headers a real httpx request sends.
"""

from __future__ import annotations

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from secai.telemetry import (
    job_trace,
    traced_async_http_client,
    traced_http_client,
)


def _echo(request: httpx.Request) -> httpx.Response:
    """Reflect the trace headers back so the test can inspect them."""
    return httpx.Response(
        200,
        json={
            "traceparent": request.headers.get("traceparent"),
            "authorization": request.headers.get("authorization"),
        },
    )


class TestSyncClient:
    def test_traceparent_is_sent_inside_a_span(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        with (
            job_trace("term_extraction"),
            traced_http_client(timeout=5.0, transport=transport) as client,
        ):
            body = client.get("http://vllm.test/v1/models").json()

        assert body["traceparent"] is not None
        version, trace_id, span_id, _flags = body["traceparent"].split("-")
        assert version == "00"
        assert len(trace_id) == 32
        assert len(span_id) == 16

    def test_trace_id_matches_the_active_trace(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        with job_trace("term_extraction"):
            with traced_http_client(timeout=5.0, transport=transport) as client:
                body = client.get("http://vllm.test/v1/models").json()
            from secai.telemetry import current_trace_id

            active = current_trace_id()

        assert body["traceparent"].split("-")[1] == active

    def test_no_traceparent_outside_a_span(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        with traced_http_client(timeout=5.0, transport=transport) as client:
            body = client.get("http://vllm.test/v1/models").json()
        # Nothing to correlate to, so nothing is sent.
        assert body["traceparent"] is None

    def test_other_headers_are_untouched(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        with (
            job_trace("term_extraction"),
            traced_http_client(
                timeout=5.0,
                transport=transport,
                headers={"authorization": "Bearer token"},
            ) as client,
        ):
            body = client.get("http://vllm.test/v1/models").json()
        assert body["authorization"] == "Bearer token"
        assert body["traceparent"] is not None

    def test_base_url_is_applied(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        client = traced_http_client(
            timeout=5.0, base_url="http://vllm.test/v1", transport=transport
        )
        # httpx normalises base_url with a trailing slash.
        assert str(client.base_url) == "http://vllm.test/v1/"
        client.close()

    def test_timeout_is_applied(self, exporter: InMemorySpanExporter) -> None:
        client = traced_http_client(timeout=12.5)
        assert client.timeout.read == 12.5
        client.close()


class TestAsyncClient:
    async def test_traceparent_is_sent_inside_a_span(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        with job_trace("term_extraction"):
            async with traced_async_http_client(timeout=5.0, transport=transport) as client:
                response = await client.get("http://vllm.test/v1/models")

        assert response.json()["traceparent"] is not None

    async def test_no_traceparent_outside_a_span(self, exporter: InMemorySpanExporter) -> None:
        transport = httpx.MockTransport(_echo)
        async with traced_async_http_client(timeout=5.0, transport=transport) as client:
            response = await client.get("http://vllm.test/v1/models")
        assert response.json()["traceparent"] is None
