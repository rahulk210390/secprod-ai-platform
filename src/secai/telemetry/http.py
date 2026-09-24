"""HTTP plumbing that carries trace context into vLLM.

vLLM emits its own server-side spans and ships them to the OTel Collector over
gRPC. Those spans only nest under the application trace if the request carries
W3C trace context, so every vLLM call goes through a client that injects
``traceparent`` (CLAUDE.md §6.4).
"""

from __future__ import annotations

import httpx
from opentelemetry.propagate import inject


def inject_trace_context(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``headers`` with W3C ``traceparent``/``tracestate`` added."""
    carrier: dict[str, str] = dict(headers) if headers else {}
    inject(carrier)
    return carrier


def _inject_on_request(request: httpx.Request) -> None:
    """httpx event hook: stamp trace context onto every outgoing request."""
    carrier: dict[str, str] = {}
    inject(carrier)
    for key, value in carrier.items():
        request.headers[key] = value


def traced_http_client(
    *,
    timeout: float,
    base_url: str | None = None,
    **kwargs: object,
) -> httpx.Client:
    """An httpx client that propagates trace context on every request.

    Pass this to the OpenAI SDK as ``http_client`` so vLLM server spans join
    the application trace.
    """
    return httpx.Client(
        base_url=base_url or "",
        timeout=timeout,
        event_hooks={"request": [_inject_on_request]},
        **kwargs,  # type: ignore[arg-type]
    )


def traced_async_http_client(
    *,
    timeout: float,
    base_url: str | None = None,
    **kwargs: object,
) -> httpx.AsyncClient:
    """Async counterpart of :func:`traced_http_client`."""

    async def _inject_async(request: httpx.Request) -> None:
        _inject_on_request(request)

    return httpx.AsyncClient(
        base_url=base_url or "",
        timeout=timeout,
        event_hooks={"request": [_inject_async]},
        **kwargs,  # type: ignore[arg-type]
    )
