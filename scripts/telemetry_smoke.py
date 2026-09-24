"""JOB-01 smoke test: send one traced job run to Langfuse.

Run it against the live stack::

    uv run python scripts/telemetry_smoke.py            # spans only
    uv run python scripts/telemetry_smoke.py --with-llm # also call vLLM

It emits one root trace ``job.telemetry_smoke`` with the canonical child spans,
a generation, scores, a deliberate error span, and a payload carrying synthetic
borrower data that must come out masked. The trace ID is printed so it can be
recorded in PROGRESS.md.

No real deal or borrower data: every value below is synthetic.
"""

from __future__ import annotations

import argparse
import sys
import time

from secai.config import get_settings
from secai.telemetry import (
    SPAN_CHUNK,
    SPAN_PARSE,
    SPAN_PERSIST,
    SPAN_VALIDATE,
    current_trace_id,
    init_telemetry,
    inject_trace_context,
    job_span,
    job_trace,
    llm_generation,
    score_trace,
    shutdown_telemetry,
    trace_url,
)
from secai.telemetry.attributes import (
    SCORE_CRITICAL_FIELD_F1,
    SCORE_FIELD_F1,
    SCORE_LATENCY_S,
    SCORE_NEEDS_REVIEW,
    SCORE_VALIDATION_PASS,
)

# Synthetic borrower records — these must appear masked in Langfuse.
SYNTHETIC_LOANS = [
    {
        "borrower_name": "Jane Doe",
        "loan_number": "L-99887766",
        "email": "jane.doe@example.com",
        "current_balance": 312_500,
    },
    {
        "borrower_name": "John Roe",
        "loan_number": "L-11223344",
        "email": "john.roe@example.com",
        "current_balance": 415_000,
    },
]

# Deal-level terms — these must appear intact.
SYNTHETIC_TERMS = {
    "deal_name": "SYNTH 2024-1",
    "asset_class": "RMBS",
    "tranches": [
        {"class": "A", "original_balance": 250_000_000, "coupon_margin": 1.35},
        {"class": "B", "original_balance": 50_000_000, "coupon_margin": 2.75},
    ],
    "oc_test_threshold": 115.5,
}


def _call_vllm(prompt: str) -> str:
    """One real vLLM call, with trace context propagated so its spans nest."""
    from openai import OpenAI

    from secai.telemetry.http import traced_http_client

    settings = get_settings().vllm
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        http_client=traced_http_client(timeout=settings.timeout_s),
    )
    response = client.chat.completions.create(
        model=settings.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.temperature,
        max_tokens=128,
    )
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="also make a real vLLM call (requires the vllm service to be up)",
    )
    args = parser.parse_args()

    settings = get_settings()
    telemetry = init_telemetry()

    if not telemetry.enabled:
        print(
            "Tracing is disabled — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "in .env, then re-run.",
            file=sys.stderr,
        )
        return 1

    started = time.perf_counter()

    with job_trace(
        "telemetry_smoke",
        deal_id="SYNTH-2024-1",
        asset_class="RMBS",
        doc_type="prospectus",
        prompt_version="smoke-v1",
        model=settings.vllm.model,
        session_id="smoke-run",
        user_id="smoke-analyst",
        tags=["job-01", "smoke"],
        metadata={"purpose": "JOB-01 acceptance"},
        input_data={"document": "synthetic_prospectus.pdf"},
    ) as root:
        with job_span(SPAN_PARSE, input_data={"pages": 120}) as span:
            span.update(output={"blocks": 842, "tables": 17})

        with job_span(SPAN_CHUNK) as span:
            span.update(output={"sections": ["Priority of Payments", "Trigger Events"]})

        # Borrower data goes in deliberately: it must come back masked.
        with job_span(SPAN_VALIDATE, input_data={"loans": SYNTHETIC_LOANS}) as span:
            span.update(output={"terms": SYNTHETIC_TERMS, "validation_pass": True})

        if args.with_llm:
            prompt = "Reply with the single word: ok"
            with llm_generation(
                model=settings.vllm.model,
                model_parameters={"temperature": settings.vllm.temperature},
                input_data=prompt,
            ) as span:
                # Header check: proves trace context leaves the process.
                span.update(metadata={"traceparent_sent": "traceparent" in inject_trace_context()})
                span.update(output=_call_vllm(prompt))

        # An error span, to prove exceptions are recorded rather than swallowed.
        try:
            with job_span(SPAN_PERSIST):
                raise RuntimeError("synthetic failure for the smoke test")
        except RuntimeError:
            pass

        root.update(output={"status": "complete"})

        score_trace(SCORE_VALIDATION_PASS, True)
        score_trace(SCORE_FIELD_F1, 0.93)
        score_trace(SCORE_CRITICAL_FIELD_F1, 0.97)
        score_trace(SCORE_NEEDS_REVIEW, False)
        score_trace(SCORE_LATENCY_S, round(time.perf_counter() - started, 3))

        trace_id = current_trace_id()
        url = trace_url()

    telemetry.flush()
    shutdown_telemetry()

    print(f"trace_id: {trace_id}")
    print(f"url:      {url}")
    print()
    print("In Langfuse, confirm on this trace:")
    print("  - root span 'job.telemetry_smoke' with the secai.* attributes")
    print("  - child spans: parse, chunk, validate, persist (persist = ERROR)")
    if args.with_llm:
        print("  - a generation with model, prompt, completion and token usage")
        print("  - vLLM server spans nested under the same trace ID")
    print("  - scores: validation_pass, field_f1, critical_field_f1, needs_review, latency_s")
    print("  - borrower_name / loan_number / email are REDACTED")
    print("  - deal terms (tranche balances, oc_test_threshold) are intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
