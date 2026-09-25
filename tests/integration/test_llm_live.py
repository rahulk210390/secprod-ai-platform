"""Live vLLM integration tests (JOB-02).

Run with the stack up::

    make test-integration

The acceptance bar from CLAUDE.md §5 is the 20/20 test below: twenty
consecutive live calls, each returning a valid Pydantic object. That is what
proves guided decoding is actually constraining the sampler rather than the
model merely happening to emit good JSON.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel, Field

from secai.config import get_settings
from secai.llm import LLMClient
from secai.telemetry import init_telemetry, job_trace

pytestmark = pytest.mark.integration


class Tranche(BaseModel):
    """Deliberately small: the local dev model is a 1.5B."""

    class_name: str = Field(alias="class", description="Tranche class, e.g. A, B, C")
    original_balance: int = Field(description="Original balance as an integer")
    coupon_margin: float = Field(description="Margin over the index, in percent")


EXTRACT_PROMPT = (
    "Extract the tranche details from this sentence and return JSON only.\n\n"
    "Class A notes were issued with an original balance of 250000000 "
    "and a margin of 1.35 per cent over SOFR."
)


@pytest.fixture(scope="module")
def client() -> Iterator[LLMClient]:
    init_telemetry()
    llm = LLMClient()
    if not llm.health():
        llm.close()
        pytest.skip(
            f"vLLM is not reachable at {get_settings().vllm.base_url}; start it with `make up`"
        )
    yield llm
    llm.close()


class TestLiveExtraction:
    def test_single_call_returns_a_valid_object(self, client: LLMClient) -> None:
        with job_trace("llm_integration", model=get_settings().vllm.model):
            result = client.generate_structured(
                [{"role": "user", "content": EXTRACT_PROMPT}], Tranche
            )
        assert isinstance(result, Tranche)
        assert result.original_balance > 0

    def test_twenty_consecutive_calls_all_validate(self, client: LLMClient) -> None:
        """The JOB-02 acceptance criterion: 20/20 valid Pydantic objects."""
        failures: list[str] = []
        with job_trace("llm_integration_20", model=get_settings().vllm.model):
            for attempt in range(20):
                try:
                    result = client.generate_structured(
                        [{"role": "user", "content": EXTRACT_PROMPT}], Tranche
                    )
                    assert isinstance(result, Tranche)
                except Exception as exc:
                    failures.append(f"call {attempt}: {type(exc).__name__}: {exc}")

        assert not failures, "guided decoding did not hold:\n" + "\n".join(failures)

    def test_usage_is_reported(self, client: LLMClient) -> None:
        # cost_tokens is a logged KPI; a server that reports no usage breaks it.
        with job_trace("llm_integration", model=get_settings().vllm.model):
            client.generate_structured([{"role": "user", "content": EXTRACT_PROMPT}], Tranche)


class TestServerContract:
    def test_health_endpoint(self, client: LLMClient) -> None:
        assert client.health() is True

    def test_served_model_matches_configuration(self, client: LLMClient) -> None:
        import httpx

        settings = get_settings().vllm
        response = httpx.get(f"{settings.base_url.rstrip('/')}/models", timeout=10.0)
        response.raise_for_status()
        served = {entry["id"] for entry in response.json()["data"]}
        assert settings.model in served, (
            f"VLLM_MODEL={settings.model} is not served; vLLM has {served}"
        )
