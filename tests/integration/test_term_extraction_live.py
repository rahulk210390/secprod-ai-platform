"""JOB-04 against live vLLM on a real public prospectus.

These assert the pipeline's guarantees, not the model's accuracy (that is the
eval's job, ``make eval JOB=04``):

* the run completes and produces a validation report;
* every value the model returned is either cited to a page or flagged;
* the priority of payments comes from JOB-03, with every step cited;
* no numeric field was produced without passing the deterministic parsers.

Needs vLLM up and the Ford sample fetched and parsed (the Docling cache is
reused, so the parse is instant after the first time).
"""

from __future__ import annotations

import httpx
import pytest

from secai.config import REPO_ROOT, get_settings
from secai.jobs.term_extraction.pipeline import run_term_extraction
from secai.validation.deal_terms import populated_fields

pytestmark = pytest.mark.integration

SAMPLES = REPO_ROOT / "data" / "samples"
FORD = SAMPLES / "ford_auto_2025.pdf"


def _vllm_up() -> bool:
    base = get_settings().vllm.base_url.rstrip("/").removesuffix("/v1")
    try:
        return httpx.get(f"{base}/health", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    if not FORD.is_file():
        pytest.skip("ford_auto_2025.pdf not fetched; run scripts/fetch_samples.py")
    if not _vllm_up():
        pytest.skip("vLLM is not reachable; start it with `make up-llm`")
    return run_term_extraction(FORD, deal_id="ford_auto_2025", parse_cache_dir=SAMPLES / ".cache")


def test_run_completes_with_calls_and_a_trace(result) -> None:  # type: ignore[no-untyped-def]
    assert result.llm_calls >= 3
    assert result.tokens > 0
    assert result.report is not None


def test_every_value_is_cited_or_flagged(result) -> None:  # type: ignore[no-untyped-def]
    flagged = {i.field for i in result.report.issues if i.code == "uncited"}
    for path, citations in populated_fields(result.terms):
        key = path.rsplit(".", 1)[-1]
        assert key in citations or path in flagged, f"{path} neither cited nor flagged"


def test_priority_of_payments_is_job_03s_verified_waterfall(result) -> None:  # type: ignore[no-untyped-def]
    steps = result.terms.priority_of_payments
    assert len(steps) == 11
    assert steps[0].page == 82 and steps[0].page_label == "78"
    assert [s.ordinal for s in steps] == list(range(1, 12))


def test_numbers_passed_the_deterministic_parsers(result) -> None:  # type: ignore[no-untyped-def]
    # A tranche only exists if its balance and coupon parsed; rates stay in percent units.
    for tranche in result.terms.tranches:
        assert tranche.original_balance > 0
        if tranche.fixed_rate_pct is not None:
            assert tranche.fixed_rate_pct < 25
