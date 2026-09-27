"""JOB-03 acceptance: parse the public EDGAR samples and check waterfall order.

    uv run python scripts/fetch_samples.py          # once: download + print to PDF
    uv run pytest tests/integration/test_parsing_samples.py -m integration

Docling is slow on CPU (~1-3 s per page, ~1,600 pages across the five
samples), so its raw output is cached under ``data/samples/.cache`` keyed by
the PDF's hash. Sections and waterfalls are always recomputed from that
cache, so a change to the chunking or waterfall logic is tested on every run;
delete the cache to re-run Docling itself.

Gold files in ``data/gold/parsing`` hold, per sample, the waterfalls a human
confirmed (title, page and the opening words of every step, in order) and the
canonical sections that must be found. A gold file with ``"reviewed": false``
is a draft and is not enforced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from secai.config import REPO_ROOT, get_settings
from secai.parsing.cache import load_or_parse
from secai.parsing.docling_parser import DoclingResult, parse_pdf
from secai.parsing.models import Waterfall
from secai.parsing.pipeline import analyse

pytestmark = pytest.mark.integration

SAMPLES = REPO_ROOT / "data" / "samples"
CACHE = SAMPLES / ".cache"
GOLD = REPO_ROOT / "data" / "gold" / "parsing"
MANIFEST = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
SAMPLE_IDS = [sample["id"] for sample in MANIFEST["samples"]]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "parsing" / "synthetic_prospectus.pdf"

_SPACES = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text).strip().lower()


def _parsed(pdf: Path) -> DoclingResult:
    """Docling output for ``pdf``, from cache when the file is unchanged."""
    result, _ = load_or_parse(pdf, get_settings().parsing, CACHE)
    return result


def _matches(waterfall: Waterfall, expected: dict[str, Any]) -> bool:
    return waterfall.start_page == expected["start_page"] and _norm(
        expected["title_contains"]
    ) in _norm(waterfall.title)


def test_synthetic_fixture_through_real_docling() -> None:
    """The committed fixture, end to end through Docling's real models."""
    result = parse_pdf(FIXTURE, get_settings().parsing)
    _, sections, waterfalls = analyse(result)

    assert result.page_count == 5
    assert "priority_of_payments" in {s.canonical for s in sections}
    assert any(t.rows and t.rows[0][0] == "Class" for t in result.tables)

    numbered = next(w for w in waterfalls if w.steps[0].marker == "(1)")
    assert [s.ordinal for s in numbered.steps] == list(range(1, 9))
    assert numbered.steps[4].page == 3 and numbered.steps[5].page == 4
    assert numbered.steps[0].page_label == "S-2"

    prose = next(w for w in waterfalls if w.steps[0].marker == "First")
    assert [s.marker for s in prose.steps] == ["First", "Second", "Third", "Fourth", "Fifth"]


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_sample_waterfalls_match_gold(sample_id: str) -> None:
    pdf = SAMPLES / f"{sample_id}.pdf"
    if not pdf.is_file():
        pytest.skip(f"{pdf.name} not fetched; run scripts/fetch_samples.py")
    gold_path = GOLD / f"{sample_id}.json"
    if not gold_path.is_file():
        pytest.fail(f"no gold file for {sample_id}")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if not gold.get("reviewed"):
        pytest.skip(f"gold for {sample_id} is an unreviewed draft")

    result = _parsed(pdf)
    _, sections, waterfalls = analyse(result)

    assert result.page_count == gold["page_count"]
    found_sections = {s.canonical for s in sections if s.canonical}
    missing = set(gold["sections"]) - found_sections
    assert not missing, f"canonical sections not found: {sorted(missing)}"

    for expected in gold["waterfalls"]:
        candidates = [w for w in waterfalls if _matches(w, expected)]
        assert candidates, (
            f"waterfall {expected['title_contains']!r} on page {expected['start_page']} "
            f"not found; got {[(w.title[:60], w.start_page) for w in waterfalls]}"
        )
        waterfall = candidates[0]
        actual = [_norm(step.text) for step in waterfall.steps]
        wanted = [_norm(prefix) for prefix in expected["steps"]]
        assert len(actual) == len(wanted), (
            f"{expected['title_contains']}: {len(actual)} steps, expected {len(wanted)}"
        )
        for ordinal, (text, prefix) in enumerate(zip(actual, wanted, strict=True), start=1):
            assert text.startswith(prefix), f"step {ordinal}: {text[:80]!r} != {prefix!r}"
