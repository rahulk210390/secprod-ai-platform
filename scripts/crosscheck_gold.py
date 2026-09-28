"""Check parsing gold files against an independent PDF text extraction.

    uv run python scripts/crosscheck_gold.py                  # every gold file
    uv run python scripts/crosscheck_gold.py ford_auto_2025

The gold files record what Docling found. This checks it with a second,
unrelated engine (pdfplumber): every step's opening words must appear in the
PDF text of the waterfall's pages, in order. Two-column pages (prospectus
summaries) defeat a plain left-to-right read, so each page is also read as
left column then right column, and either reading may pass.

A waterfall that fails here is either a parser error or a gold typo; both need
a human look before the gold file is trusted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import pdfplumber
from pdfplumber.pdf import PDF

from secai.config import REPO_ROOT

GOLD = REPO_ROOT / "data" / "gold" / "parsing"
SAMPLES = REPO_ROOT / "data" / "samples"
PROBE_CHARS = 35
PAGES_AFTER_START = 3

_SPACES = re.compile(r"\s+")
# Curly quotes and em/en dashes, which the two engines render differently.
_TRANSLATE = str.maketrans(
    {chr(0x2019): "'", chr(0x201C): '"', chr(0x201D): '"', chr(0x2014): "-", chr(0x2013): "-"}
)


def norm(text: str) -> str:
    return _SPACES.sub(" ", text.translate(_TRANSLATE)).strip().lower()


def readings(pdf: PDF, first: int) -> list[str]:
    """The pages' text read straight across, and read column by column."""
    pages = [
        pdf.pages[n - 1] for n in range(first, min(first + PAGES_AFTER_START, len(pdf.pages)) + 1)
    ]
    across = " ".join(page.extract_text() or "" for page in pages)
    columns = []
    for page in pages:
        half = page.width / 2
        columns.append(page.crop((0, 0, half, page.height)).extract_text() or "")
        columns.append(page.crop((half, 0, page.width, page.height)).extract_text() or "")
    return [norm(across), norm(" ".join(columns))]


def missing_steps(text: str, steps: list[str]) -> list[str]:
    """Steps whose opening words are absent, or present only out of order."""
    position, problems = -1, []
    for ordinal, step in enumerate(steps, start=1):
        probe = norm(step)[:PROBE_CHARS]
        found = text.find(probe, position + 1)
        if found == -1:
            problems.append(f"step {ordinal} not found in order: {probe!r}")
        else:
            position = found
    return problems


def check_sample(sample_id: str) -> bool:
    gold: dict[str, Any] = json.loads((GOLD / f"{sample_id}.json").read_text(encoding="utf-8"))
    pdf_path = SAMPLES / f"{sample_id}.pdf"
    if not pdf_path.is_file():
        print(f"SKIP {sample_id}: {pdf_path.name} not fetched")
        return True
    ok = True
    with pdfplumber.open(pdf_path) as pdf:
        for waterfall in gold["waterfalls"]:
            results = [
                missing_steps(text, waterfall["steps"])
                for text in readings(pdf, waterfall["start_page"])
            ]
            problems = min(results, key=len)
            layout = " (column reading)" if not problems and results[0] else ""
            print(
                f"{'OK ' if not problems else 'BAD'} {sample_id} p{waterfall['start_page']} "
                f"{len(waterfall['steps']):>2} steps{layout}  {waterfall['title_contains'][:60]}"
            )
            for problem in problems:
                print(f"      {problem}")
            ok &= not problems
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("samples", nargs="*", help="sample ids (default: every gold file)")
    args = parser.parse_args()
    samples = args.samples or sorted(path.stem for path in GOLD.glob("*.json"))
    results = [check_sample(sample_id) for sample_id in samples]
    print("\nall waterfalls confirmed" if all(results) else "\nMISMATCHES FOUND")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
