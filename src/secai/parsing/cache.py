"""Cache Docling's output per file hash.

Docling is the slow part of parsing (seconds per page on CPU); chunking and
waterfall extraction are milliseconds. Caching only Docling's blocks and
tables means changes to the fast logic are always re-run, while a large
prospectus is converted once. The key is the PDF's sha256, so an edited file
is never served stale output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from secai.config import ParsingSettings
from secai.parsing.docling_parser import Converter, DoclingResult, parse_pdf
from secai.parsing.models import Block, Table
from secai.parsing.pipeline import file_sha256


def cache_path(pdf: Path, cache_dir: Path, digest: str | None = None) -> Path:
    key = (digest or file_sha256(pdf))[:16]
    return cache_dir / f"{pdf.stem}.{key}.json"


def load_or_parse(
    pdf: Path,
    settings: ParsingSettings,
    cache_dir: Path,
    *,
    converter: Converter | None = None,
) -> tuple[DoclingResult, bool]:
    """Docling output for ``pdf`` and whether it came from the cache."""
    cached = cache_path(pdf, cache_dir)
    if cached.is_file():
        raw: dict[str, Any] = json.loads(cached.read_text(encoding="utf-8"))
        result = DoclingResult(
            blocks=[Block.model_validate(block) for block in raw["blocks"]],
            tables=[Table.model_validate(table) for table in raw["tables"]],
            page_count=raw["page_count"],
            fallback_tables=raw["fallback_tables"],
        )
        return result, True

    result = parse_pdf(pdf, settings, converter=converter)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps(
            {
                "blocks": [block.model_dump(mode="json") for block in result.blocks],
                "tables": [table.model_dump(mode="json") for table in result.tables],
                "page_count": result.page_count,
                "fallback_tables": result.fallback_tables,
            }
        ),
        encoding="utf-8",
    )
    return result, False
