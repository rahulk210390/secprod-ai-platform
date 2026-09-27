"""The Docling output cache: round trip, and invalidation by file content."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DoclingDocument,
    ProvenanceItem,
    Size,
)

from secai.config import ParsingSettings
from secai.parsing.cache import cache_path, load_or_parse

FIXTURE = Path(__file__).parents[1] / "fixtures" / "parsing" / "synthetic_prospectus.pdf"


class _CountingConverter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, source: Path, *, page_range: tuple[int, int]) -> SimpleNamespace:
        self.calls += 1
        doc = DoclingDocument(name=source.name)
        for page in range(page_range[0], page_range[1] + 1):
            doc.add_page(page_no=page, size=Size(width=612, height=792))
            prov = ProvenanceItem(
                page_no=page,
                bbox=BoundingBox(l=0, t=10, r=10, b=0, coord_origin=CoordOrigin.BOTTOMLEFT),
                charspan=(0, 0),
            )
            doc.add_heading(f"PAGE {page}", prov=prov)
        return SimpleNamespace(document=doc)


def test_second_load_comes_from_cache(tmp_path: Path) -> None:
    converter = _CountingConverter()
    settings = ParsingSettings(batch_pages=5)

    first, first_cached = load_or_parse(FIXTURE, settings, tmp_path, converter=converter)
    second, second_cached = load_or_parse(FIXTURE, settings, tmp_path, converter=converter)

    assert (first_cached, second_cached) == (False, True)
    assert converter.calls == 1
    assert second.blocks == first.blocks
    assert second.page_count == first.page_count == 5
    assert cache_path(FIXTURE, tmp_path).is_file()


def test_changed_file_is_not_served_stale(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    shutil.copyfile(FIXTURE, pdf)
    converter = _CountingConverter()
    settings = ParsingSettings(batch_pages=5)
    cache_dir = tmp_path / "cache"

    load_or_parse(pdf, settings, cache_dir, converter=converter)
    pdf.write_bytes(pdf.read_bytes() + b"\n% edited\n")
    _, cached = load_or_parse(pdf, settings, cache_dir, converter=converter)

    assert cached is False
    assert converter.calls == 2
    assert len(list(cache_dir.iterdir())) == 2
