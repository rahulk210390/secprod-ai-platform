"""Docling → blocks mapping and page batching, without running Docling's models.

Documents are built in code with docling-core, the same object model the
converter returns, so the mapping is tested exactly and hermetically.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from secai.config import ParsingSettings
from secai.parsing import docling_parser
from secai.parsing.docling_parser import (
    PAGE_LABEL,
    convert_document,
    count_pages,
    heading_level,
    page_labels,
    parse_pdf,
)
from secai.parsing.models import BlockKind

FIXTURE = Path(__file__).parents[1] / "fixtures" / "parsing" / "synthetic_prospectus.pdf"


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page,
        # Docling reports PDF boxes with a bottom-left origin.
        bbox=BoundingBox(l=72, t=700, r=540, b=600, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 0),
    )


def _table_data(rows: list[list[str]]) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


def _document(pages: tuple[int, ...] = (82, 83)) -> DoclingDocument:
    doc = DoclingDocument(name="synthetic")
    for page in pages:
        doc.add_page(page_no=page, size=Size(width=612, height=792))
    first, second = pages
    doc.add_table(
        data=_table_data([["Contents", "Page"], ["Risk Factors", "5"]]),
        prov=_prov(first),
        label=DocItemLabel.DOCUMENT_INDEX,
    )
    doc.add_heading("DESCRIPTION OF THE NOTES", level=1, prov=_prov(first))
    doc.add_heading("Priority of Payments", level=1, prov=_prov(first))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="On each  payment date,\n the servicer will pay:",
        prov=_prov(first),
    )
    group = doc.add_list_group()
    doc.add_list_item(
        "to the trustee, fees,", enumerated=True, marker="(1)", prov=_prov(first), parent=group
    )
    doc.add_list_item(
        "a bullet point", enumerated=False, marker="•", prov=_prov(first), parent=group
    )
    doc.add_text(
        label=DocItemLabel.PAGE_FOOTER,
        text="78",
        prov=_prov(first),
        content_layer=ContentLayer.FURNITURE,
    )
    doc.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="Table of Contents",
        prov=_prov(first),
        content_layer=ContentLayer.FURNITURE,
    )
    doc.add_table(
        data=_table_data([["Class", "Balance"], ["A-1", "$250,000,000"]]), prov=_prov(second)
    )
    doc.add_text(label=DocItemLabel.CAPTION, text="Table 1", prov=_prov(second))
    doc.add_text(label=DocItemLabel.TEXT, text="   ", prov=_prov(second))
    # Footer left in the body: the bare number that ends the page.
    doc.add_text(label=DocItemLabel.TEXT, text="79", prov=_prov(second))
    return doc


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "docling_level", "expected"),
    [
        ("DESCRIPTION OF THE NOTES", 1, 1),
        ("Priority of Payments", 1, 2),
        ("Sub point", 2, 3),
        ("CLASS A-1 NOTES", 1, 1),
        ("2025", 1, 2),
    ],
)
def test_heading_level(text: str, docling_level: int, expected: int) -> None:
    assert heading_level(text, docling_level) == expected


@pytest.mark.parametrize("label", ["78", "S-12", "A-1", "iv", "xii"])
def test_page_label_accepts_printed_numbers(label: str) -> None:
    assert PAGE_LABEL.match(label)


@pytest.mark.parametrize("text", ["Table of Contents", "$250,000", "78 of 90", "Class A-1"])
def test_page_label_rejects_prose(text: str) -> None:
    assert not PAGE_LABEL.match(text)


def test_page_labels_from_furniture_footers() -> None:
    assert page_labels(_document()) == {82: "78"}


def test_count_pages_on_fixture() -> None:
    assert count_pages(FIXTURE) == 5


# --------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------
def test_convert_document_maps_items_in_reading_order() -> None:
    batch = convert_document(_document())
    kinds = [(b.kind, b.text) for b in batch.blocks]

    assert kinds == [
        (BlockKind.HEADING, "DESCRIPTION OF THE NOTES"),
        (BlockKind.HEADING, "Priority of Payments"),
        (BlockKind.TEXT, "On each payment date, the servicer will pay:"),
        (BlockKind.LIST_ITEM, "to the trustee, fees,"),
        (BlockKind.LIST_ITEM, "a bullet point"),
        (BlockKind.TABLE, "Class | Balance\nA-1 | $250,000,000"),
        (BlockKind.CAPTION, "Table 1"),
    ]
    assert [b.index for b in batch.blocks] == list(range(7))


def test_convert_document_levels_markers_and_labels() -> None:
    blocks = convert_document(_document()).blocks
    assert [b.level for b in blocks[:2]] == [1, 2]
    assert blocks[3].marker == "(1)"
    assert blocks[4].marker is None  # bullets are not step markers
    # Page 82 labelled by its footer; page 83 by the trailing bare number.
    assert blocks[0].page_label == "78"
    assert blocks[5].page == 83 and blocks[5].page_label == "79"


def test_convert_document_skips_toc_and_running_headers() -> None:
    texts = " ".join(b.text for b in convert_document(_document()).blocks)
    assert "Risk Factors" not in texts
    assert "Table of Contents" not in texts


def test_convert_document_tables() -> None:
    batch = convert_document(_document(), first_block=10, first_table=3)
    (table,) = batch.tables
    assert table.index == 3
    assert table.rows == [["Class", "Balance"], ["A-1", "$250,000,000"]]
    assert table.source == "docling"
    assert table.page_label == "79"
    table_block = next(b for b in batch.blocks if b.kind is BlockKind.TABLE)
    assert table_block.table_index == 3
    assert batch.blocks[0].index == 10


class _Fallback:
    def __init__(self, rows: list[list[str]]) -> None:
        self.rows = rows
        self.calls: list[tuple[int, tuple[float, float, float, float]]] = []

    def extract(self, page_no: int, bbox: tuple[float, float, float, float]) -> list[list[str]]:
        self.calls.append((page_no, bbox))
        return self.rows


def _empty_table_document() -> DoclingDocument:
    doc = DoclingDocument(name="empty-table")
    doc.add_page(page_no=4, size=Size(width=612, height=792))
    doc.add_table(data=_table_data([["", ""], ["", ""]]), prov=_prov(4))
    return doc


def test_empty_docling_table_falls_back_to_pdfplumber() -> None:
    fallback = _Fallback([["Class", "Rate"], ["A", "4.50%"]])
    batch = convert_document(_empty_table_document(), fallback=fallback)  # type: ignore[arg-type]

    (table,) = batch.tables
    assert table.source == "pdfplumber"
    assert table.rows == [["Class", "Rate"], ["A", "4.50%"]]
    assert batch.fallback_tables == 1
    # Docling's bottom-left box is converted to pdfplumber's top-left origin.
    page_no, (left, top, right, bottom) = fallback.calls[0]
    assert page_no == 4
    assert (left, top, right, bottom) == (72, 792 - 700, 540, 792 - 600)


def test_empty_table_dropped_when_fallback_also_fails() -> None:
    batch = convert_document(_empty_table_document(), fallback=_Fallback([]))  # type: ignore[arg-type]
    assert batch.tables == []
    assert batch.fallback_tables == 1


def test_empty_table_dropped_without_fallback() -> None:
    batch = convert_document(_empty_table_document(), fallback=None)
    assert batch.tables == [] and batch.blocks == []


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------
class _FakeConverter:
    """Returns a one-heading document per batch and records the page ranges."""

    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []

    def convert(self, source: Path, *, page_range: tuple[int, int]) -> SimpleNamespace:
        self.ranges.append(page_range)
        first, last = page_range
        doc = DoclingDocument(name=source.name)
        for page in range(first, last + 1):
            doc.add_page(page_no=page, size=Size(width=612, height=792))
            doc.add_heading(f"SECTION ON PAGE {page}", level=1, prov=_prov(page))
        return SimpleNamespace(document=doc)


def test_parse_pdf_converts_in_page_batches() -> None:
    converter = _FakeConverter()
    result = parse_pdf(FIXTURE, ParsingSettings(batch_pages=2), converter=converter)

    assert converter.ranges == [(1, 2), (3, 4), (5, 5)]
    assert result.page_count == 5
    # Indices run on across batches.
    assert [b.index for b in result.blocks] == [0, 1, 2, 3, 4]
    assert [b.page for b in result.blocks] == [1, 2, 3, 4, 5]


def test_build_converter_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setitem(docling_parser._converters, False, sentinel)
    assert docling_parser.build_converter(ParsingSettings(do_ocr=False)) is sentinel
