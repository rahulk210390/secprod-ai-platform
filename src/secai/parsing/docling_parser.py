"""Docling → :mod:`secai.parsing.models`.

Docling runs layout and table-structure models on every page it converts and
holds a whole call's pages in memory, so a prospectus is converted a batch of
pages at a time (``SECAI_PARSING_BATCH_PAGES``). Page numbers stay absolute
across batches: converting pages 81-100 reports them as 81-100.

Mapping rules:

* section headers and titles become HEADING blocks; all-caps headings
  ("DESCRIPTION OF THE NOTES") are level 1, others level 2, which matches how
  US prospectuses nest them, since Docling reports nearly every header as
  level 1;
* list items keep their enumerator ("(3)") in ``Block.marker``; bullets are
  dropped;
* page footers that look like a page number become ``page_label`` for every
  block on that page; other furniture (running headers) is discarded;
* the table of contents is skipped: its entries are not headings and would
  otherwise create phantom sections.
"""

from __future__ import annotations

import gc
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import ListItem, SectionHeaderItem, TextItem, TitleItem
from docling_core.types.doc.labels import DocItemLabel

from secai.config import ParsingSettings
from secai.parsing.models import Block, BlockKind, Table
from secai.parsing.pdfplumber_tables import TableFallback

logger = logging.getLogger(__name__)

PAGE_LABEL = re.compile(r"^(?:[A-Z]{1,3}-)?\d{1,4}$|^[ivxlcdm]{1,7}$", re.IGNORECASE)
# Bullet glyphs (middle dot, bullets, squares, dashes) are not step markers.
_BULLETS: Final = frozenset(
    {"", "-", "*", "o"}
    | {chr(code) for code in (0x00B7, 0x2022, 0x25CF, 0x25CB, 0x25AA, 0x25A0, 0x2013, 0x2014)}
)
_SKIPPED_LABELS: Final = frozenset(
    {DocItemLabel.DOCUMENT_INDEX, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
)
_WHITESPACE = re.compile(r"\s+")


class Converter(Protocol):
    """The slice of ``docling.DocumentConverter`` this module uses."""

    def convert(self, source: Path, *, page_range: tuple[int, int]) -> Any: ...


_converters: dict[bool, Converter] = {}


def build_converter(settings: ParsingSettings) -> Converter:
    """A Docling converter tuned for low memory. Cached: loading the models takes ~25 s."""
    cached = _converters.get(settings.do_ocr)
    if cached is not None:
        return cached

    # Imported here: pulling in docling's pipeline loads torch, which unit
    # tests of the mapping never need.
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=settings.do_ocr,
        do_table_structure=True,
        # One page at a time through each model keeps the peak near 1 GB.
        layout_batch_size=1,
        table_batch_size=1,
        ocr_batch_size=1,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=settings.num_threads),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    converter.initialize_pipeline(InputFormat.PDF)
    _converters[settings.do_ocr] = converter
    return converter


def count_pages(path: Path) -> int:
    import pypdfium2

    pdf = pypdfium2.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def heading_level(text: str, docling_level: int = 1) -> int:
    letters = [char for char in text if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        return 1
    return max(2, docling_level + 1)


def page_labels(document: DoclingDocument) -> dict[int, str]:
    """Printed page numbers, from items Docling labels as page footers.

    Footers normally sit in the furniture layer, but the label is what counts:
    one left in the body layer is still a footer.
    """
    labels: dict[int, str] = {}
    layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
    for item, _ in document.iterate_items(included_content_layers=layers):
        if not isinstance(item, TextItem) or item.label is not DocItemLabel.PAGE_FOOTER:
            continue
        text = clean(item.text)
        if item.prov and PAGE_LABEL.match(text):
            labels[item.prov[0].page_no] = text
    return labels


def _marker(item: ListItem) -> str | None:
    marker = clean(item.marker or "")
    return None if marker in _BULLETS else marker


def _table_rows(item: TableItem) -> list[list[str]]:
    return [[clean(cell.text) for cell in row] for row in item.data.grid]


def _bbox(document: DoclingDocument, item: TableItem) -> tuple[float, float, float, float] | None:
    if not item.prov:
        return None
    prov = item.prov[0]
    page = document.pages.get(prov.page_no)
    if page is None:
        return None
    box = prov.bbox.to_top_left_origin(page_height=page.size.height)
    return (box.l, box.t, box.r, box.b)


@dataclass
class Batch:
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    fallback_tables: int = 0


def convert_document(
    document: DoclingDocument,
    *,
    first_block: int = 0,
    first_table: int = 0,
    fallback: TableFallback | None = None,
) -> Batch:
    """Map one converted batch onto blocks and tables, numbering from the given offsets."""
    labels = page_labels(document)
    batch = Batch()
    # Pages whose footer Docling left in the body: the page's last bare number.
    trailing: dict[int, tuple[int, str]] = {}
    pending: list[dict[str, Any]] = []

    for item, _ in document.iterate_items():
        label = getattr(item, "label", None)
        provenance = getattr(item, "prov", None)
        if label in _SKIPPED_LABELS or not provenance:
            continue
        page: int = provenance[0].page_no

        if isinstance(item, TableItem):
            rows = _table_rows(item)
            source = "docling"
            if not any(cell for row in rows for cell in row) and fallback is not None:
                bbox = _bbox(document, item)
                rows = fallback.extract(page, bbox) if bbox else []
                source = "pdfplumber"
                batch.fallback_tables += 1
            if not any(cell for row in rows for cell in row):
                continue
            caption = clean(item.caption_text(document)) or None
            pending.append(
                {
                    "kind": BlockKind.TABLE,
                    "page": page,
                    "rows": rows,
                    "caption": caption,
                    "source": source,
                }
            )
            continue

        if not isinstance(item, TextItem):
            continue  # pictures, groups
        text = clean(item.text)
        if not text:
            continue

        if isinstance(item, TitleItem):
            pending.append({"kind": BlockKind.HEADING, "page": page, "text": text, "level": 1})
        elif isinstance(item, SectionHeaderItem):
            pending.append(
                {
                    "kind": BlockKind.HEADING,
                    "page": page,
                    "text": text,
                    "level": heading_level(text, item.level),
                }
            )
        elif isinstance(item, ListItem):
            pending.append(
                {"kind": BlockKind.LIST_ITEM, "page": page, "text": text, "marker": _marker(item)}
            )
        else:
            kind = BlockKind.CAPTION if label is DocItemLabel.CAPTION else BlockKind.TEXT
            if kind is BlockKind.TEXT and PAGE_LABEL.match(text):
                trailing[page] = (len(pending), text)
            pending.append({"kind": kind, "page": page, "text": text})

    # A bare number that is the last thing on its page is that page's footer.
    last_on_page: dict[int, int] = {}
    for position, entry in enumerate(pending):
        last_on_page[entry["page"]] = position
    dropped: set[int] = set()
    for page, (position, text) in trailing.items():
        if page not in labels and last_on_page.get(page) == position:
            labels[page] = text
            dropped.add(position)

    per_page_tables: dict[int, int] = defaultdict(int)
    for position, entry in enumerate(pending):
        if position in dropped:
            continue
        page = entry["page"]
        index = first_block + len(batch.blocks)
        if entry["kind"] is BlockKind.TABLE:
            table_index = first_table + len(batch.tables)
            per_page_tables[page] += 1
            table = Table(
                index=table_index,
                page=page,
                page_label=labels.get(page),
                rows=entry["rows"],
                caption=entry["caption"],
                source=entry["source"],
            )
            batch.tables.append(table)
            batch.blocks.append(
                Block(
                    index=index,
                    kind=BlockKind.TABLE,
                    text="\n".join(" | ".join(row) for row in entry["rows"]),
                    page=page,
                    page_label=labels.get(page),
                    table_index=table_index,
                )
            )
            continue
        batch.blocks.append(
            Block(
                index=index,
                kind=entry["kind"],
                text=entry["text"],
                page=page,
                page_label=labels.get(page),
                level=entry.get("level"),
                marker=entry.get("marker"),
            )
        )
    return batch


@dataclass
class DoclingResult:
    blocks: list[Block]
    tables: list[Table]
    page_count: int
    fallback_tables: int


def parse_pdf(
    path: Path,
    settings: ParsingSettings,
    *,
    converter: Converter | None = None,
) -> DoclingResult:
    """Convert a PDF batch by batch and concatenate the results."""
    page_count = count_pages(path)
    active = converter if converter is not None else build_converter(settings)
    blocks: list[Block] = []
    tables: list[Table] = []
    fallback_tables = 0

    with TableFallback(path) as fallback:
        for first in range(1, page_count + 1, settings.batch_pages):
            last = min(first + settings.batch_pages - 1, page_count)
            started = time.perf_counter()
            result = active.convert(path, page_range=(first, last))
            batch = convert_document(
                result.document,
                first_block=len(blocks),
                first_table=len(tables),
                fallback=fallback if settings.table_fallback else None,
            )
            blocks.extend(batch.blocks)
            tables.extend(batch.tables)
            fallback_tables += batch.fallback_tables
            # Docling keeps page images and model outputs on the result.
            del result
            gc.collect()
            logger.info(
                "parsed %s pages %d-%d/%d in %.1fs",
                path.name,
                first,
                last,
                page_count,
                time.perf_counter() - started,
            )

    return DoclingResult(
        blocks=blocks, tables=tables, page_count=page_count, fallback_tables=fallback_tables
    )
