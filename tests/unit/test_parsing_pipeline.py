"""The traced parse pipeline: spans, attributes, errors, and the PDF fallback reader.

Docling's models are replaced by a fake converter that returns a hand-built
document for the synthetic fixture PDF, so these run offline in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from secai.config import Settings
from secai.parsing import JOB_NAME, parse_document, run_parsing_job
from secai.parsing.pdfplumber_tables import TableFallback
from secai.telemetry import SPAN_CHUNK, SPAN_PARSE, get_telemetry, job_span_name
from secai.telemetry.attributes import (
    ATTR_CHUNK_SECTION_COUNT,
    ATTR_CHUNK_WATERFALL_COUNT,
    ATTR_DEAL_ID,
    ATTR_DOC_SHA256,
    ATTR_DOC_TYPE,
    ATTR_PARSE_BATCH_PAGES,
    ATTR_PARSE_PAGE_COUNT,
    ATTR_PARSE_PARSER,
    ATTR_PARSE_TABLE_COUNT,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "parsing" / "synthetic_prospectus.pdf"


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=72, t=700, r=540, b=100, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 0),
    )


class _WaterfallConverter:
    """Pages 3-4 carry a three-step waterfall; every page gets a footer label."""

    def convert(self, source: Path, *, page_range: tuple[int, int]) -> SimpleNamespace:
        doc = DoclingDocument(name=source.name)
        for page in range(page_range[0], page_range[1] + 1):
            doc.add_page(page_no=page, size=Size(width=612, height=792))
            if page == 3:
                doc.add_heading("DESCRIPTION OF THE NOTES", prov=_prov(page))
                doc.add_heading("Priority of Payments", prov=_prov(page))
                group = doc.add_list_group()
                doc.add_list_item("to the trustee,", marker="(1)", prov=_prov(page), parent=group)
                doc.add_list_item("to the servicer,", marker="(2)", prov=_prov(page), parent=group)
            if page == 4:
                group = doc.add_list_group()
                doc.add_list_item("to the residual.", marker="(3)", prov=_prov(page), parent=group)
            if page > 1:
                doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=f"S-{page - 1}", prov=_prov(page))
        return SimpleNamespace(document=doc)


class _FailingConverter:
    def convert(self, source: Path, *, page_range: tuple[int, int]) -> SimpleNamespace:
        raise RuntimeError("layout model crashed")


def _spans(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    get_telemetry().flush()
    return {span.name: span for span in exporter.get_finished_spans()}


@pytest.fixture
def small_batches() -> Settings:
    settings = Settings()
    settings.parsing.batch_pages = 2
    return settings


def test_parse_document_extracts_waterfall(small_batches: Settings) -> None:
    document = parse_document(FIXTURE, settings=small_batches, converter=_WaterfallConverter())

    assert document.source == "synthetic_prospectus.pdf"
    assert document.page_count == 5
    assert len(document.sha256) == 64
    (waterfall,) = document.waterfalls
    assert [s.text for s in waterfall.steps] == [
        "to the trustee,",
        "to the servicer,",
        "to the residual.",
    ]
    assert [s.page_label for s in waterfall.steps] == ["S-2", "S-2", "S-3"]
    assert {s.canonical for s in document.sections} >= {
        "description_of_notes",
        "priority_of_payments",
    }


def test_parse_and_chunk_spans_carry_measurements(
    exporter: InMemorySpanExporter, small_batches: Settings
) -> None:
    parse_document(FIXTURE, settings=small_batches, converter=_WaterfallConverter())
    spans = _spans(exporter)

    parse = spans[SPAN_PARSE]
    assert parse.attributes is not None
    assert parse.attributes[ATTR_PARSE_PARSER] == "docling"
    assert parse.attributes[ATTR_PARSE_PAGE_COUNT] == 5
    assert parse.attributes[ATTR_PARSE_TABLE_COUNT] == 0
    assert parse.attributes[ATTR_PARSE_BATCH_PAGES] == 2
    assert len(str(parse.attributes[ATTR_DOC_SHA256])) == 64

    chunk = spans[SPAN_CHUNK]
    assert chunk.attributes is not None
    assert chunk.attributes[ATTR_CHUNK_SECTION_COUNT] == 2
    assert chunk.attributes[ATTR_CHUNK_WATERFALL_COUNT] == 1


def test_trace_never_carries_the_local_path(
    exporter: InMemorySpanExporter, small_batches: Settings
) -> None:
    parse_document(FIXTURE, settings=small_batches, converter=_WaterfallConverter())
    payload = json.dumps([dict(span.attributes or {}) for span in _spans(exporter).values()])
    assert "synthetic_prospectus.pdf" in payload
    assert str(FIXTURE.parent) not in payload


def test_run_parsing_job_is_one_root_trace(
    exporter: InMemorySpanExporter, small_batches: Settings
) -> None:
    document = run_parsing_job(
        FIXTURE, deal_id="SYN-2099-1", settings=small_batches, converter=_WaterfallConverter()
    )
    spans = _spans(exporter)

    root = spans[job_span_name(JOB_NAME)]
    assert root.parent is None
    # The parse records which trace produced it.
    assert document.trace_id == format(root.context.trace_id, "032x")
    assert root.attributes is not None
    assert root.attributes[ATTR_DEAL_ID] == "SYN-2099-1"
    assert root.attributes[ATTR_DOC_TYPE] == "prospectus"
    for child in (SPAN_PARSE, SPAN_CHUNK):
        parent = spans[child].parent
        assert parent is not None and parent.span_id == root.context.span_id


def test_converter_failure_marks_spans_error_and_reraises(
    exporter: InMemorySpanExporter, small_batches: Settings
) -> None:
    with pytest.raises(RuntimeError, match="layout model crashed"):
        run_parsing_job(FIXTURE, settings=small_batches, converter=_FailingConverter())
    spans = _spans(exporter)
    assert spans[SPAN_PARSE].status.status_code is StatusCode.ERROR
    assert spans[job_span_name(JOB_NAME)].status.status_code is StatusCode.ERROR


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "absent.pdf", converter=_WaterfallConverter())


# --------------------------------------------------------------------------
# pdfplumber fallback on a real (synthetic) PDF
# --------------------------------------------------------------------------
def test_table_fallback_reads_the_ruled_table() -> None:
    with TableFallback(FIXTURE) as fallback:
        # The whole of page 2 contains exactly one ruled table: the note classes.
        rows = fallback.extract(2, (0, 0, 612, 792))
    assert rows[0] == ["Class", "Original Balance", "Interest Rate", "Final Payment Date"]
    assert rows[1][:2] == ["A-1", "$250,000,000"]
    assert len(rows) == 4


def test_table_fallback_empty_region_and_clamped_box() -> None:
    fallback = TableFallback(FIXTURE)
    try:
        # A box overhanging the page is clamped rather than rejected.
        assert fallback.extract(3, (-5, -5, 700, 60)) == []
    finally:
        fallback.close()
        fallback.close()  # idempotent
