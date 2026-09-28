"""Parse a document end to end, traced.

:func:`parse_document` emits the ``parse`` and ``chunk`` child spans and is what
later jobs call inside their own root trace. :func:`run_parsing_job` wraps it
in a ``job.document_parsing`` root trace for standalone runs (CLI, smoke).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Final

from secai.config import Settings, get_settings
from secai.parsing.cache import file_sha256, load_or_parse
from secai.parsing.docling_parser import Converter, DoclingResult, parse_pdf
from secai.parsing.models import Block, ParsedDocument, Section, Waterfall
from secai.parsing.sections import chunk_by_section, remove_running_headers
from secai.parsing.waterfall import find_waterfalls
from secai.telemetry import current_trace_id, job_span, job_trace, set_span_attributes
from secai.telemetry.attributes import (
    ATTR_CHUNK_SECTION_COUNT,
    ATTR_CHUNK_WATERFALL_COUNT,
    ATTR_DOC_SHA256,
    ATTR_PARSE_BATCH_PAGES,
    ATTR_PARSE_CACHED,
    ATTR_PARSE_DURATION_S,
    ATTR_PARSE_FALLBACK_TABLES,
    ATTR_PARSE_PAGE_COUNT,
    ATTR_PARSE_PARSER,
    ATTR_PARSE_TABLE_COUNT,
    SPAN_CHUNK,
    SPAN_PARSE,
)

JOB_NAME: Final = "document_parsing"
PARSER_NAME: Final = "docling"


def analyse(result: DoclingResult) -> tuple[list[Block], list[Section], list[Waterfall]]:
    """Everything after Docling: clean-up, sections, waterfalls. Cheap and deterministic.

    Kept separate from :func:`parse_pdf` so cached Docling output always goes
    through the current version of this logic.
    """
    blocks = remove_running_headers(result.blocks)
    sections = chunk_by_section(blocks)
    return blocks, sections, find_waterfalls(blocks, result.tables, sections)


def parse_document(
    path: Path,
    *,
    settings: Settings | None = None,
    converter: Converter | None = None,
    cache_dir: Path | None = None,
) -> ParsedDocument:
    """Parse ``path`` into blocks, tables, sections and waterfalls.

    With ``cache_dir``, Docling's output is reused when the file is unchanged
    (same sha256), which turns a 45-minute parse into a file read. Chunking and
    waterfall extraction always run fresh.
    """
    resolved = settings if settings is not None else get_settings()
    if not path.is_file():
        raise FileNotFoundError(path)

    # Only the file name reaches the trace: a local path can carry a user name.
    with job_span(SPAN_PARSE, input_data={"source": path.name}) as span:
        started = time.perf_counter()
        digest = file_sha256(path)
        if cache_dir is not None:
            result, cached = load_or_parse(path, resolved.parsing, cache_dir, converter=converter)
        else:
            result, cached = parse_pdf(path, resolved.parsing, converter=converter), False
        duration = time.perf_counter() - started
        set_span_attributes(
            span,
            {
                ATTR_DOC_SHA256: digest,
                ATTR_PARSE_CACHED: cached,
                ATTR_PARSE_PARSER: PARSER_NAME,
                ATTR_PARSE_PAGE_COUNT: result.page_count,
                ATTR_PARSE_TABLE_COUNT: len(result.tables),
                ATTR_PARSE_FALLBACK_TABLES: result.fallback_tables,
                ATTR_PARSE_BATCH_PAGES: resolved.parsing.batch_pages,
                ATTR_PARSE_DURATION_S: round(duration, 3),
            },
        )
        span.update(
            output={
                "pages": result.page_count,
                "blocks": len(result.blocks),
                "tables": len(result.tables),
                "duration_s": round(duration, 3),
            }
        )

    with job_span(SPAN_CHUNK) as span:
        blocks, sections, waterfalls = analyse(result)
        set_span_attributes(
            span,
            {
                ATTR_CHUNK_SECTION_COUNT: len(sections),
                ATTR_CHUNK_WATERFALL_COUNT: len(waterfalls),
            },
        )
        span.update(
            output={
                "sections": len(sections),
                "canonical": sorted({s.canonical for s in sections if s.canonical}),
                "waterfalls": [
                    {
                        "title": w.title[:120],
                        "steps": len(w.steps),
                        "pages": [w.start_page, w.end_page],
                    }
                    for w in waterfalls
                ],
            }
        )

    return ParsedDocument(
        source=path.name,
        sha256=digest,
        parser=PARSER_NAME,
        page_count=result.page_count,
        blocks=blocks,
        tables=result.tables,
        sections=sections,
        waterfalls=waterfalls,
        duration_s=round(duration, 3),
    )


def run_parsing_job(
    path: Path,
    *,
    deal_id: str | None = None,
    asset_class: str | None = None,
    session_id: str | None = None,
    settings: Settings | None = None,
    converter: Converter | None = None,
) -> ParsedDocument:
    """One parse as its own ``job.document_parsing`` trace."""
    with job_trace(
        JOB_NAME,
        deal_id=deal_id,
        asset_class=asset_class,
        doc_type="prospectus",
        session_id=session_id,
        tags=["job-03"],
        input_data={"source": path.name},
    ) as root:
        document = parse_document(path, settings=settings, converter=converter)
        root.update(
            output={
                "pages": document.page_count,
                "sections": len(document.sections),
                "waterfalls": len(document.waterfalls),
            }
        )
        return document.model_copy(update={"trace_id": current_trace_id()})
