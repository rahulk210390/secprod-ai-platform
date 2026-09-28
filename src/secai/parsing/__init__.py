"""Document parsing: PDF → ordered blocks, tables, sections and waterfalls (JOB-03)."""

from secai.parsing.models import (
    Block,
    BlockKind,
    ParsedDocument,
    Section,
    Table,
    Waterfall,
    WaterfallStep,
)
from secai.parsing.pipeline import JOB_NAME, parse_document, run_parsing_job
from secai.parsing.sections import canonical_name, chunk_by_section
from secai.parsing.waterfall import find_waterfalls

__all__ = [
    "JOB_NAME",
    "Block",
    "BlockKind",
    "ParsedDocument",
    "Section",
    "Table",
    "Waterfall",
    "WaterfallStep",
    "canonical_name",
    "chunk_by_section",
    "find_waterfalls",
    "parse_document",
    "run_parsing_job",
]
