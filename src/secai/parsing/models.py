"""The parsed-document model every downstream job consumes.

A document is a flat, reading-ordered list of :class:`Block` objects plus the
:class:`Table` objects some of them point at. Sections and waterfalls are views
over that list (block index ranges), so nothing is duplicated and every piece
of text can be cited back to its page.

Two page numbers are kept because they differ: ``page`` is the 1-based PDF
page, which tools open; ``page_label`` is the number printed on the page
("78", "S-12"), which is what an analyst cites.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BlockKind(StrEnum):
    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Block(_Frozen):
    """One unit of content in reading order."""

    index: int = Field(ge=0, description="position in ParsedDocument.blocks")
    kind: BlockKind
    text: str
    page: int = Field(ge=1, description="1-based PDF page")
    page_label: str | None = Field(default=None, description="page number as printed")
    level: int | None = Field(default=None, ge=1, description="heading level; 1 is top")
    marker: str | None = Field(default=None, description="list marker, e.g. '(3)'")
    table_index: int | None = Field(default=None, description="for TABLE blocks")


class Table(_Frozen):
    """A table as a grid of cell strings, header row(s) included."""

    index: int = Field(ge=0)
    page: int = Field(ge=1)
    page_label: str | None = None
    rows: list[list[str]]
    caption: str | None = None
    source: str = Field(description="'docling' or 'pdfplumber' (fallback)")

    @property
    def num_rows(self) -> int:
        return len(self.rows)


class Section(_Frozen):
    """A heading and everything under it, down to the next heading at its level or above."""

    title: str
    canonical: str | None = Field(
        default=None, description="normalised section name, e.g. 'priority_of_payments'"
    )
    level: int = Field(ge=1)
    heading_index: int = Field(ge=0, description="block index of the heading")
    start: int = Field(ge=0, description="first block index after the heading")
    end: int = Field(ge=0, description="one past the last block index")
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    path: list[str] = Field(description="titles of enclosing sections, outermost first")


class WaterfallStep(_Frozen):
    """One step of a priority of payments."""

    ordinal: int = Field(ge=1, description="position in the waterfall, 1-based")
    marker: str = Field(description="as printed: '(3)', 'Third', 'iii.'")
    text: str
    page: int = Field(ge=1)
    page_label: str | None = None


class Waterfall(_Frozen):
    """An ordered priority of payments found in the document."""

    title: str = Field(description="the heading or lead-in sentence that introduces it")
    section_path: list[str]
    form: str = Field(description="'list' or 'table'")
    steps: list[WaterfallStep]
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)


class ParsedDocument(_Frozen):
    source: str = Field(description="file name only; never a full local path")
    sha256: str
    parser: str
    page_count: int = Field(ge=0)
    blocks: list[Block]
    tables: list[Table]
    sections: list[Section]
    waterfalls: list[Waterfall]
    duration_s: float = Field(ge=0)
    trace_id: str | None = Field(
        default=None, description="Langfuse trace of the run that produced this parse"
    )

    def section_text(self, section: Section) -> str:
        """The section's content as plain text, one block per line."""
        return "\n".join(block.text for block in self.blocks[section.start : section.end])

    def sections_named(self, canonical: str) -> list[Section]:
        return [section for section in self.sections if section.canonical == canonical]
