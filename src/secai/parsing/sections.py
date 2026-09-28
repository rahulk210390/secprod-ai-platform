"""Chunk a parsed document by section, not by token count.

A section runs from its heading to the next heading at the same level or
above, so a top-level "DESCRIPTION OF THE NOTES" contains its "Priority of
Payments" subsection and both are returned. Headings are mapped to canonical
names so JOB-04 can ask for "the priority of payments" without knowing how
each issuer titles it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from secai.parsing.models import Block, BlockKind, Section

# Checked in order; the first match wins. Post-acceleration waterfalls come
# before the general pattern so they are not mistaken for the main one.
CANONICAL_SECTIONS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "post_acceleration_priority_of_payments",
        # Both phrases, in either order: "Post-Acceleration Priority of
        # Payments" and "Priority of Payments After an Event of Default".
        re.compile(
            r"^(?=.*(post[- ]?acceleration|after (an )?(acceleration|event of default)))"
            r"(?=.*(priority|payments|distributions))",
            re.IGNORECASE,
        ),
    ),
    (
        "priority_of_payments",
        re.compile(
            r"priority of (payments|distributions)|order of (priority|distributions)"
            r"|application of (available funds|collections|available collections)"
            # "Application of Card Series Finance Charge Amounts" (card trusts)
            r"|application of .*\b(amounts|funds|collections)$"
            r"|distributions? of available funds|payments? and distributions"
            r"|^distributions?$",
            re.IGNORECASE,
        ),
    ),
    (
        "trigger_events",
        re.compile(
            r"trigger|(early )?amortization events?|pay ?out events?|early redemption events?",
            re.IGNORECASE,
        ),
    ),
    ("events_of_default", re.compile(r"events? of default", re.IGNORECASE)),
    (
        "description_of_notes",
        re.compile(r"description of the (offered )?(notes|certificates|securities)", re.IGNORECASE),
    ),
    ("credit_enhancement", re.compile(r"credit enhancement", re.IGNORECASE)),
)

_WHITESPACE = re.compile(r"\s+")


def remove_running_headers(
    blocks: Sequence[Block], *, min_pages: int = 3, share: float = 0.2, max_chars: int = 60
) -> list[Block]:
    """Drop short text that opens many pages, then renumber.

    EDGAR filings printed to PDF carry a "Table of Contents" link at the top of
    every page, and layout models tag it as a heading. Left in, it would split
    sections and cut waterfalls in half at every page break. A text counts as
    a running header when it is the first block on at least ``share`` of the
    pages (and on ``min_pages`` pages at least).
    """
    first_on_page: dict[int, Block] = {}
    for block in blocks:
        first_on_page.setdefault(block.page, block)
    counts: dict[str, int] = {}
    for block in first_on_page.values():
        if block.kind in (BlockKind.HEADING, BlockKind.TEXT) and len(block.text) <= max_chars:
            key = _WHITESPACE.sub(" ", block.text).strip().lower()
            counts[key] = counts.get(key, 0) + 1
    threshold = max(min_pages, share * len(first_on_page))
    running = {text for text, count in counts.items() if count >= threshold}
    if not running:
        return list(blocks)

    kept = [
        block
        for block in blocks
        if not (
            block.kind in (BlockKind.HEADING, BlockKind.TEXT)
            and _WHITESPACE.sub(" ", block.text).strip().lower() in running
        )
    ]
    return [block.model_copy(update={"index": index}) for index, block in enumerate(kept)]


def canonical_name(title: str) -> str | None:
    """The canonical section a heading belongs to, or None."""
    normalised = _WHITESPACE.sub(" ", title).strip(" .:—-")
    for name, pattern in CANONICAL_SECTIONS:
        if pattern.search(normalised):
            return name
    return None


def chunk_by_section(blocks: Sequence[Block]) -> list[Section]:
    """One :class:`Section` per heading, in document order.

    Content before the first heading (cover page, notices) belongs to no
    section; downstream jobs never need it by name.
    """
    headings = [block for block in blocks if block.kind is BlockKind.HEADING]
    sections: list[Section] = []
    # Stack of (level, title) for the enclosing-section path.
    stack: list[tuple[int, str]] = []

    for position, heading in enumerate(headings):
        level = heading.level or 1
        while stack and stack[-1][0] >= level:
            stack.pop()
        path = [title for _, title in stack]
        stack.append((level, heading.text))

        end = len(blocks)
        for later in headings[position + 1 :]:
            if (later.level or 1) <= level:
                end = later.index
                break

        start = heading.index + 1
        last_page = blocks[end - 1].page if end > start else heading.page
        sections.append(
            Section(
                title=heading.text,
                canonical=canonical_name(heading.text),
                level=level,
                heading_index=heading.index,
                start=start,
                end=end,
                start_page=heading.page,
                end_page=last_page,
                path=path,
            )
        )
    return sections
