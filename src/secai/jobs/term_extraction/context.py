"""Choose what the model reads for each part of the deal terms.

Section-targeted extraction (CLAUDE.md JOB-04): the model sees only the blocks
relevant to one question, labelled with printed page numbers, and never more
than fits its context window. A 3B model given four relevant pages does far
better than one given forty.

Risk-factor sections are excluded throughout. Their headings mention the same
words ("...could result in an early redemption event...") but describe what
*might* happen, not the deal's terms.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from secai.parsing.models import Block, BlockKind, ParsedDocument, Section, Table

# Characters per token for budgeting. Prospectus tables full of figures
# tokenise worse than prose: a John Deere context measured 2.96, which a 3.0
# estimate overran by one token. 2.6 leaves room.
CHARS_PER_TOKEN: Final = 2.6
# Chat template, role markers and the JSON the model starts writing.
SAFETY_TOKENS: Final = 150
TRUNCATION_MARK: Final = "\n[...]"
# Below this there is no useful room for the document itself.
MIN_CONTEXT_TOKENS: Final = 500
# Risk-factor headings are whole sentences; real section titles are short.
MAX_TITLE_CHARS: Final = 100

_AMOUNT_CELL = re.compile(r"\d{1,3}(?:,\d{3}){2,}")
_CLASS_CELL = re.compile(r"\bclass\b|^[A-Z]-?\d[A-Z]?\b|^[A-Z]-[A-Z]\b", re.IGNORECASE)
_RISK = re.compile(r"risk factors", re.IGNORECASE)
_RATINGS = re.compile(r"^ratings?$|ratings? of the (notes|certificates|securities)", re.IGNORECASE)
# Tranche tables sit on the cover and in the summary, well before this page.
TRANCHE_TABLE_PAGES: Final = 25
MAX_TRANCHE_TABLES: Final = 3
_FACT_TITLES = re.compile(
    r"closing date|issuance date|payment dates?|distribution dates?|interest payment dates?"
    r"|clean.?up call|optional (redemption|purchase|termination)|servicer'?s? purchase option"
    # CMBS: the pool purchase threshold sits under the PSA termination section.
    r"|termination; retirement|retirement of (the )?certificates",
    re.IGNORECASE,
)
# Key/value summary tables with a "Closing Date" row sit near the front.
FACT_TABLE_PAGES: Final = 60
MAX_FACT_CONTEXTS: Final = 3
MAX_HEADLINE_CHARS: Final = 250
_COVERAGE_TITLES = re.compile(
    r"overcollaterali[sz]ation (test|ratio)|coverage test|interest coverage|\bO/C\b|\bI/C\b",
    re.IGNORECASE,
)
# Trigger sections not named as such: asset-review delinquency triggers, and
# CMBS control/consultation events. Static-pool loss history tables are data,
# not triggers, and stay out.
_TRIGGER_EXTRA = re.compile(
    r"delinquency trigger|asset (representations )?review|amortization events?"
    r"|control termination|consultation (termination )?event",
    re.IGNORECASE,
)
COVER_PAGES: Final = 2


class Target(StrEnum):
    TRANCHES = "tranches"
    FACTS = "facts"
    TRIGGERS = "triggers"
    COVERAGE = "coverage"


@dataclass
class Context:
    """One model input: rendered text and the blocks it came from."""

    target: Target
    text: str
    block_indices: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.block_indices


def budget_chars(max_model_len: int, max_output_tokens: int, prompt_chars: int = 0) -> int:
    """How many characters of document text fit beside the prompt and the answer.

    ``prompt_chars`` is the length of the system prompt plus the target's
    instructions, measured rather than assumed: the prompts differ in length
    and change as they are edited.

    Raises ``ValueError`` when the output budget leaves too little room: a
    silent zero would split every block into its own model call, the same
    misconfiguration that made every call fail with a 400 in JOB-02.
    """
    prompt_tokens = -(-prompt_chars // CHARS_PER_TOKEN)  # ceiling
    tokens = int(max_model_len - max_output_tokens - prompt_tokens - SAFETY_TOKENS)
    if tokens < MIN_CONTEXT_TOKENS:
        raise ValueError(
            f"VLLM_MAX_MODEL_LEN={max_model_len} with VLLM_MAX_TOKENS={max_output_tokens} "
            f"leaves {tokens} tokens for document text; at least {MIN_CONTEXT_TOKENS} are "
            "needed. Lower VLLM_MAX_TOKENS or raise VLLM_MAX_MODEL_LEN."
        )
    return int(tokens * CHARS_PER_TOKEN)


def is_risk_factor(section: Section) -> bool:
    return (
        len(section.title) > MAX_TITLE_CHARS
        or bool(_RISK.search(section.title))
        or any(_RISK.search(title) for title in section.path)
    )


def render(blocks: Sequence[Block], indices: Iterable[int]) -> str:
    """Blocks as plain text, with a "[page N]" marker whenever the printed page changes."""
    lines: list[str] = []
    current: str | None = None
    for index in indices:
        block = blocks[index]
        label = block.page_label or f"pdf {block.page}"
        if label != current:
            lines.append(f"[page {label}]")
            current = label
        prefix = f"{block.marker} " if block.marker else ""
        heading = "## " if block.kind is BlockKind.HEADING else ""
        lines.append(f"{heading}{prefix}{block.text}")
    return "\n".join(lines)


def _pack(
    target: Target,
    blocks: Sequence[Block],
    groups: Iterable[Sequence[int]],
    limit: int,
) -> list[Context]:
    """Fill contexts with whole groups where possible; split a group only if it alone is too big."""
    contexts: list[Context] = []
    current: list[int] = []
    seen: set[int] = set()

    def size(indices: list[int]) -> int:
        return len(render(blocks, indices))

    def flush() -> None:
        if current:
            text = render(blocks, current)
            if len(text) > limit:
                # Only a single block can exceed the limit (whole blocks are
                # packed otherwise). Cut it rather than send an oversized
                # prompt that vLLM rejects with a 400.
                text = text[: max(limit - len(TRUNCATION_MARK), 0)] + TRUNCATION_MARK
            contexts.append(Context(target, text, list(current)))
            current.clear()

    for group in groups:
        fresh = [i for i in group if i not in seen]
        seen.update(fresh)
        if not fresh:
            continue
        if size(current + fresh) <= limit:
            current.extend(fresh)
            continue
        flush()
        for index in fresh:
            if current and size([*current, index]) > limit:
                flush()
            current.append(index)
    flush()
    return contexts


def _section_blocks(section: Section) -> list[int]:
    return [section.heading_index, *range(section.start, section.end)]


def _sections(document: ParsedDocument, keep: Callable[[Section], bool]) -> list[Section]:
    """Matching sections, skipping ones nested inside an already-chosen section."""
    chosen: list[Section] = []
    for section in document.sections:
        if is_risk_factor(section) or not keep(section):
            continue
        if any(c.heading_index < section.heading_index < c.end for c in chosen):
            continue
        chosen.append(section)
    return chosen


def _is_tranche_table(table: Table) -> bool:
    """At least two rows naming a class, and figures (amounts or rates) beside them."""
    class_rows = sum(1 for row in table.rows if row and _CLASS_CELL.search(" ".join(row[:2])))
    figures = sum(
        1 for row in table.rows for cell in row if _AMOUNT_CELL.search(cell) or "%" in cell
    )
    return class_rows >= 2 and figures >= 2


def tranche_tables(document: ParsedDocument) -> list[Table]:
    """Tranche tables near the front, summary tables before the cover.

    Covers and summaries split the terms differently: John Deere's cover says
    only "Floating Rate Asset Backed Notes" where its summary table gives
    "Benchmark + 0.34%". Summary tables also carry printed page labels, which
    cover pages lack, so they produce better citations.
    """
    candidates = [
        t for t in document.tables if t.page <= TRANCHE_TABLE_PAGES and _is_tranche_table(t)
    ]
    return sorted(candidates, key=lambda t: (t.page_label is None, t.index))


def build_contexts(document: ParsedDocument, target: Target, limit: int) -> list[Context]:
    """The model inputs for one target, each within ``limit`` characters. May be empty."""
    blocks = document.blocks
    if target is Target.TRANCHES:
        # Short cover lines that print the total ("$779,883,000 John Deere
        # Owner Trust 2026"): the tables alone often carry no "Total" row.
        headline = [
            b.index
            for b in blocks
            if b.page <= COVER_PAGES
            and b.kind is not BlockKind.TABLE
            and len(b.text) <= MAX_HEADLINE_CHARS
            and _AMOUNT_CELL.search(b.text)
        ]
        groups: list[list[int]] = [headline] if headline else []
        groups += [
            [b.index for b in blocks if b.table_index == table.index]
            for table in tranche_tables(document)[:MAX_TRANCHE_TABLES]
        ]
        for section in _sections(document, lambda s: bool(_RATINGS.search(s.title))):
            groups.append(_section_blocks(section))
            break  # the summary's ratings section is enough
        return _pack(target, blocks, groups, limit)[:1]

    if target is Target.FACTS:
        # Dedicated sections first (most precise), then key/value fact tables
        # ("Closing Date | On or about April 24, 2025"), then the cover. Every
        # context is used: the clean-up call is often far from the cover.
        groups = [
            _section_blocks(s)
            # match(), not search(): "Payment Dates; Interest Accrual" is a fact
            # section, "As of the February 2026 Payment Date" is a data table.
            for s in _sections(document, lambda s: bool(_FACT_TITLES.match(s.title.strip())))
        ]
        groups += [
            [b.index for b in blocks if b.table_index == table.index]
            for table in document.tables
            if table.page <= FACT_TABLE_PAGES
            and any(row and _FACT_TITLES.search(row[0]) for row in table.rows)
        ]
        groups.append(
            [b.index for b in blocks if b.page <= COVER_PAGES and b.kind is not BlockKind.TABLE]
        )
        return _pack(target, blocks, groups, limit)[:MAX_FACT_CONTEXTS]

    if target is Target.TRIGGERS:
        sections = _sections(
            document,
            lambda s: s.canonical == "trigger_events" or bool(_TRIGGER_EXTRA.search(s.title)),
        )
        return _pack(target, blocks, (_section_blocks(s) for s in sections), limit)

    sections = _sections(document, lambda s: bool(_COVERAGE_TITLES.search(s.title)))
    return _pack(target, blocks, (_section_blocks(s) for s in sections), limit)
