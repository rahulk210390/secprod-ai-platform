"""Find priorities of payments and return their steps in order.

Real prospectuses write waterfalls three ways, all seen in the sample set:

* a numbered list, one step per paragraph: ``(1) to the servicer, ...``;
* ordinal prose, often several steps in one paragraph:
  ``First, to the Class A-1 ...; Second, to ...``;
* a table whose first column is the step number.

A waterfall is located either by its section heading ("Priority of Payments")
or by the lead-in sentence that introduces it ("... in the following order of
priority:"). Steps must count up from 1 without gaps; a marker that breaks the
sequence ("items (1) through (7)") is treated as ordinary text.

This module only *finds and orders* the steps. Interpreting them is JOB-04's
job, and computing anything from them is deterministic code's, never the
LLM's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from secai.parsing.models import (
    Block,
    BlockKind,
    Section,
    Table,
    Waterfall,
    WaterfallStep,
)
from secai.parsing.sections import canonical_name

WATERFALL_SECTIONS: Final = frozenset(
    {"priority_of_payments", "post_acceleration_priority_of_payments"}
)
MIN_STEPS: Final = 3
# How far past a lead-in or heading the first step may start, in blocks.
_START_WINDOW: Final = 4
# Upper bound on how many blocks one waterfall may span.
_MAX_SPAN: Final = 400

LEAD_IN = re.compile(
    r"in the following (amounts and )?order( of priorit(y|ies))?"
    r"|in the following priorit(y|ies)"
    r"|in the order of priority"
    r"|in the following manner and order"
    # "will apply ... Finance Charge Amounts as follows:" (card master trusts).
    # A payment verb is required: "may be reduced as follows:" is not a waterfall.
    r"|\b(apply|applied|distribute|distributed|pay|paid|allocate|allocated)\b[^.:]{0,160}"
    r"\bas follows:",
    re.IGNORECASE,
)

_UNITS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth",
]  # fmt: skip
_TENS_ORDINAL = {"twentieth": 20, "thirtieth": 30}
_TENS_PREFIX = {"twenty": 20, "thirty": 30}


def _ordinal_words() -> dict[str, int]:
    words = {word: position for position, word in enumerate(_UNITS, start=1)}
    words.update(_TENS_ORDINAL)
    for prefix, tens in _TENS_PREFIX.items():
        for unit, value in list(words.items())[:9]:
            words[f"{prefix}-{unit}"] = tens + value
    return words


ORDINAL_WORDS: Final = _ordinal_words()
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50}


def _roman_value(numeral: str) -> int | None:
    total, previous = 0, 0
    for char in reversed(numeral.lower()):
        value = _ROMAN.get(char)
        if value is None:
            return None
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total or None


# --------------------------------------------------------------------------
# markers
# --------------------------------------------------------------------------
# A marker counts only at the start of a block or after punctuation that ends
# a clause, so "Class A-1" and "Section 4(2)" are never read as steps.
_BOUNDARY = r"(?:^|(?<=[:;,.])\s+|(?<=\band)\s+|(?<=\bor)\s+)"
_STYLES: Final[dict[str, re.Pattern[str]]] = {
    "paren_number": re.compile(_BOUNDARY + r"\((\d{1,2})\)\s*"),
    "number_dot": re.compile(_BOUNDARY + r"(\d{1,2})[.)]\s+"),
    "paren_roman": re.compile(_BOUNDARY + r"\(([ivxl]{1,6})\)\s*"),
    "paren_alpha": re.compile(_BOUNDARY + r"\(([a-z])\)\s*"),
    # Longest first, so "twenty-first" wins over "twenty".
    "ordinal_word": re.compile(
        _BOUNDARY
        + r"("
        + "|".join(sorted(ORDINAL_WORDS, key=len, reverse=True))
        + r")\b\s*[,:.]?\s*",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Marker:
    style: str
    value: int
    text: str
    start: int
    end: int


def _value(style: str, raw: str) -> int | None:
    if style in ("paren_number", "number_dot"):
        return int(raw)
    if style == "paren_roman":
        return _roman_value(raw)
    if style == "paren_alpha":
        return ord(raw.lower()) - ord("a") + 1
    return ORDINAL_WORDS.get(raw.lower())


def find_markers(text: str, style: str) -> list[Marker]:
    """Every marker of ``style`` in ``text``, in order of position."""
    found: list[Marker] = []
    for match in _STYLES[style].finditer(text):
        raw = match.group(1)
        value = _value(style, raw)
        if value is None:
            continue
        if style == "ordinal_word" and not raw[0].isupper() and match.start(1) != 0:
            # "the first priority principal payment" is prose, not a step. A
            # lowercase ordinal only counts when it opens the block, as in a
            # list of "first, to ...; second, to ..." items.
            continue
        # Paren styles capture inside the "(": the marker starts one earlier.
        start = match.start(1)
        if text[start - 1 : start] == "(":
            start -= 1
        found.append(
            Marker(
                style=style,
                value=value,
                text=match.group(0).strip(" ,:.;"),
                start=start,
                end=match.end(),
            )
        )
    return found


def detect_first_marker(text: str) -> Marker | None:
    """The earliest marker with value 1, whatever its style.

    "(i)" is both roman one and the ninth letter; as a first step it can only
    be roman, so roman is tried before alpha.
    """
    candidates = [
        marker
        for style in ("paren_number", "number_dot", "paren_roman", "paren_alpha", "ordinal_word")
        for marker in find_markers(text, style)
        if marker.value == 1
    ]
    return min(candidates, key=lambda marker: marker.start) if candidates else None


# --------------------------------------------------------------------------
# list-form waterfalls
# --------------------------------------------------------------------------
@dataclass
class _Draft:
    marker: str
    text: str
    page: int
    page_label: str | None


@dataclass
class _Run:
    style: str
    first_block: int
    steps: list[_Draft] = field(default_factory=list)
    last_block: int = 0


def _scan_text(block: Block) -> str:
    """The block's text with its list marker restored.

    Docling lifts a list item's marker ("(3)") out of the text into its own
    field; the sequence logic needs it back in front.
    """
    return f"{block.marker} {block.text}" if block.marker else block.text


def _continues(previous: str, text: str) -> bool:
    """Whether a marker-less block carries on the step before it.

    Steps end in "," or ";" and the last one in "."; a block after a finished
    sentence is new prose ("If available funds are insufficient ...") unless
    it is clearly a continuation or a nested sub-list. A step that stops
    mid-sentence ("up to a maximum of") was broken by a page, and the next
    block finishes it whatever its first character.
    """
    if not text:
        return True
    tail = previous.rstrip()
    if not tail or tail[-1] not in ".!?":
        return True
    return text[0].islower() or text[0] == "("


def _build_run(blocks: Sequence[Block], start: int, offset: int, stop: int) -> _Run | None:
    """Collect consecutive steps beginning in ``blocks[start]`` at ``offset``."""
    first: Marker | None = None
    first_block = start
    for index in range(start, min(stop, start + _START_WINDOW)):
        block = blocks[index]
        if block.kind is BlockKind.HEADING and index != start:
            return None
        text = _scan_text(block)[offset:] if index == start else _scan_text(block)
        first = detect_first_marker(text)
        if first is not None:
            first_block = index
            break
    if first is None:
        return None

    run = _Run(style=first.style, first_block=first_block)
    expected = 1
    for index in range(first_block, min(stop, first_block + _MAX_SPAN)):
        block = blocks[index]
        if index != first_block and block.kind in (BlockKind.HEADING, BlockKind.TABLE):
            break
        text = _scan_text(block)[offset:] if index == start else _scan_text(block)

        cuts: list[Marker] = []
        for marker in find_markers(text, run.style):
            if marker.value == expected + len(cuts):
                cuts.append(marker)

        if not cuts:
            if run.steps and _continues(run.steps[-1].text, text):
                run.steps[-1].text = f"{run.steps[-1].text} {text}".strip()
                run.last_block = index
                continue
            break

        prefix = text[: cuts[0].start].strip()
        # A list item that opens with a label ("Class A Shortfalls. First, ...")
        # is a new step whose label precedes its marker. In running prose the
        # text before a marker finishes the previous step instead.
        label = prefix if block.kind is BlockKind.LIST_ITEM and index != start else ""
        if prefix and not label and run.steps:
            run.steps[-1].text = f"{run.steps[-1].text} {prefix}".strip()
        for position, cut in enumerate(cuts):
            end = cuts[position + 1].start if position + 1 < len(cuts) else len(text)
            # A labelled step is kept verbatim, marker included ("Transferor.
            # Ninth, remaining ..."), so its text can be found in the source.
            begin = 0 if position == 0 and label else cut.end
            run.steps.append(
                _Draft(
                    marker=cut.text,
                    text=text[begin:end].strip(),
                    page=block.page,
                    page_label=block.page_label,
                )
            )
        expected += len(cuts)
        run.last_block = index
    return run if len(run.steps) >= MIN_STEPS else None


# --------------------------------------------------------------------------
# table-form waterfalls
# --------------------------------------------------------------------------
_BARE_NUMBER = re.compile(r"(\d{1,2})[.)]?")


def table_steps(table: Table) -> list[WaterfallStep]:
    """Steps from a table whose first column counts 1, 2, 3 ... (any marker style)."""
    steps: list[WaterfallStep] = []
    style: str | None = None
    for row in table.rows:
        if not row:
            continue
        head = row[0].strip()
        bare = _BARE_NUMBER.fullmatch(head)
        if bare is not None and style in (None, "bare_number"):
            # A step column often holds just "1", "2", "3".
            marker: Marker | None = Marker(
                style="bare_number",
                value=int(bare.group(1)),
                text=head,
                start=0,
                end=len(head),
            )
        elif style is None:
            marker = detect_first_marker(head)
        elif style == "bare_number":
            marker = None  # the numbered rows have ended
        else:
            marker = next(iter(find_markers(head, style)), None)
        if marker is None or marker.value != len(steps) + 1 or marker.start != 0:
            if steps:
                break
            continue  # header rows before the first step
        style = marker.style
        rest = " ".join(cell.strip() for cell in (head[marker.end :], *row[1:]) if cell.strip())
        steps.append(
            WaterfallStep(
                ordinal=marker.value,
                marker=marker.text,
                text=rest,
                page=table.page,
                page_label=table.page_label,
            )
        )
    return steps if len(steps) >= MIN_STEPS else []


# --------------------------------------------------------------------------
# document level
# --------------------------------------------------------------------------
def _innermost(sections: Sequence[Section], block_index: int) -> Section | None:
    containing = [s for s in sections if s.heading_index <= block_index < s.end]
    return max(containing, key=lambda s: s.level) if containing else None


def _enclosing(sections: Sequence[Section], block: Block | None) -> list[Section]:
    """Every section containing ``block``, outermost first."""
    if block is None:
        return []
    return [s for s in sections if s.heading_index <= block.index < s.end]


def _path(section: Section | None) -> list[str]:
    return [*section.path, section.title] if section is not None else []


def _lead_in_sentence(text: str, match: re.Match[str]) -> str:
    previous_stop = text.rfind(". ", 0, match.start())
    begin = previous_stop + 2 if previous_stop != -1 else 0
    colon = text.find(":", match.end())
    end = colon + 1 if colon != -1 else match.end()
    return text[begin:end].strip()


def find_waterfalls(
    blocks: Sequence[Block],
    tables: Sequence[Table],
    sections: Sequence[Section],
) -> list[Waterfall]:
    """Every waterfall in the document, in document order.

    Where a heading and a lead-in point at the same list, it is reported once,
    titled by the heading.
    """
    found: dict[int, tuple[str, _Run]] = {}

    def consider(title: str, run: _Run | None) -> None:
        if run is None:
            return
        existing = found.get(run.first_block)
        if existing is None or len(run.steps) > len(existing[1].steps):
            found[run.first_block] = (title, run)

    for heading_section in sections:
        if heading_section.canonical in WATERFALL_SECTIONS:
            consider(
                heading_section.title,
                _build_run(blocks, heading_section.start, 0, heading_section.end),
            )

    for block in blocks:
        if block.kind is BlockKind.HEADING:
            continue
        scan = _scan_text(block)
        for match in LEAD_IN.finditer(scan):
            enclosing = _innermost(sections, block.index)
            stop = enclosing.end if enclosing is not None else len(blocks)
            run = _build_run(blocks, block.index, match.end(), stop)
            if run is not None and run.first_block not in found:
                consider(_lead_in_sentence(scan, match), run)

    waterfalls = [
        Waterfall(
            title=title,
            section_path=_path(_innermost(sections, run.first_block)),
            form="list",
            steps=[
                WaterfallStep(
                    ordinal=ordinal,
                    marker=draft.marker,
                    text=draft.text,
                    page=draft.page,
                    page_label=draft.page_label,
                )
                for ordinal, draft in enumerate(run.steps, start=1)
            ],
            start_page=run.steps[0].page,
            end_page=run.steps[-1].page,
        )
        for title, run in found.values()
    ]

    table_blocks = {b.table_index: b for b in blocks if b.table_index is not None}
    for table in tables:
        anchor = table_blocks.get(table.index)
        owner = _innermost(sections, anchor.index) if anchor is not None else None
        # A numbered first column alone is not evidence: static-pool and
        # weighted-average-life tables count 1, 2, 3 too. The table must sit
        # in a priority-of-payments section or be captioned as one.
        in_waterfall_section = owner is not None and any(
            s.canonical in WATERFALL_SECTIONS for s in _enclosing(sections, anchor)
        )
        captioned = canonical_name(table.caption or "") in WATERFALL_SECTIONS
        if not (in_waterfall_section or captioned):
            continue
        steps = table_steps(table)
        if not steps:
            continue
        waterfalls.append(
            Waterfall(
                title=table.caption or (owner.title if owner else "table"),
                section_path=_path(owner),
                form="table",
                steps=steps,
                start_page=table.page,
                end_page=table.page,
            )
        )

    return sorted(waterfalls, key=lambda w: (w.start_page, w.title))
