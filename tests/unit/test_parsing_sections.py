"""Section chunking and canonical section names."""

from __future__ import annotations

import pytest
from parsing_helpers import blocks, head, para

from secai.parsing.models import ParsedDocument
from secai.parsing.sections import canonical_name, chunk_by_section, remove_running_headers


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Priority of Payments", "priority_of_payments"),
        ("PRIORITY OF PAYMENTS", "priority_of_payments"),
        ("Order of Priority", "priority_of_payments"),
        ("Application of Available Funds", "priority_of_payments"),
        ("Distributions", "priority_of_payments"),
        ("Priority of Distributions", "priority_of_payments"),
        ("Post-Acceleration Priority of Payments", "post_acceleration_priority_of_payments"),
        (
            "Priority of Payments After an Event of Default",
            "post_acceleration_priority_of_payments",
        ),
        ("Distributions after acceleration", "post_acceleration_priority_of_payments"),
        ("Trigger Events", "trigger_events"),
        ("Early Amortization Events", "trigger_events"),
        ("Pay Out Events", "trigger_events"),
        ("Events of Default", "events_of_default"),
        ("DESCRIPTION OF THE NOTES", "description_of_notes"),
        ("Description of the Offered Certificates", "description_of_notes"),
        ("Credit Enhancement", "credit_enhancement"),
        ("Application of Card Series Finance Charge Amounts", "priority_of_payments"),
        ("Application of Card Series Principal Amounts", "priority_of_payments"),
        ("Application of the Proceeds of the Offering", None),
        ("The Receivables Pool", None),
        ("Risk Factors", None),
    ],
)
def test_canonical_name(title: str, expected: str | None) -> None:
    assert canonical_name(title) == expected


def test_canonical_name_ignores_whitespace_and_punctuation() -> None:
    assert canonical_name("  Priority of\n Payments: ") == "priority_of_payments"


def test_sections_nest_by_level() -> None:
    doc = blocks(
        para("cover text"),
        head("DESCRIPTION OF THE NOTES", level=1, page=10),
        para("intro", page=10),
        head("Priority of Payments", level=2, page=11),
        para("(1) to the servicer", page=11),
        head("Events of Default", level=2, page=12),
        para("default text", page=13),
        head("THE RECEIVABLES POOL", level=1, page=14),
        para("pool text", page=14),
    )
    sections = chunk_by_section(doc)

    assert [s.title for s in sections] == [
        "DESCRIPTION OF THE NOTES",
        "Priority of Payments",
        "Events of Default",
        "THE RECEIVABLES POOL",
    ]
    notes, priority, defaults, pool = sections

    # A top-level section contains its subsections, up to the next level-1 heading.
    assert (notes.start, notes.end) == (2, 7)
    assert (notes.start_page, notes.end_page) == (10, 13)
    assert notes.path == []

    assert (priority.start, priority.end) == (4, 5)
    assert priority.canonical == "priority_of_payments"
    assert priority.path == ["DESCRIPTION OF THE NOTES"]

    assert defaults.path == ["DESCRIPTION OF THE NOTES"]
    assert defaults.end_page == 13
    assert pool.path == []
    assert pool.end == len(doc)


def test_heading_without_content_ends_on_its_own_page() -> None:
    sections = chunk_by_section(blocks(head("A", level=1, page=3), head("B", level=1, page=4)))
    assert sections[0].start_page == sections[0].end_page == 3


def test_no_headings_means_no_sections() -> None:
    assert chunk_by_section(blocks(para("just text"))) == []


def test_running_header_is_removed_and_blocks_renumbered() -> None:
    specs = []
    for page in range(1, 6):
        specs.append(head("Table of Contents", level=2, page=page))
        specs.append(para(f"body of page {page}", page=page))
    specs.append(head("Priority of Payments", level=2, page=5))
    cleaned = remove_running_headers(blocks(*specs))

    assert [b.text for b in cleaned][:2] == ["body of page 1", "body of page 2"]
    assert all(b.text != "Table of Contents" for b in cleaned)
    assert [b.index for b in cleaned] == list(range(len(cleaned)))


def test_rare_repeated_text_is_kept() -> None:
    # Two pages out of ten opening the same way is coincidence, not a header.
    specs = [
        head("Overview", level=2, page=p) if p <= 2 else para(f"text {p}", page=p)
        for p in range(1, 11)
    ]
    assert len(remove_running_headers(blocks(*specs))) == 10


def test_long_repeated_text_is_kept() -> None:
    sentence = "This sentence is far too long to be a running header on every single page."
    specs = [para(sentence, page=p) for p in range(1, 6)]
    assert len(remove_running_headers(blocks(*specs))) == 5


def test_section_text_and_lookup_on_parsed_document() -> None:
    doc_blocks = blocks(
        head("Priority of Payments", level=2), para("first line"), para("second line")
    )
    sections = chunk_by_section(doc_blocks)
    document = ParsedDocument(
        source="x.pdf",
        sha256="0" * 64,
        parser="docling",
        page_count=1,
        blocks=doc_blocks,
        tables=[],
        sections=sections,
        waterfalls=[],
        duration_s=0.0,
    )
    (priority,) = document.sections_named("priority_of_payments")
    assert document.section_text(priority) == "first line\nsecond line"
    assert document.sections_named("trigger_events") == []
