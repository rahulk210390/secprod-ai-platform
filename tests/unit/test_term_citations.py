"""Citations are found in the document, never taken from the model."""

from __future__ import annotations

from parsing_helpers import blocks, para

from secai.jobs.term_extraction.citations import find_citation, normalise
from secai.parsing.models import Block, BlockKind


def _table_block(text: str, page: int, label: str) -> Block:
    return Block(index=0, kind=BlockKind.TABLE, text=text, page=page, page_label=label)


def test_normalise_tolerates_pdf_spacing_and_typography() -> None:
    assert normalise("$ 320,400,000") == normalise("$320,400,000")
    assert normalise("4.057 %") == "4.057%"
    en_dash, curly_apostrophe = chr(0x2013), chr(0x2019)
    assert normalise(f"Clean{en_dash}Up  Call") == "clean-up call"
    assert normalise(f"Ford{curly_apostrophe}s") == "ford's"


def test_citation_from_a_table_cell() -> None:
    doc = [_table_block("Class A-1 notes | $ 320,400,000 | 4.057%", page=11, label="9")]
    citation = find_citation("$320,400,000", doc)
    assert citation is not None
    assert (citation.page, citation.page_label, citation.quote) == (11, "9", "$320,400,000")


def test_value_not_in_the_document_is_not_cited() -> None:
    doc = blocks(para("The Class A-1 notes bear interest at 4.057%.", page=11))
    assert find_citation("0.04057", doc) is None


def test_preferred_blocks_win_over_earlier_mentions() -> None:
    doc = blocks(
        para("A 4.057% rate appears in the cover summary.", page=1),
        para("Class A-1 interest rate: 4.057%", page=11, label="9"),
    )
    citation = find_citation("4.057%", doc, prefer=[1])
    assert citation is not None and citation.page == 11
    fallback = find_citation("4.057%", doc)
    assert fallback is not None and fallback.page == 1


def test_empty_or_trivial_quotes_are_not_cited() -> None:
    doc = blocks(para("A B C", page=1))
    assert find_citation(None, doc) is None
    assert find_citation("", doc) is None
    assert find_citation("A", doc) is None


def test_out_of_range_preference_is_ignored() -> None:
    doc = blocks(para("closing date September 16, 2025", page=3))
    citation = find_citation("September 16, 2025", doc, prefer=[99, -1])
    assert citation is not None and citation.page == 3
