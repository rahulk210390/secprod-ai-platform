"""What the model gets to read, and that it fits."""

from __future__ import annotations

import pytest
import term_fixtures as fx

from secai.jobs.term_extraction.context import (
    Target,
    budget_chars,
    build_contexts,
    is_risk_factor,
    render,
    tranche_tables,
)
from secai.parsing.models import Block, BlockKind, Section, Table


def test_budget_leaves_room_for_prompt_and_answer() -> None:
    assert budget_chars(4096, 1024) == int((4096 - 1024 - 150) * 2.6)
    # A longer prompt leaves less room for the document.
    assert budget_chars(4096, 1024, prompt_chars=2600) == int((4096 - 1024 - 1000 - 150) * 2.6)
    with pytest.raises(ValueError, match="VLLM_MAX_TOKENS"):
        budget_chars(4096, 4096)


def test_render_marks_printed_pages() -> None:
    doc = fx.document()
    text = render(doc.blocks, [0, 3, 4])
    assert text.splitlines()[0] == "[page pdf 1]"
    assert "[page 1]" in text
    assert "## SUMMARY OF TERMS" in text


def test_render_restores_list_markers() -> None:
    doc = fx.document()
    assert "(3) to the noteholders, principal." in render(doc.blocks, [15])


def test_facts_context_has_cover_and_fact_sections() -> None:
    (context,) = build_contexts(fx.document(), Target.FACTS, 10_000)
    assert "March 18, 2026" in context.text
    assert "10% or less" in context.text
    assert "15th day of each month" in context.text
    assert "delinquency trigger" not in context.text


def test_facts_include_key_value_fact_tables_and_all_contexts() -> None:
    doc = fx.document()
    rows = [["Closing Date", "On or about April 24, 2025."], ["Distribution Date", "Monthly."]]
    table = Table(index=1, page=34, page_label="34", rows=rows, source="docling")
    block = Block(
        index=len(doc.blocks),
        kind=BlockKind.TABLE,
        text="Closing Date | On or about April 24, 2025.",
        page=34,
        page_label="34",
        table_index=1,
    )
    doc = doc.model_copy(update={"tables": [*doc.tables, table], "blocks": [*doc.blocks, block]})
    contexts = build_contexts(doc, Target.FACTS, 10_000)
    assert any("April 24, 2025" in c.text for c in contexts)
    # A small budget spreads the facts over several contexts rather than dropping them.
    small = build_contexts(doc, Target.FACTS, 300)
    assert 1 < len(small) <= 3


def test_oversized_single_block_is_truncated_not_sent_whole() -> None:
    doc = fx.document()
    huge = doc.blocks[6].model_copy(update={"text": "March 18, 2026. " + "x" * 5_000})
    doc = doc.model_copy(update={"blocks": [*doc.blocks[:6], huge, *doc.blocks[7:]]})
    contexts = build_contexts(doc, Target.FACTS, 1_000)
    assert all(len(c.text) <= 1_000 for c in contexts)
    assert any(c.text.endswith("[...]") for c in contexts)


def test_data_sections_that_mention_a_fact_are_not_fact_sections() -> None:
    # John Deere: a static-pool table titled "As of the February 2026 Payment
    # Date" was pulled in as a fact section and overflowed the context.
    doc = fx.document()
    title = "As of the February 2026 Payment Date"
    sections = [
        s.model_copy(update={"title": title}) if s.title == "Clean Up Call" else s
        for s in doc.sections
    ]
    doc = doc.model_copy(update={"sections": sections})
    text = "\n".join(c.text for c in build_contexts(doc, Target.FACTS, 10_000))
    assert "10% or less of the initial pool balance" not in text
    assert "March 18, 2026" in text  # the real fact section is still there


def test_tranche_context_includes_the_cover_headline_total() -> None:
    doc = fx.document()
    headline = doc.blocks[0].model_copy(
        update={"text": "$600,000,000 Synthetic Auto Owner Trust 2099-1", "kind": BlockKind.TEXT}
    )
    doc = doc.model_copy(update={"blocks": [headline, *doc.blocks[1:]]})
    (context,) = build_contexts(doc, Target.TRANCHES, 10_000)
    assert context.text.index("$600,000,000 Synthetic") < context.text.index("Class A-1")


def test_trigger_context_selects_the_trigger_section() -> None:
    (context,) = build_contexts(fx.document(), Target.TRIGGERS, 10_000)
    assert "1.25%" in context.text
    assert context.block_indices == fx.TRIGGER_BLOCKS


def test_no_coverage_sections_means_no_calls() -> None:
    assert build_contexts(fx.document(), Target.COVERAGE, 10_000) == []


def test_large_sections_are_split_to_fit() -> None:
    doc = fx.document()
    blocks = [
        Block(index=i, kind=BlockKind.TEXT, text="x" * 900, page=10 + i, page_label=str(i))
        for i in range(10)
    ]
    heading = Block(index=10, kind=BlockKind.HEADING, text="Trigger Events", page=9, level=2)
    blocks = [
        heading.model_copy(update={"index": 0}),
        *[b.model_copy(update={"index": b.index + 1}) for b in blocks],
    ]
    section = Section(
        title="Trigger Events",
        canonical="trigger_events",
        level=2,
        heading_index=0,
        start=1,
        end=len(blocks),
        start_page=9,
        end_page=19,
        path=[],
    )
    big = doc.model_copy(update={"blocks": blocks, "sections": [section], "tables": []})
    contexts = build_contexts(big, Target.TRIGGERS, 2_500)
    assert len(contexts) > 1
    assert all(len(c.text) <= 2_500 for c in contexts)
    assert sorted(i for c in contexts for i in c.block_indices) == list(range(len(blocks)))


def test_risk_factor_sections_are_never_context() -> None:
    risk = Section(
        title="Risk Factors",
        canonical=None,
        level=1,
        heading_index=0,
        start=1,
        end=2,
        start_page=1,
        end_page=1,
        path=[],
    )
    sentence = risk.model_copy(update={"title": "x" * 150, "path": []})
    nested = risk.model_copy(update={"title": "Trigger Events", "path": ["RISK FACTORS"]})
    assert is_risk_factor(risk) and is_risk_factor(sentence) and is_risk_factor(nested)


def test_tranche_tables_prefer_labelled_pages() -> None:
    rows = [["Class", "Balance"], ["Class A-1", "$250,000,000"], ["Class B", "$50,000,000"]]
    cover = Table(index=0, page=1, page_label=None, rows=rows, source="docling")
    summary = Table(index=1, page=6, page_label="1", rows=rows, source="docling")
    not_tranches = Table(
        index=2, page=6, page_label="1", rows=[["Date", "Rate"], ["x", "y"]], source="docling"
    )
    doc = fx.document().model_copy(update={"tables": [cover, summary, not_tranches]})
    assert [t.index for t in tranche_tables(doc)] == [1, 0]
