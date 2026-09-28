"""Waterfall extraction: every layout the sample prospectuses use.

The acceptance criterion for JOB-03 is that priority-of-payments steps come
out in order, so these tests assert order (ordinals and step text), not just
counts.
"""

from __future__ import annotations

import pytest
from parsing_helpers import blocks, head, item, para

from secai.parsing.models import Table
from secai.parsing.sections import chunk_by_section
from secai.parsing.waterfall import (
    ORDINAL_WORDS,
    detect_first_marker,
    find_markers,
    find_waterfalls,
    table_steps,
)


def _waterfalls(*specs):  # type: ignore[no-untyped-def]
    doc = blocks(*specs)
    return find_waterfalls(doc, [], chunk_by_section(doc))


# --------------------------------------------------------------------------
# markers
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "style", "values"),
    [
        ("(1) a, (2) b", "paren_number", [1, 2]),
        ("1. a; 2. b", "number_dot", [1, 2]),
        ("(i) a; (ii) b; (iv) c", "paren_roman", [1, 2, 4]),
        ("(a) x; (b) y", "paren_alpha", [1, 2]),
        ("First, to A; Second, to B; Twenty-first, to C", "ordinal_word", [1, 2, 21]),
    ],
)
def test_find_markers(text: str, style: str, values: list[int]) -> None:
    assert [m.value for m in find_markers(text, style)] == values


def test_markers_need_a_clause_boundary() -> None:
    # "Class A-1" and "Section 4(2)" are not steps.
    assert find_markers("Section 4(2) of the Act", "paren_number") == []
    assert find_markers("the Class A-1 notes", "number_dot") == []


def test_lowercase_ordinals_are_prose() -> None:
    assert find_markers("the first priority principal payment", "ordinal_word") == []


def test_marker_span_covers_the_parenthesis() -> None:
    (marker,) = find_markers("(3) to the servicer", "paren_number")
    assert (marker.start, marker.text) == (0, "(3)")


def test_first_marker_prefers_roman_for_i() -> None:
    marker = detect_first_marker("(i) accrued interest")
    assert marker is not None
    assert (marker.style, marker.value) == ("paren_roman", 1)


def test_first_marker_none_without_a_one() -> None:
    assert detect_first_marker("(2) second only") is None


def test_ordinal_words_cover_long_waterfalls() -> None:
    assert ORDINAL_WORDS["nineteenth"] == 19
    assert ORDINAL_WORDS["twentieth"] == 20
    assert ORDINAL_WORDS["twenty-ninth"] == 29
    assert ORDINAL_WORDS["thirtieth"] == 30


# --------------------------------------------------------------------------
# list form (Ford, Verizon): one step per block, marker lifted by Docling
# --------------------------------------------------------------------------
def test_numbered_list_under_heading_keeps_order_across_pages() -> None:
    (waterfall,) = _waterfalls(
        head("DESCRIPTION OF THE NOTES", level=1, page=81),
        head("Priority of Payments", level=2, page=82),
        para("On each payment date ... in the order of priority listed below:", page=82),
        item("to the indenture trustee, all amounts due,", marker="(1)", page=82),
        item("to the servicer, all unpaid servicing fees,", marker="(2)", page=82),
        item("to the Class A noteholders, interest due,", marker="(3)", page=82),
        item("to the reserve account, the amount required,", marker="(4)", page=82),
        item("to the residual holder, all remaining available funds.", marker="(5)", page=83),
        para("If available funds are insufficient to cover items (1) through (4), ...", page=83),
    )
    assert waterfall.title == "Priority of Payments"
    assert waterfall.form == "list"
    assert waterfall.section_path == ["DESCRIPTION OF THE NOTES", "Priority of Payments"]
    assert [s.ordinal for s in waterfall.steps] == [1, 2, 3, 4, 5]
    assert [s.marker for s in waterfall.steps] == ["(1)", "(2)", "(3)", "(4)", "(5)"]
    assert waterfall.steps[1].text == "to the servicer, all unpaid servicing fees,"
    assert (waterfall.start_page, waterfall.end_page) == (82, 83)
    assert waterfall.steps[-1].page == 83


def test_step_split_across_blocks_is_rejoined() -> None:
    (waterfall,) = _waterfalls(
        head("Priority of Payments", level=2),
        item("to the trustee, fees up to a maximum of", marker="(1)", page=5),
        para("$375,000 per year,", page=6),
        item("to the servicer, the servicing fee,", marker="(2)", page=6),
        item("to the noteholders, interest,", marker="(3)", page=6),
    )
    assert waterfall.steps[0].text == "to the trustee, fees up to a maximum of $375,000 per year,"
    assert waterfall.steps[0].page == 5


def test_nested_sub_list_stays_inside_its_step() -> None:
    (waterfall,) = _waterfalls(
        head("Priority of Payments", level=2),
        item("to the servicer, the servicing fee;", marker="(1)"),
        item("to the noteholders, principal as follows:", marker="(2)"),
        item("to the Class A-1 notes until paid in full;", marker="(a)"),
        item("to the Class A-2 notes until paid in full;", marker="(b)"),
        item("to the residual holder, the remainder.", marker="(3)"),
    )
    assert [s.marker for s in waterfall.steps] == ["(1)", "(2)", "(3)"]
    assert "(a) to the Class A-1 notes" in waterfall.steps[1].text
    assert "(b) to the Class A-2 notes" in waterfall.steps[1].text


def test_cross_reference_does_not_extend_the_list() -> None:
    (waterfall,) = _waterfalls(
        head("Priority of Payments", level=2),
        item("to A,", marker="(1)"),
        item("to B,", marker="(2)"),
        item("to C.", marker="(3)"),
        para("Amounts under (4) are paid from the reserve account."),
    )
    assert len(waterfall.steps) == 3


def test_sequence_gap_ends_the_list() -> None:
    (waterfall,) = _waterfalls(
        head("Priority of Payments", level=2),
        item("to A,", marker="(1)"),
        item("to B,", marker="(2)"),
        item("to C,", marker="(3)"),
        item("to E,", marker="(5)"),
    )
    assert [s.ordinal for s in waterfall.steps] == [1, 2, 3]


def test_too_short_is_not_a_waterfall() -> None:
    assert (
        _waterfalls(
            head("Priority of Payments", level=2), item("to A,", "(1)"), item("to B.", "(2)")
        )
        == []
    )


def test_heading_without_list_is_not_a_waterfall() -> None:
    assert _waterfalls(head("Priority of Payments", level=2), para("See the indenture.")) == []


def test_next_heading_ends_the_list() -> None:
    (waterfall,) = _waterfalls(
        head("Priority of Payments", level=2),
        item("to A,", "(1)"),
        item("to B,", "(2)"),
        item("to C,", "(3)"),
        head("Events of Default", level=2),
        item("to D,", "(4)"),
    )
    assert len(waterfall.steps) == 3


# --------------------------------------------------------------------------
# ordinal prose (CMBS): several steps inside one paragraph, found by lead-in
# --------------------------------------------------------------------------
def test_ordinal_prose_after_lead_in() -> None:
    (waterfall,) = _waterfalls(
        head("DISTRIBUTIONS ON THE CERTIFICATES", level=1, page=340),
        para(
            "On each distribution date, available funds will be distributed in the following "
            "order of priority: First, to the Class A-1 certificates, interest; Second, to the "
            "Class A-1 certificates, principal; Third, to the Class B certificates, interest; "
            "Fourth, to the Class B certificates, principal.",
            page=340,
            label="330",
        ),
    )
    assert waterfall.title.startswith("On each distribution date")
    assert waterfall.title.endswith("order of priority:")
    assert [s.marker for s in waterfall.steps] == ["First", "Second", "Third", "Fourth"]
    assert waterfall.steps[0].text == "to the Class A-1 certificates, interest;"
    assert waterfall.steps[3].text == "to the Class B certificates, principal."
    assert waterfall.steps[0].page_label == "330"


def test_inline_parenthesised_steps_in_one_paragraph() -> None:
    (waterfall,) = _waterfalls(
        para(
            "Collections will be applied in the following priority: (1) to the servicer, the "
            "fee; (2) to the noteholders, interest; and (3) to the certificateholders, the rest."
        ),
    )
    assert [s.text for s in waterfall.steps] == [
        "to the servicer, the fee;",
        "to the noteholders, interest; and",
        "to the certificateholders, the rest.",
    ]


def test_roman_steps_after_lead_in_with_nested_alpha() -> None:
    waterfalls = _waterfalls(
        para("will be distributed to the noteholders in the following order of priority:"),
        item("accrued and unpaid interest;", "(i)"),
        item("the principal distributable amount in the following order of priority:", "(ii)"),
        item("to the class A-1 noteholders until paid in full,", "(a)"),
        item("to the class A-2 noteholders until paid in full,", "(b)"),
        item("to the class A-3 noteholders until paid in full,", "(c)"),
        item("any remaining amounts to the certificateholders.", "(iii)"),
    )
    outer = next(w for w in waterfalls if w.steps[0].marker == "(i)")
    assert [s.marker for s in outer.steps] == ["(i)", "(ii)", "(iii)"]
    # The principal sub-waterfall is reported on its own as well.
    inner = next(w for w in waterfalls if w.steps[0].marker == "(a)")
    assert [s.marker for s in inner.steps] == ["(a)", "(b)", "(c)"]


def test_heading_and_lead_in_for_the_same_list_report_once() -> None:
    waterfalls = _waterfalls(
        head("Priority of Payments", level=2),
        para("Available funds will be paid in the order of priority listed below:"),
        item("to A,", "(1)"),
        item("to B,", "(2)"),
        item("to C.", "(3)"),
    )
    assert len(waterfalls) == 1
    assert waterfalls[0].title == "Priority of Payments"


def test_post_acceleration_waterfall_is_separate() -> None:
    waterfalls = _waterfalls(
        head("Priority of Payments", level=2, page=10),
        item("to A,", "(1)", page=10),
        item("to B,", "(2)", page=10),
        item("to C.", "(3)", page=10),
        head("Post-Acceleration Priority of Payments", level=2, page=12),
        item("to X,", "(1)", page=12),
        item("to Y,", "(2)", page=12),
        item("to Z.", "(3)", page=12),
    )
    assert [w.title for w in waterfalls] == [
        "Priority of Payments",
        "Post-Acceleration Priority of Payments",
    ]
    assert [s.text for s in waterfalls[1].steps] == ["to X,", "to Y,", "to Z."]


def test_lead_in_without_enumeration_is_ignored() -> None:
    assert (
        _waterfalls(para("Losses are allocated in the following order of priority: pro rata."))
        == []
    )


# --------------------------------------------------------------------------
# credit-card master trust layouts (Capital One sample)
# --------------------------------------------------------------------------
def test_lowercase_ordinal_list_items_after_as_follows() -> None:
    (waterfall,) = _waterfalls(
        head("Application of Card Series Finance Charge Amounts", level=2),
        para(
            "On each Distribution Date, the trustee will apply Finance Charge Amounts as follows:"
        ),
        item("first, to make the targeted deposits for Class A interest;"),
        item("second, to make the targeted deposits for Class B interest;"),
        item("third, to pay the servicing fee; and"),
        item("fourth, to the transferor."),
    )
    assert [s.marker for s in waterfall.steps] == ["first", "second", "third", "fourth"]
    assert waterfall.steps[2].text == "to pay the servicing fee; and"


def test_labelled_steps_keep_their_label() -> None:
    (waterfall,) = _waterfalls(
        para("The trustee will apply Principal Amounts in the following order and priority:"),
        item(
            "Class A Shortfalls. First, if Finance Charge Amounts are insufficient, the lesser of:"
        ),
        para("- the deficiency, and"),
        para("- the available subordinated amount."),
        item("Class B Shortfalls. Second, likewise for Class B."),
        item("Transferor. Third, the remainder to the transferor."),
    )
    first, second, third = waterfall.steps
    assert first.text.startswith("Class A Shortfalls. First, if Finance Charge Amounts")
    assert first.text.endswith("- the available subordinated amount.")
    assert second.text == "Class B Shortfalls. Second, likewise for Class B."
    assert third.text == "Transferor. Third, the remainder to the transferor."


def test_as_follows_needs_a_payment_verb() -> None:
    assert (
        _waterfalls(
            para("The Nominal Liquidation Amount of a note may be reduced as follows:"),
            item("first, by charge-offs;"),
            item("second, by reallocations;"),
            item("third, by payments."),
        )
        == []
    )


def test_prose_lowercase_ordinal_mid_sentence_is_not_a_marker() -> None:
    assert find_markers("amounts will first, be deposited", "ordinal_word") == []


# --------------------------------------------------------------------------
# table form
# --------------------------------------------------------------------------
def _table(rows: list[list[str]], caption: str | None = "Priority of Payments") -> Table:
    return Table(index=0, page=7, page_label="S-5", rows=rows, caption=caption, source="docling")


def test_table_steps_skip_header_and_keep_order() -> None:
    steps = table_steps(
        _table(
            [
                ["Priority", "Payee", "Amount"],
                ["1", "Trustee", "Fees and expenses"],
                ["2", "Servicer", "Servicing fee"],
                ["3", "Class A", "Interest"],
                ["", "Note: pro rata within a step", ""],
            ]
        )
    )
    assert [s.ordinal for s in steps] == [1, 2, 3]
    assert steps[0].text == "Trustee Fees and expenses"
    assert steps[2].page_label == "S-5"


def test_table_steps_with_parenthesised_markers() -> None:
    steps = table_steps(_table([["(1) Trustee fees"], ["(2) Servicing fee"], ["(3) Interest"]]))
    assert [s.text for s in steps] == ["Trustee fees", "Servicing fee", "Interest"]


def test_table_without_sequence_is_not_a_waterfall() -> None:
    assert table_steps(_table([["Class", "Balance"], ["A-1", "$250,000,000"], []])) == []


def _table_in_section(title: str, caption: str | None) -> list:  # type: ignore[type-arg]
    doc = blocks(head(title, level=2, page=7), para("see table", page=7))
    doc[1] = doc[1].model_copy(update={"table_index": 0})
    table = _table([["1", "0.95%"], ["2", "1.10%"], ["3", "1.24%"]], caption=caption)
    return find_waterfalls(doc, [table], chunk_by_section(doc))


def test_numbered_data_table_outside_a_waterfall_section_is_ignored() -> None:
    # Static-pool tables count months 1, 2, 3 in their first column; the
    # Ford sample has eighteen of them. They are data, not payment priorities.
    assert _table_in_section("Static Pool Information", caption=None) == []
    assert _table_in_section("Weighted Average Life", caption="Percent of Balance") == []


def test_numbered_table_captioned_as_priority_counts_anywhere() -> None:
    (waterfall,) = _table_in_section("Summary", caption="Priority of Payments")
    assert waterfall.form == "table"


def test_table_waterfall_is_found_in_its_section() -> None:
    doc = blocks(head("Application of Available Funds", level=2, page=7), para("see table", page=7))
    table_block = doc[1].model_copy(update={"table_index": 0})
    doc[1] = table_block
    table = _table([["1", "Trustee"], ["2", "Servicer"], ["3", "Class A"]], caption=None)
    (waterfall,) = find_waterfalls(doc, [table], chunk_by_section(doc))
    assert waterfall.form == "table"
    assert waterfall.title == "Application of Available Funds"
    assert waterfall.section_path == ["Application of Available Funds"]
