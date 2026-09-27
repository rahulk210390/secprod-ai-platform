"""Assembly and validation, including the JOB-04 fault-injection acceptance test.

"Validation catches 100% of injected errors": each case below takes a correct
extraction, injects one realistic model mistake, and asserts the deal is
routed to needs_review with the expected reason.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest
import term_fixtures as fx

from secai.jobs.term_extraction.assemble import Extraction, assemble, main_waterfall
from secai.schemas.deal_terms import (
    CouponType,
    CoverageTest,
    CoverageTestType,
    PaymentFrequency,
    RawCoverageTest,
    RawCoverageTests,
    RawTranche,
)
from secai.validation.deal_terms import (
    Issue,
    Severity,
    citation_coverage,
    validate_deal_terms,
)

TODAY = date(2026, 9, 27)


def _run(extraction: Extraction):  # type: ignore[no-untyped-def]
    terms, issues = assemble(fx.document(), extraction)
    return terms, validate_deal_terms(terms, extra=issues, today=TODAY)


# --------------------------------------------------------------------------
# the clean path
# --------------------------------------------------------------------------
def test_correct_extraction_passes_with_every_field_cited() -> None:
    terms, report = _run(fx.extraction())

    assert report.passed, report.issues
    cited, populated = citation_coverage(terms)
    assert cited == populated and populated > 0

    assert terms.deal_name == "SYNTHETIC AUTO OWNER TRUST 2099-1"
    assert terms.closing_date == date(2026, 3, 18)
    assert terms.payment_frequency is PaymentFrequency.MONTHLY
    assert terms.cleanup_call_pct == Decimal("10")
    assert terms.total_original_balance == terms.stated_total_balance == Decimal("600000000")

    a1, a2, b = terms.tranches
    assert (a1.coupon_type, a1.fixed_rate_pct) == (CouponType.FIXED, Decimal("4.50"))
    assert (a2.coupon_type, a2.index, a2.margin_pct) == (
        CouponType.FLOATING,
        "SOFR",
        Decimal("0.45"),
    )
    assert b.original_balance == Decimal("50000000")

    (trigger,) = terms.triggers
    assert trigger.thresholds_pct == [Decimal("1.25"), Decimal("2.50")]


def test_citations_point_at_printed_pages() -> None:
    terms, _ = _run(fx.extraction())
    balance = terms.tranches[0].citations["original_balance"]
    assert (balance.page, balance.page_label) == (3, "1")
    assert terms.citations["cleanup_call_pct"].page_label == "2"
    assert terms.triggers[0].citations["thresholds_pct"].page_label == "38"
    assert terms.citations["closing_date"].quote == "March 18, 2026"


def test_priority_of_payments_comes_from_job_03_not_the_model() -> None:
    terms, _ = _run(fx.extraction())
    assert [s.ordinal for s in terms.priority_of_payments] == [1, 2, 3]
    assert terms.priority_of_payments[-1].page_label == "49"


def test_interest_only_and_residual_classes_are_excluded() -> None:
    extraction = fx.extraction()
    assert extraction.tranches is not None
    extra = [
        RawTranche(class_name="Class X-A", original_balance="$550,000,000", interest_rate="0.98%"),
        RawTranche(class_name="Class R", original_balance="NAP", interest_rate="NAP"),
    ]
    extraction.tranches = extraction.tranches.model_copy(
        update={"tranches": [*extraction.tranches.tranches, *extra]}
    )
    terms, report = _run(extraction)
    assert len(terms.tranches) == 3
    assert report.passed


def test_duplicate_triggers_from_overlapping_contexts_are_merged() -> None:
    extraction = fx.extraction()
    extraction.triggers = extraction.triggers * 2
    terms, _ = _run(extraction)
    assert len(terms.triggers) == 1


def test_coverage_tests_are_assembled_and_cited() -> None:
    document = fx.document()
    blocks = [
        *document.blocks,
        document.blocks[-1].model_copy(
            update={
                "index": len(document.blocks),
                "text": "Class A OC Test: 125.0%",
                "marker": None,
            }
        ),
    ]
    document = document.model_copy(update={"blocks": blocks})
    extraction = fx.extraction()
    extraction.coverage = [
        (
            RawCoverageTests(
                tests=[
                    RawCoverageTest(
                        tranche="Class A", test_type="overcollateralization", threshold="125.0%"
                    )
                ]
            ),
            [len(blocks) - 1],
        )
    ]
    terms, issues = assemble(document, extraction)
    (test,) = terms.coverage_tests
    assert test.threshold_pct == Decimal("125.0")
    assert test.citations["threshold_pct"].page_label == "49"
    assert validate_deal_terms(terms, extra=issues, today=TODAY).passed


def test_facts_merge_first_context_that_states_each_field() -> None:
    extraction = fx.extraction()
    full, blocks = extraction.facts[0]
    # A clean-up-only context read first: it knows the percentage, not the date,
    # and its asset-class guess has no evidence in the document.
    partial = full.model_copy(
        update={
            "deal_name": None,
            "closing_date": "the closing date",
            "cleanup_call": "10% or less of the initial pool balance",
            "asset_class": "rmbs",
            "asset_class_evidence": "residential mortgage loans",
        }
    )
    extraction.facts = [(partial, blocks), (full, blocks)]
    terms, report = _run(extraction)
    assert terms.deal_name == "SYNTHETIC AUTO OWNER TRUST 2099-1"
    assert terms.closing_date == date(2026, 3, 18)
    # The unparseable date in the first context is not an issue: another context had it.
    assert "unparsed" not in report.codes()
    # "rmbs" had no evidence in the document; the cited guess wins.
    assert str(terms.asset_class) == "auto_loan"
    assert report.passed


def test_asset_class_without_any_evidence_falls_back_to_first_guess() -> None:
    extraction = fx.extraction()
    facts, blocks = extraction.facts[0]
    extraction.facts = [
        (facts.model_copy(update={"asset_class_evidence": "made up words"}), blocks)
    ]
    terms, report = _run(extraction)
    assert str(terms.asset_class) == "auto_loan"
    assert any(i.field == "asset_class" and i.code == "uncited" for i in report.issues)


@pytest.mark.parametrize(
    "described",
    [None, "the delinquency trigger rate, as that rate may be adjusted"],
    ids=["null", "described-not-quoted"],
)
def test_described_threshold_is_recovered_from_the_context(described: str | None) -> None:
    extraction = fx.extraction()
    raw, blocks = extraction.triggers[0]
    trigger = raw.triggers[0].model_copy(update={"threshold": described})
    extraction.triggers = [(raw.model_copy(update={"triggers": [trigger]}), blocks)]

    terms, report = _run(extraction)
    (recovered,) = terms.triggers
    assert recovered.thresholds_pct == [Decimal("1.25"), Decimal("2.50")]
    # The number came from code, and is cited to the sentence it came from.
    assert recovered.citations["thresholds_pct"].page_label == "38"
    assert "threshold_from_context" in report.codes()
    assert report.passed  # recovery is a warning, not an error


def test_recovery_skips_voting_and_historical_sentences() -> None:
    from secai.jobs.term_extraction.assemble import recover_threshold
    from secai.parsing.models import Block, BlockKind

    text = (
        "Noteholders of at least 5% may demand a vote after a delinquency trigger. "
        "The historical peak rate before any delinquency trigger was 4.96%. "
        "The delinquency trigger will be 9.00%."
    )
    block = Block(index=0, kind=BlockKind.TEXT, text=text, page=1, page_label="1")
    result = recover_threshold("Delinquency Trigger", [block], [0])
    assert result is not None
    sentence, values = result
    assert values == [Decimal("9.00")] and sentence == "The delinquency trigger will be 9.00%."
    assert recover_threshold("x", [block], [0]) is None  # too short to search for
    assert recover_threshold("Pay Out Event", [block], [0, 5]) is None


def test_main_waterfall_none_when_document_has_none() -> None:
    assert main_waterfall([]) is None
    extraction = fx.extraction()
    terms, issues = assemble(fx.document().model_copy(update={"waterfalls": []}), extraction)
    report = validate_deal_terms(terms, extra=issues, today=TODAY)
    assert "missing_required" in report.codes() and "no_waterfall" in report.codes()


# --------------------------------------------------------------------------
# fault injection: every injected error must be caught
# --------------------------------------------------------------------------
def _tranche(i: int, **update: str) -> Callable[[Extraction], None]:
    def mutate(e: Extraction) -> None:
        assert e.tranches is not None
        items = list(e.tranches.tranches)
        items[i] = items[i].model_copy(update=update)
        e.tranches = e.tranches.model_copy(update={"tranches": items})

    return mutate


def _facts(**update: str | None) -> Callable[[Extraction], None]:
    def mutate(e: Extraction) -> None:
        facts, blocks = e.facts[0]
        e.facts = [(facts.model_copy(update=update), blocks)]

    return mutate


def _threshold(text: str) -> Callable[[Extraction], None]:
    def mutate(e: Extraction) -> None:
        raw, blocks = e.triggers[0]
        trigger = raw.triggers[0].model_copy(update={"threshold": text})
        e.triggers = [(raw.model_copy(update={"triggers": [trigger]}), blocks)]

    return mutate


def _add_tranche(e: Extraction) -> None:
    assert e.tranches is not None
    invented = RawTranche(
        class_name="Class C notes", original_balance="$40,000,000", interest_rate="6.00%"
    )
    e.tranches = e.tranches.model_copy(update={"tranches": [*e.tranches.tranches, invented]})


def _drop_tranche(e: Extraction) -> None:
    assert e.tranches is not None
    e.tranches = e.tranches.model_copy(update={"tranches": e.tranches.tranches[:2]})


def _duplicate_tranche(e: Extraction) -> None:
    assert e.tranches is not None
    items = e.tranches.tranches
    e.tranches = e.tranches.model_copy(update={"tranches": [*items, items[0]]})


INJECTIONS: list[tuple[str, Callable[[Extraction], None], set[str]]] = [
    (
        "balance off by one",
        _tranche(0, original_balance="$250,000,001"),
        {"uncited", "tranche_sum_mismatch"},
    ),
    ("balance in millions, unit dropped", _tranche(1, original_balance="300"), {"unparsed"}),
    ("rate as a fraction (JOB-02's 0.0135)", _tranche(0, interest_rate="0.045"), {"unparsed"}),
    ("rate reworded", _tranche(2, interest_rate="5.1 percent"), {"uncited"}),
    ("floating index dropped", _tranche(1, interest_rate="+ 0.45%"), {"unparsed"}),
    ("implausible coupon", _tranche(2, interest_rate="45.10%"), {"uncited", "out_of_range"}),
    ("invented tranche", _add_tranche, {"uncited", "tranche_sum_mismatch"}),
    ("missing tranche", _drop_tranche, {"tranche_sum_mismatch"}),
    ("duplicated tranche", _duplicate_tranche, {"duplicate_tranche", "tranche_sum_mismatch"}),
    (
        "wrong stated total",
        lambda e: setattr(
            e, "tranches", e.tranches.model_copy(update={"stated_total": "$650,000,000"})
        ),
        {"uncited", "tranche_sum_mismatch"},
    ),  # type: ignore[union-attr]
    ("closing date missing", _facts(closing_date=None), {"missing_required"}),
    (
        "closing date garbled",
        _facts(closing_date="the closing date"),
        {"unparsed", "missing_required"},
    ),
    ("closing date invented", _facts(closing_date="March 19, 2026"), {"uncited"}),
    ("clean-up call as 100%", _facts(cleanup_call="100%"), {"uncited", "out_of_range"}),
    ("deal name invented", _facts(deal_name="Other Trust 2099-9"), {"uncited"}),
    ("threshold without unit", _threshold("0.0125"), {"unparsed"}),
    ("threshold invented", _threshold("3.00% at all times"), {"uncited"}),
    ("threshold impossible", _threshold("125.0% of the pool"), {"uncited", "out_of_range"}),
]


@pytest.mark.parametrize(
    ("case", "inject", "expected"), INJECTIONS, ids=[c for c, _, _ in INJECTIONS]
)
def test_injected_error_is_caught(
    case: str, inject: Callable[[Extraction], None], expected: set[str]
) -> None:
    extraction = fx.extraction()
    inject(extraction)
    _, report = _run(extraction)
    assert report.needs_review, f"{case}: not caught"
    assert expected <= report.codes(), f"{case}: {report.codes()}"


def test_injection_catch_rate_is_100_percent() -> None:
    caught = 0
    for _, inject, _ in INJECTIONS:
        extraction = fx.extraction()
        inject(extraction)
        caught += _run(extraction)[1].needs_review
    assert caught == len(INJECTIONS)


# --------------------------------------------------------------------------
# validation rules in isolation
# --------------------------------------------------------------------------
def test_no_stated_total_is_only_a_warning() -> None:
    extraction = fx.extraction()
    assert extraction.tranches is not None
    extraction.tranches = extraction.tranches.model_copy(update={"stated_total": None})
    _, report = _run(extraction)
    assert report.passed
    assert [i.severity for i in report.issues if i.code == "no_stated_total"] == [Severity.WARNING]


def test_out_of_range_closing_date_and_coverage_threshold() -> None:
    terms, _ = _run(fx.extraction())
    bad = terms.model_copy(
        update={
            "closing_date": date(1980, 1, 1),
            "coverage_tests": [
                CoverageTest(
                    tranche="Class A", test_type=CoverageTestType.OC, threshold_pct=Decimal("50")
                )
            ],
        }
    )
    report = validate_deal_terms(bad, today=TODAY)
    fields = {i.field for i in report.errors}
    assert {"closing_date", "coverage_tests[0].threshold_pct"} <= fields


def test_incomplete_coupons_and_non_positive_balance() -> None:
    terms, _ = _run(fx.extraction())
    a1, a2, b = terms.tranches
    bad = terms.model_copy(
        update={
            "tranches": [
                a1.model_copy(update={"fixed_rate_pct": None}),
                a2.model_copy(update={"index": None}),
                b.model_copy(update={"original_balance": Decimal(0)}),
            ]
        }
    )
    report = validate_deal_terms(bad, today=TODAY)
    assert sum(i.code == "incomplete_coupon" for i in report.issues) == 2
    assert any(i.field == "tranches[2].original_balance" for i in report.errors)


def test_empty_terms_list_every_required_field() -> None:
    from secai.schemas.deal_terms import DealTerms

    report = validate_deal_terms(DealTerms(), today=TODAY)
    missing = {i.field for i in report.issues if i.code == "missing_required"}
    assert missing == {
        "deal_name",
        "closing_date",
        "asset_class",
        "payment_frequency",
        "tranches",
        "priority_of_payments",
    }
    assert citation_coverage(DealTerms()) == (0, 0)


def test_issue_defaults_to_error() -> None:
    assert Issue(code="x", field="y", message="z").severity is Severity.ERROR
