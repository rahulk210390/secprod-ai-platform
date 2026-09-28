"""Field-level scoring against gold."""

from __future__ import annotations

from decimal import Decimal

import term_fixtures as fx

from secai.jobs.term_extraction.assemble import assemble
from secai.jobs.term_extraction.evaluate import CRITICAL, Counts, EvalResult, evaluate
from secai.schemas.deal_terms import DealTerms, TriggerEvent


def _terms() -> DealTerms:
    return assemble(fx.document(), fx.extraction())[0]


def test_perfect_extraction_scores_one() -> None:
    result = evaluate(_terms(), fx.gold())
    assert result.field_f1 == 1.0
    assert result.critical_field_f1 == 1.0
    assert result.mismatches == []
    # Real positives, not a vacuous score: 3 balances, 3 coupons, 2 thresholds.
    assert result.critical.tp == 8


def test_deal_name_matches_ignoring_case() -> None:
    result = evaluate(_terms(), fx.gold())
    assert result.per_field["deal_name"].tp == 1


def test_wrong_coupon_is_a_false_positive_and_a_miss() -> None:
    terms = _terms()
    a1 = terms.tranches[0].model_copy(update={"fixed_rate_pct": Decimal("4.05")})
    terms = terms.model_copy(update={"tranches": [a1, *terms.tranches[1:]]})
    coupon = evaluate(terms, fx.gold()).per_field["tranche.coupon"]
    assert (coupon.tp, coupon.fp, coupon.fn) == (2, 1, 1)


def test_missing_and_extra_tranches() -> None:
    terms = _terms()
    extra = terms.tranches[2].model_copy(update={"class_name": "Class C notes"})
    terms = terms.model_copy(update={"tranches": [*terms.tranches[:2], extra]})
    balance = evaluate(terms, fx.gold()).per_field["tranche.original_balance"]
    # B is missing (fn), C is extra (fp).
    assert (balance.tp, balance.fp, balance.fn) == (2, 1, 1)


def test_threshold_schedule_scores_each_step() -> None:
    terms = _terms()
    partial = TriggerEvent(
        trigger_type="delinquency trigger",
        metric="x",
        thresholds_pct=[Decimal("1.25"), Decimal("3.00")],
        consequence="y",
    )
    result = evaluate(terms.model_copy(update={"triggers": [partial]}), fx.gold())
    thresholds = result.per_field["trigger.threshold"]
    assert (thresholds.tp, thresholds.fp, thresholds.fn) == (1, 1, 1)
    assert any("trigger.threshold" in m for m in result.mismatches)


def test_threshold_numeric_equality_ignores_trailing_zeros() -> None:
    terms = _terms()
    same = TriggerEvent(
        trigger_type="t",
        metric="m",
        thresholds_pct=[Decimal("1.250"), Decimal("2.5")],
        consequence="c",
    )
    result = evaluate(terms.model_copy(update={"triggers": [same]}), fx.gold())
    assert result.per_field["trigger.threshold"].tp == 2


def test_empty_terms_score_zero_on_everything_gold_has() -> None:
    result = evaluate(DealTerms(), fx.gold())
    assert result.critical_field_f1 == 0.0
    assert result.overall.tp == 0


def test_value_where_gold_has_none_is_a_false_positive() -> None:
    gold = fx.gold() | {"cleanup_call_pct": None}
    counts = evaluate(_terms(), gold).per_field["cleanup_call_pct"]
    assert (counts.tp, counts.fp, counts.fn) == (0, 1, 0)


def test_counts_and_merge() -> None:
    empty = Counts()
    assert (empty.precision, empty.recall, empty.f1) == (1.0, 1.0, 1.0)
    first, second = evaluate(_terms(), fx.gold()), evaluate(DealTerms(), fx.gold())
    total = EvalResult()
    total.merge(first)
    total.merge(second)
    assert 0.0 < total.critical_field_f1 < 1.0
    assert total.critical.tp == first.critical.tp
    assert {
        "tranche.original_balance",
        "tranche.coupon",
        "trigger.threshold",
        "coverage.threshold",
    } == CRITICAL
