"""Field-level precision/recall/F1 of extracted deal terms against gold (JOB-04).

The minimal eval that JOB-05 will generalise. Rules, fixed before any model
output was seen:

* a scalar field is a true positive when it equals gold (after
  normalisation); a wrong value is both a false positive and a false negative;
  a value where gold has none is a false positive;
* tranches are matched on class ("A-2B"); an extra tranche's fields are false
  positives, a missing tranche's are false negatives;
* trigger and coverage-test thresholds are compared as multisets of numbers,
  so a threshold schedule (0.80%, 1.30%, 2.40%, 4.55%) scores each step;
* critical fields, per CLAUDE.md, are tranche balances, coupons, trigger
  thresholds and coverage-test thresholds.

With nothing to find and nothing claimed, F1 is reported as 1.0; the report
also carries the raw counts, so a vacuous score is visible as one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Final

from secai.jobs.term_extraction.normalise import class_key
from secai.schemas.deal_terms import CouponType, DealTerms, Tranche

CRITICAL: Final = frozenset(
    {"tranche.original_balance", "tranche.coupon", "trigger.threshold", "coverage.threshold"}
)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 1.0


@dataclass
class EvalResult:
    per_field: dict[str, Counts] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)

    def _total(self, names: set[str] | frozenset[str] | None = None) -> Counts:
        total = Counts()
        for name, counts in self.per_field.items():
            if names is None or name in names:
                total.add(counts)
        return total

    @property
    def overall(self) -> Counts:
        return self._total()

    @property
    def critical(self) -> Counts:
        return self._total(CRITICAL)

    @property
    def field_f1(self) -> float:
        return self.overall.f1

    @property
    def critical_field_f1(self) -> float:
        return self.critical.f1

    def merge(self, other: EvalResult) -> None:
        for name, counts in other.per_field.items():
            self.per_field.setdefault(name, Counts()).add(counts)
        self.mismatches.extend(other.mismatches)


def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _same(a: object, b: object) -> bool:
    if isinstance(a, Decimal) and isinstance(b, Decimal):
        return a == b  # Decimal("25") == Decimal("25.0")
    return a == b


class _Scorer:
    def __init__(self, sample: str) -> None:
        self.result = EvalResult()
        self.sample = sample

    def score(self, name: str, predicted: object, gold: object) -> None:
        counts = self.result.per_field.setdefault(name, Counts())
        if gold is None and predicted is None:
            return
        if gold is None:
            counts.fp += 1
        elif predicted is None:
            counts.fn += 1
        elif _same(predicted, gold):
            counts.tp += 1
            return
        else:
            counts.fp += 1
            counts.fn += 1
        self.result.mismatches.append(
            f"{self.sample}: {name} predicted={predicted!r} gold={gold!r}"
        )

    def multiset(self, name: str, predicted: list[Decimal], gold: list[Decimal]) -> None:
        counts = self.result.per_field.setdefault(name, Counts())
        p = Counter(v.normalize() for v in predicted)
        g = Counter(v.normalize() for v in gold)
        both = p & g
        counts.tp += sum(both.values())
        extra, missing = p - both, g - both
        counts.fp += sum(extra.values())
        counts.fn += sum(missing.values())
        if extra or missing:
            self.result.mismatches.append(
                f"{self.sample}: {name} extra={sorted(map(str, extra.elements()))} "
                f"missing={sorted(map(str, missing.elements()))}"
            )


def _name_matches(predicted: str | None, gold: str | None) -> str | None:
    """Deal names match when one contains the other, ignoring case and spacing."""
    if predicted is None or gold is None:
        return predicted
    p, g = " ".join(predicted.lower().split()), " ".join(gold.lower().split())
    return gold if g in p or p in g else predicted


def _coupon(t: Tranche | None) -> tuple[str, Decimal | None] | None:
    if t is None:
        return None
    if t.coupon_type is CouponType.FIXED:
        return ("fixed", t.fixed_rate_pct.normalize() if t.fixed_rate_pct is not None else None)
    return ("floating", t.margin_pct.normalize() if t.margin_pct is not None else None)


def _gold_coupon(g: dict[str, Any]) -> tuple[str, Decimal | None]:
    if g["coupon_type"] == "fixed":
        return ("fixed", Decimal(g["fixed_rate_pct"]).normalize())
    margin = g.get("margin_pct")
    return ("floating", Decimal(margin).normalize() if margin is not None else None)


def evaluate(terms: DealTerms, gold: dict[str, Any]) -> EvalResult:
    s = _Scorer(gold["id"])

    s.score(
        "deal_name", _name_matches(terms.deal_name, gold.get("deal_name")), gold.get("deal_name")
    )
    closing = gold.get("closing_date")
    s.score("closing_date", terms.closing_date, date.fromisoformat(closing) if closing else None)
    s.score("asset_class", terms.asset_class and str(terms.asset_class), gold.get("asset_class"))
    s.score(
        "payment_frequency",
        terms.payment_frequency and str(terms.payment_frequency),
        gold.get("payment_frequency"),
    )
    s.score("cleanup_call_pct", terms.cleanup_call_pct, _dec(gold.get("cleanup_call_pct")))
    s.score(
        "stated_total_balance", terms.stated_total_balance, _dec(gold.get("stated_total_balance"))
    )

    predicted = {class_key(t.class_name): t for t in terms.tranches}
    expected = {g["class_key"].upper(): g for g in gold["tranches"]}
    for key in sorted(set(predicted) | set(expected)):
        t, g = predicted.get(key), expected.get(key)
        s.score(
            "tranche.original_balance",
            t.original_balance if t else None,
            _dec(g["original_balance"]) if g else None,
        )
        s.score("tranche.coupon", _coupon(t), _gold_coupon(g) if g else None)
        s.score(
            "tranche.wal_years", t.wal_years if t else None, _dec(g.get("wal_years")) if g else None
        )
        s.score("tranche.rating", t.rating if t else None, g.get("rating") if g else None)

    s.multiset(
        "trigger.threshold",
        [v for t in terms.triggers for v in t.thresholds_pct],
        [Decimal(v) for t in gold["triggers"] for v in t["thresholds_pct"]],
    )
    s.multiset(
        "coverage.threshold",
        [t.threshold_pct for t in terms.coverage_tests],
        [Decimal(t["threshold_pct"]) for t in gold["coverage_tests"]],
    )

    pop = gold.get("priority_of_payments")
    got = (
        (terms.priority_of_payments[0].page, len(terms.priority_of_payments))
        if terms.priority_of_payments
        else None
    )
    s.score("priority_of_payments", got, (pop["start_page"], pop["step_count"]) if pop else None)
    return s.result
