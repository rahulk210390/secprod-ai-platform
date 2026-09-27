"""Deterministic checks on extracted deal terms (JOB-04).

Nothing here asks the model anything. A deal fails validation, and is routed
to ``needs_review``, when:

* a required field is missing;
* the tranche balances do not add up to the total the document prints;
* a value is outside a plausible range (a 45% coupon, a 0.0135% threshold);
* a populated field has no citation, meaning its quoted text was not found in
  the document (the model may have invented or reworded it);
* two tranches share a class, or a coupon is incomplete.

Warnings are recorded but do not fail the deal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from secai.jobs.term_extraction.normalise import class_key
from secai.schemas.deal_terms import (
    CouponType,
    CoverageTestType,
    DealTerms,
)

RATE_RANGE: Final = (Decimal(0), Decimal(25))
MARGIN_RANGE: Final = (Decimal(-1), Decimal(10))
CLEANUP_RANGE: Final = (Decimal(0), Decimal(25))
THRESHOLD_RANGE: Final = (Decimal(0), Decimal(100))
WAL_RANGE: Final = (Decimal(0), Decimal(40))
COVERAGE_RANGE: Final = {
    CoverageTestType.OC: (Decimal(100), Decimal(300)),
    CoverageTestType.IC: (Decimal(100), Decimal(500)),
}
EARLIEST_CLOSING: Final = date(1990, 1, 1)
REQUIRED: Final = ("deal_name", "closing_date", "asset_class", "payment_frequency")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    field: str
    message: str
    severity: Severity = Severity.ERROR


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def needs_review(self) -> bool:
        return not self.passed

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}


def populated_fields(terms: DealTerms) -> Iterable[tuple[str, dict[str, object]]]:
    """(path, citations) for every populated field that must carry a citation."""
    for name in (
        "deal_name",
        "closing_date",
        "asset_class",
        "payment_frequency",
        "cleanup_call_pct",
    ):
        if getattr(terms, name) is not None:
            yield name, dict(terms.citations)
    if terms.stated_total_balance is not None:
        yield "stated_total_balance", dict(terms.citations)
    for i, tranche in enumerate(terms.tranches):
        prefix = f"tranches[{i}]"
        yield f"{prefix}.original_balance", dict(tranche.citations)
        yield f"{prefix}.coupon", dict(tranche.citations)
        for optional in ("rating", "wal_years"):
            if getattr(tranche, optional) is not None:
                yield f"{prefix}.{optional}", dict(tranche.citations)
    for i, trigger in enumerate(terms.triggers):
        yield f"triggers[{i}].trigger_type", dict(trigger.citations)
        if trigger.thresholds_pct:
            yield f"triggers[{i}].thresholds_pct", dict(trigger.citations)
    for i, test in enumerate(terms.coverage_tests):
        yield f"coverage_tests[{i}].threshold_pct", dict(test.citations)


def _citation_key(path: str) -> str:
    """ "tranches[2].coupon" → "coupon"; "closing_date" → "closing_date"."""
    return path.rsplit(".", 1)[-1]


def citation_coverage(terms: DealTerms) -> tuple[int, int]:
    """(cited, populated) field counts. The KPI is cited / populated == 1.0."""
    fields = list(populated_fields(terms))
    cited = sum(1 for path, citations in fields if _citation_key(path) in citations)
    # Payment steps carry their own page; they are cited by construction.
    return cited, len(fields)


def _range(
    issues: list[Issue], field: str, value: Decimal | None, bounds: tuple[Decimal, Decimal]
) -> None:
    low, high = bounds
    if value is not None and not (low <= value <= high):
        issues.append(
            Issue(code="out_of_range", field=field, message=f"{value} outside [{low}, {high}]")
        )


def validate_deal_terms(
    terms: DealTerms,
    *,
    extra: Iterable[Issue] = (),
    today: date | None = None,
) -> ValidationReport:
    """Check ``terms``; ``extra`` carries issues found while assembling them (unparsed values)."""
    issues: list[Issue] = list(extra)
    today = today or date.today()

    for name in REQUIRED:
        if getattr(terms, name) is None:
            issues.append(Issue(code="missing_required", field=name, message=f"{name} is missing"))
    if not terms.tranches:
        issues.append(Issue(code="missing_required", field="tranches", message="no tranches"))
    if not terms.priority_of_payments:
        issues.append(
            Issue(code="missing_required", field="priority_of_payments", message="no waterfall")
        )

    # --- the tranches add up to what the document prints ---------------------
    if terms.tranches:
        if terms.stated_total_balance is None:
            issues.append(
                Issue(
                    code="no_stated_total",
                    field="stated_total_balance",
                    message="no printed total to check the tranche sum against",
                    severity=Severity.WARNING,
                )
            )
        elif terms.total_original_balance != terms.stated_total_balance:
            issues.append(
                Issue(
                    code="tranche_sum_mismatch",
                    field="tranches",
                    message=(
                        f"tranches sum to {terms.total_original_balance:,} but the document "
                        f"prints {terms.stated_total_balance:,}"
                    ),
                )
            )

    keys = Counter(class_key(t.class_name) for t in terms.tranches)
    for key, count in keys.items():
        if count > 1:
            issues.append(
                Issue(code="duplicate_tranche", field="tranches", message=f"class {key} x{count}")
            )

    # --- plausible ranges -------------------------------------------------------
    for i, t in enumerate(terms.tranches):
        prefix = f"tranches[{i}]"
        if t.original_balance <= 0:
            issues.append(
                Issue(
                    code="out_of_range",
                    field=f"{prefix}.original_balance",
                    message="balance must be positive",
                )
            )
        _range(issues, f"{prefix}.fixed_rate_pct", t.fixed_rate_pct, RATE_RANGE)
        _range(issues, f"{prefix}.margin_pct", t.margin_pct, MARGIN_RANGE)
        _range(issues, f"{prefix}.wal_years", t.wal_years, WAL_RANGE)
        if t.coupon_type is CouponType.FIXED and t.fixed_rate_pct is None:
            issues.append(
                Issue(
                    code="incomplete_coupon",
                    field=f"{prefix}.coupon",
                    message="fixed coupon without a rate",
                )
            )
        if t.coupon_type is CouponType.FLOATING and not t.index:
            issues.append(
                Issue(
                    code="incomplete_coupon",
                    field=f"{prefix}.coupon",
                    message="floating coupon without an index",
                )
            )
    _range(issues, "cleanup_call_pct", terms.cleanup_call_pct, CLEANUP_RANGE)
    for i, trigger in enumerate(terms.triggers):
        for value in trigger.thresholds_pct:
            _range(issues, f"triggers[{i}].thresholds_pct", value, THRESHOLD_RANGE)
    for i, test in enumerate(terms.coverage_tests):
        _range(
            issues,
            f"coverage_tests[{i}].threshold_pct",
            test.threshold_pct,
            COVERAGE_RANGE[test.test_type],
        )
    if terms.closing_date is not None and not (
        EARLIEST_CLOSING <= terms.closing_date <= today + timedelta(days=730)
    ):
        issues.append(
            Issue(
                code="out_of_range",
                field="closing_date",
                message=f"{terms.closing_date} implausible",
            )
        )

    # --- every populated field is cited ----------------------------------------
    for path, citations in populated_fields(terms):
        if _citation_key(path) not in citations:
            issues.append(
                Issue(
                    code="uncited",
                    field=path,
                    message="value not found in the document; it may be invented or reworded",
                )
            )

    return ValidationReport(issues=issues)
