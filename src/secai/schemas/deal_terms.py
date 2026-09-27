"""Deal terms extracted from an offering document (JOB-04).

Two layers, deliberately separate:

* the ``Raw*`` models are what the LLM fills in. Every value is a *string
  copied verbatim* from the document ("$320,400,000", "4.057%",
  "30-day average SOFR + 0.60%"). The model never converts units or does
  arithmetic, which is how a small model turns 1.35% into 0.0135;
* :class:`DealTerms` is what the platform uses. Deterministic code parses the
  raw strings into typed values and attaches a :class:`Citation` found by
  searching the document for the verbatim text, so a value that is not in the
  source cannot be cited and is flagged.

Units: rates and thresholds are **percent** (``Decimal("4.057")`` means
4.057%); balances are currency units (``Decimal("320400000")``).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------
class AssetClass(StrEnum):
    AUTO_LOAN = "auto_loan"
    AUTO_LEASE = "auto_lease"
    EQUIPMENT = "equipment"
    CREDIT_CARD = "credit_card"
    DEVICE_PAYMENT = "device_payment"
    STUDENT_LOAN = "student_loan"
    RMBS = "rmbs"
    CMBS = "cmbs"
    CLO = "clo"
    OTHER = "other"


class PaymentFrequency(StrEnum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


class CouponType(StrEnum):
    FIXED = "fixed"
    FLOATING = "floating"


class CoverageTestType(StrEnum):
    OC = "overcollateralization"
    IC = "interest_coverage"


# --------------------------------------------------------------------------
# what the LLM returns: verbatim strings only
# --------------------------------------------------------------------------
class RawTranche(BaseModel):
    class_name: str = Field(description="as printed, e.g. 'Class A-1 notes'")
    original_balance: str = Field(description="as printed, e.g. '$320,400,000'")
    interest_rate: str = Field(description="as printed, e.g. '4.057%' or 'SOFR + 0.60%'")
    rating: str | None = Field(default=None, description="as printed, e.g. 'AAA(sf)'")
    weighted_average_life: str | None = Field(default=None, description="as printed, in years")


class RawTranches(BaseModel):
    tranches: list[RawTranche] = Field(max_length=30)
    stated_total: str | None = Field(
        default=None, description="the total principal of the classes, only if printed"
    )


class RawDealFacts(BaseModel):
    deal_name: str | None = Field(default=None, description="issuing entity, as printed")
    closing_date: str | None = Field(default=None, description="as printed")
    payment_frequency: Literal["monthly", "quarterly", "semi_annual", "annual"] | None = None
    payment_frequency_text: str | None = Field(
        default=None, description="the words that state the frequency, as printed"
    )
    cleanup_call: str | None = Field(
        default=None, description="the clean-up call threshold sentence or percentage, as printed"
    )
    asset_class: Literal[
        "auto_loan",
        "auto_lease",
        "equipment",
        "credit_card",
        "device_payment",
        "student_loan",
        "rmbs",
        "cmbs",
        "clo",
        "other",
    ]
    asset_class_evidence: str | None = Field(
        default=None,
        description=(
            "the words describing the assets, as printed, "
            "e.g. 'motor vehicle retail installment sale contracts'"
        ),
    )


class RawTrigger(BaseModel):
    trigger_type: str = Field(description="the name the document uses, e.g. 'Delinquency Trigger'")
    metric: str = Field(description="what is measured, in the document's words")
    threshold: str | None = Field(default=None, description="the limit as printed, e.g. '5.00%'")
    consequence: str = Field(description="what happens when it is breached, briefly")


class RawTriggers(BaseModel):
    triggers: list[RawTrigger] = Field(max_length=20)


class RawCoverageTest(BaseModel):
    tranche: str = Field(description="the class the test protects, as printed")
    test_type: Literal["overcollateralization", "interest_coverage"]
    threshold: str = Field(description="the required ratio as printed, e.g. '125.0%'")


class RawCoverageTests(BaseModel):
    tests: list[RawCoverageTest] = Field(max_length=30)


# --------------------------------------------------------------------------
# what the platform uses: typed, cited
# --------------------------------------------------------------------------
class Citation(_Frozen):
    """Where a value was found. ``quote`` is the verbatim source text."""

    quote: str
    page: int = Field(ge=1, description="1-based PDF page")
    page_label: str | None = Field(default=None, description="page number as printed")


class Tranche(_Frozen):
    class_name: str
    original_balance: Decimal = Field(ge=0)
    coupon_type: CouponType
    fixed_rate_pct: Decimal | None = None
    index: str | None = Field(default=None, description="floating-rate index, e.g. 'SOFR'")
    margin_pct: Decimal | None = None
    rating: str | None = None
    wal_years: Decimal | None = None
    citations: dict[str, Citation] = Field(default_factory=dict)


class CoverageTest(_Frozen):
    tranche: str
    test_type: CoverageTestType
    threshold_pct: Decimal
    citations: dict[str, Citation] = Field(default_factory=dict)


class TriggerEvent(_Frozen):
    trigger_type: str
    metric: str
    # A list because thresholds are often schedules: Ford's delinquency trigger
    # is 0.80% for 12 months, then 1.30%, 2.40% and 4.55%.
    thresholds_pct: list[Decimal] = Field(default_factory=list)
    threshold_text: str | None = None
    consequence: str
    citations: dict[str, Citation] = Field(default_factory=dict)


class PaymentStep(_Frozen):
    ordinal: int = Field(ge=1)
    text: str
    page: int = Field(ge=1)
    page_label: str | None = None


class DealTerms(_Frozen):
    """The extracted terms of one deal. Every populated field is cited."""

    deal_name: str | None = None
    closing_date: date | None = None
    asset_class: AssetClass | None = None
    payment_frequency: PaymentFrequency | None = None
    cleanup_call_pct: Decimal | None = Field(
        default=None, description="clean-up call threshold, % of initial pool balance"
    )
    tranches: list[Tranche] = Field(default_factory=list)
    stated_total_balance: Decimal | None = Field(
        default=None, description="the total the document prints for the tranche table"
    )
    coverage_tests: list[CoverageTest] = Field(default_factory=list)
    triggers: list[TriggerEvent] = Field(default_factory=list)
    priority_of_payments: list[PaymentStep] = Field(default_factory=list)
    citations: dict[str, Citation] = Field(default_factory=dict)

    @property
    def total_original_balance(self) -> Decimal:
        return sum((t.original_balance for t in self.tranches), Decimal(0))
