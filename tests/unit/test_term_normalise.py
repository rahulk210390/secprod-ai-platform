"""Printed text → typed values. Every case here is a string from a real sample."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from secai.jobs.term_extraction.normalise import (
    class_key,
    clean_class_name,
    parse_coupon,
    parse_date,
    parse_frequency,
    parse_money,
    parse_percent,
    parse_years,
)
from secai.schemas.deal_terms import CouponType, PaymentFrequency


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$ 320,400,000", "320400000"),
        ("$94,800,000", "94800000"),
        ("47,370,000", "47370000"),
        ("$750,000,000", "750000000"),
        ("$1.2 billion", "1200000000"),
        ("$100 MM", "100000000"),
        ("$8,607,500.25", "8607500.25"),
    ],
)
def test_parse_money(text: str, expected: str) -> None:
    assert parse_money(text) == Decimal(expected)


@pytest.mark.parametrize("text", [None, "", "none", "Class A-1", "15", "2025"])
def test_parse_money_rejects_non_amounts(text: str | None) -> None:
    assert parse_money(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4.057%", "4.057"),
        ("4.02% per year", "4.02"),
        ("1.35%", "1.35"),
        ("10 percent", "10"),
        ("5.00 per cent", "5.00"),
    ],
)
def test_parse_percent_stays_in_percent_units(text: str, expected: str) -> None:
    # JOB-02's 3B model turned "1.35%" into 0.0135; the parser never does.
    assert parse_percent(text) == Decimal(expected)


def test_parse_percent_none() -> None:
    assert parse_percent("no rate here") is None
    assert parse_percent(None) is None


def test_fixed_coupon() -> None:
    coupon = parse_coupon("3.88%")
    assert coupon is not None
    assert coupon.coupon_type is CouponType.FIXED
    assert coupon.fixed_rate_pct == Decimal("3.88")
    assert coupon.index_name is None and coupon.margin_pct is None


@pytest.mark.parametrize(
    ("text", "index", "margin"),
    [
        ("30-day average SOFR + 0.60%", "30-day average SOFR", "0.60"),
        ("One-month Term SOFR plus 0.45%", "One-month Term SOFR", "0.45"),
        ("SOFR", "SOFR", None),
        # John Deere sample: a generic benchmark, not a fixed 0.34% coupon.
        ("Benchmark + 0.34%(1)", "Benchmark", "0.34"),
        ("Floating Rate Asset Backed Notes", "Floating Rate Asset Backed Notes", None),
    ],
)
def test_floating_coupon(text: str, index: str, margin: str | None) -> None:
    coupon = parse_coupon(text)
    assert coupon is not None
    assert coupon.coupon_type is CouponType.FLOATING
    assert coupon.index_name == index
    assert coupon.margin_pct == (Decimal(margin) if margin else None)
    assert coupon.fixed_rate_pct is None


def test_unrecognised_coupon_is_none() -> None:
    assert parse_coupon("to be determined") is None
    assert parse_coupon(None) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("September 16, 2025", date(2025, 9, 16)),
        ("on or about Sept. 16 2025", date(2025, 9, 16)),
        ("Oct 15, 2026", date(2026, 10, 15)),
        ("09/16/2025", date(2025, 9, 16)),
        ("2025-09-16", date(2025, 9, 16)),
    ],
)
def test_parse_date(text: str, expected: date) -> None:
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text", [None, "", "the closing date", "Smarch 40, 2025", "13/45/2025", "2025-13-40"]
)
def test_parse_date_rejects_garbage(text: str | None) -> None:
    assert parse_date(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the 15th day of each month", PaymentFrequency.MONTHLY),
        ("monthly", PaymentFrequency.MONTHLY),
        ("quarterly on each Distribution Date", PaymentFrequency.QUARTERLY),
        ("semi-annually", PaymentFrequency.SEMI_ANNUAL),
        ("annually", PaymentFrequency.ANNUAL),
        ("from time to time", None),
        (None, None),
    ],
)
def test_parse_frequency(text: str | None, expected: PaymentFrequency | None) -> None:
    assert parse_frequency(text) == expected


def test_parse_years() -> None:
    assert parse_years("2.51 years") == Decimal("2.51")
    assert parse_years("0.35") == Decimal("0.35")
    assert parse_years(None) is None
    assert parse_years("n/a") is None


def test_class_names() -> None:
    assert clean_class_name("Class A-2b notes (1)(2") == "Class A-2b notes"
    assert class_key("Class A-2b notes (1)(2") == "A-2B"
    assert class_key("A-2B") == "A-2B"
    assert class_key("Class A(2025-2) Notes") == "A(2025-2)"
    assert class_key("") == ""
