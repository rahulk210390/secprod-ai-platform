"""Printed text → typed values. Deterministic, and the only place it happens.

The LLM copies values exactly as printed; these functions turn "$320,400,000"
into ``Decimal("320400000")`` and "30-day average SOFR + 0.60%" into a floating
coupon. Keeping unit handling out of the model is the fix for JOB-02's
observation that a 3B model rewrote "1.35%" as "0.0135".

Every parser returns ``None`` on text it does not recognise, rather than
guessing: an unparsed value goes to review, a wrong value would not.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Final, NamedTuple

from secai.schemas.deal_terms import CouponType, PaymentFrequency

_MULTIPLIERS: Final = {
    "thousand": Decimal(1_000),
    "million": Decimal(1_000_000),
    "mm": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "bn": Decimal(1_000_000_000),
}
_MONEY = re.compile(
    r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>thousand|million|billion|mm|bn)?\b",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent\b|per\s*cent\b)", re.IGNORECASE)
# Index names, plus the generic words issuers use instead of naming one:
# John Deere prints "Benchmark + 0.34%", which would otherwise read as a fixed
# 0.34% coupon.
_FLOATING = re.compile(
    r"\b(SOFR|LIBOR|EURIBOR|SONIA|ESTR|BBSW|CDOR|prime rate|treasury|index|benchmark"
    r"|floating rate)\b",
    re.IGNORECASE,
)
_SPREAD_SPLIT = re.compile(r"\s*(?:\+|\bplus\b)\s*", re.IGNORECASE)
_MONTH_DATE = re.compile(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2}),?\s+(\d{4})\b")
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", re.IGNORECASE)
# "Class A-2b", and card-trust tranches with a series in brackets: "Class A(2025-2)".
_CLASS = re.compile(r"\bclass\s+([A-Z0-9][A-Z0-9\-]*(?:\([^)\s]*\))?)", re.IGNORECASE)
_FOOTNOTE = re.compile(r"\(\d+\)|\(\d+$")


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def parse_money(text: str | None) -> Decimal | None:
    """The first amount in ``text``: "$ 94,800,000" → 94800000; "$1.2 billion" → 1200000000.

    A bare small number ("15") is not an amount; a real balance has thousands
    separators or a unit word.
    """
    if not text:
        return None
    for match in _MONEY.finditer(text):
        number, unit = match.group("number"), match.group("unit")
        if "," not in number and unit is None:
            continue
        value = _decimal(number)
        if value is None:
            continue
        return value * _MULTIPLIERS[unit.lower()] if unit else value
    return None


def parse_percent(text: str | None) -> Decimal | None:
    """The first percentage: "4.02% per year" → 4.02. Stays in percent units."""
    if not text:
        return None
    match = _PERCENT.search(text)
    return _decimal(match.group(1)) if match else None


def parse_percents(text: str | None) -> list[Decimal]:
    """Every percentage, in order: a threshold schedule "0.80% ... then 1.30%" → [0.80, 1.30]."""
    if not text:
        return []
    values = (_decimal(match.group(1)) for match in _PERCENT.finditer(text))
    return [value for value in values if value is not None]


class Coupon(NamedTuple):
    coupon_type: CouponType
    fixed_rate_pct: Decimal | None
    index_name: str | None
    margin_pct: Decimal | None


def parse_coupon(text: str | None) -> Coupon | None:
    """Fixed ("4.057%") or floating ("30-day average SOFR + 0.60%") coupon."""
    if not text:
        return None
    if _FLOATING.search(text):
        parts = _SPREAD_SPLIT.split(text, maxsplit=1)
        index = parts[0].strip(" ,;") or None
        margin = parse_percent(parts[1]) if len(parts) == 2 else None
        return Coupon(CouponType.FLOATING, None, index, margin)
    rate = parse_percent(text)
    if rate is None:
        return None
    return Coupon(CouponType.FIXED, rate, None, None)


def parse_date(text: str | None) -> date | None:
    """ "September 16, 2025", "Sept. 16 2025", "09/16/2025" or "2025-09-16"."""
    if not text:
        return None
    match = _MONTH_DATE.search(text)
    if match:
        month, day, year = match.groups()
        # Full month name first, then the 3-letter form ("Sept." → "Sep").
        for candidate, fmt in (
            (f"{month} {day} {year}", "%B %d %Y"),
            (f"{month[:3]} {day} {year}", "%b %d %Y"),
        ):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    match = _NUMERIC_DATE.search(text)
    if match:
        month_n, day_n, year_n = (int(g) for g in match.groups())
        try:
            return date(year_n, month_n, day_n)
        except ValueError:
            return None
    match = _ISO_DATE.search(text)
    if match:
        try:
            return date.fromisoformat(match.group(0))
        except ValueError:
            return None
    return None


def parse_years(text: str | None) -> Decimal | None:
    """A weighted average life: "2.51 years" or "2.51" → 2.51."""
    if not text:
        return None
    match = _YEARS.search(text)
    return _decimal(match.group(1)) if match else None


_FREQUENCY_WORDS: Final = (
    (
        re.compile(r"\bsemi-?annual|twice a year|every six months\b", re.I),
        PaymentFrequency.SEMI_ANNUAL,
    ),
    (re.compile(r"\bquarter", re.I), PaymentFrequency.QUARTERLY),
    (re.compile(r"\bannual(ly)?\b|once a year", re.I), PaymentFrequency.ANNUAL),
    (re.compile(r"\bmonth", re.I), PaymentFrequency.MONTHLY),
)


def parse_frequency(text: str | None) -> PaymentFrequency | None:
    """ "the 15th day of each month" → monthly."""
    if not text:
        return None
    for pattern, frequency in _FREQUENCY_WORDS:
        if pattern.search(text):
            return frequency
    return None


def clean_class_name(text: str) -> str:
    """ "Class A-2b notes (1)(2" → "Class A-2b notes": footnote markers dropped."""
    return re.sub(r"\s+", " ", _FOOTNOTE.sub("", text)).strip()


def class_key(text: str) -> str:
    """A comparable key for a tranche: "Class A-2b notes" and "A-2B" → "A-2B"."""
    cleaned = clean_class_name(text)
    match = _CLASS.search(cleaned)
    token = match.group(1) if match else cleaned.split()[0] if cleaned else ""
    return token.upper()
