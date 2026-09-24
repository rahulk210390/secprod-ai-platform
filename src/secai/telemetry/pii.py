"""PII masking applied before anything leaves the process.

Borrower-level data must never reach a span attribute (CLAUDE.md §6.5). Deal
level terms — tranche balances, triggers, thresholds — are *not* PII and are
deliberately left alone, since they are the whole point of the traces.

Two layers use this module:

* the Langfuse ``mask`` hook, which sees observation inputs and outputs;
* the Collector's ``attributes/pii`` processor, as defence in depth.

Masking is applied structurally (by field name) *and* textually (by pattern),
because borrower data turns up both as ``{"borrower_name": ...}`` and buried in
free text a model echoed back.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from re import Pattern
from typing import Any, Final

from secai.config import PIISettings, get_settings

# Substrings that mark a key as borrower-level. Matched case-insensitively
# against the normalised key, so "borrowerName" and "borrower_name" both hit.
_DEFAULT_KEY_HINTS: Final = (
    "borrower",
    "obligor",
    "ssn",
    "social_security",
    "national_id",
    "passport",
    "account_number",
    "acct_number",
    "loan_number",
    "loan_id",
    "email",
    "phone",
    "address",
    "postcode",
    "zip_code",
    "date_of_birth",
    "dob",
)

# Textual patterns.
#
# Langfuse serialises observation inputs and outputs to a JSON *string* before
# the mask hook runs, so these patterns see whole documents — deal terms
# included. That cuts both ways: a naive "long run of digits" rule redacts
# tranche balances (250000000 is nine digits), destroying the very data the
# traces exist to show. Account and loan numbers are therefore matched only
# where they are *structurally* identifiable — keyword-anchored or
# separator-grouped — never as bare integers. Bare identifiers are caught by
# field name instead (``_DEFAULT_KEY_HINTS``), which is how they actually
# appear in our payloads.
_PATTERNS: Final[tuple[tuple[str, Pattern[str]], ...]] = (
    ("email", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    # US SSN and similar 3-2-4 groupings.
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # IBAN.
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    # "account no. 1234567890", "loan #A-55231099887", "card number 4485 ...".
    (
        "keyed_identifier",
        re.compile(
            r"(?i)\b(?:account|acct|loan|card|policy|customer|member)\s*"
            r"(?:no\.?|number|num|id|#)?\s*[:#]?\s*"
            r"[A-Z]{0,4}-?\d[\d -]{4,}\d"
        ),
    ),
    # Phone numbers in international form.
    ("phone", re.compile(r"(?<![\w.])\+\d[\d ().-]{7,}\d(?![\w.])")),
)

# Separator-grouped digit runs (card/account style: 4485-2938-1029-3847).
# Validated in code rather than by the pattern alone, so ISO dates and
# formatted money survive — see :func:`_is_identifier_like`.
_GROUPED_DIGITS: Final = re.compile(r"\b\d{2,6}(?:[ -]\d{2,6}){2,}\b")

# Below this many digits, a grouped run is a date, a reference or a formatted
# figure rather than an account number.
_MIN_IDENTIFIER_DIGITS: Final = 10

_DATE_LIKE: Final = re.compile(r"(19|20)\d{2}[ -]\d{1,2}[ -]\d{1,2}")

_MAX_DEPTH: Final = 20


def _is_identifier_like(candidate: str) -> bool:
    """True if a separator-grouped digit run is plausibly an account number.

    Rejects ISO dates ("2024-03-15") and anything too short to be an account or
    card number. Comma-formatted money never reaches here, because the pattern
    accepts only space and hyphen separators.
    """
    digits = re.sub(r"\D", "", candidate)
    if len(digits) < _MIN_IDENTIFIER_DIGITS:
        return False
    return _DATE_LIKE.fullmatch(candidate) is None


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower())


@dataclass(frozen=True)
class Masker:
    """Applies field-name and pattern-based redaction to arbitrary payloads."""

    redaction_token: str
    field_names: frozenset[str]
    key_hints: tuple[str, ...] = _DEFAULT_KEY_HINTS

    @classmethod
    def from_settings(cls, settings: PIISettings | None = None) -> Masker:
        pii = settings if settings is not None else get_settings().pii
        return cls(
            redaction_token=pii.redaction_token,
            field_names=frozenset(_normalise_key(f) for f in pii.fields),
        )

    # -- key handling ------------------------------------------------------
    def is_sensitive_key(self, key: str) -> bool:
        """True if a field with this name must be redacted outright."""
        normalised = _normalise_key(key)
        if normalised in self.field_names:
            return True
        return any(hint in normalised for hint in self.key_hints)

    # -- value handling ----------------------------------------------------
    def mask_text(self, text: str) -> str:
        """Redact PII patterns inside a string, leaving the rest intact.

        Deal-level figures — tranche balances, thresholds, coupons — survive by
        design; see the note on ``_PATTERNS``.
        """
        masked = text
        for _, pattern in _PATTERNS:
            masked = pattern.sub(self.redaction_token, masked)
        return _GROUPED_DIGITS.sub(
            lambda m: self.redaction_token if _is_identifier_like(m.group()) else m.group(),
            masked,
        )

    def mask(self, data: Any, *, _depth: int = 0) -> Any:
        """Recursively mask a payload.

        Mappings are walked by key, sequences element-wise, and strings by
        pattern. Anything past ``_MAX_DEPTH`` is redacted wholesale rather than
        risking unbounded recursion on a cyclic or pathological structure.
        """
        if _depth > _MAX_DEPTH:
            return self.redaction_token

        if isinstance(data, str):
            return self.mask_text(data)

        if isinstance(data, Mapping):
            out: dict[Any, Any] = {}
            for key, value in data.items():
                if isinstance(key, str) and self.is_sensitive_key(key):
                    out[key] = self.redaction_token
                else:
                    out[key] = self.mask(value, _depth=_depth + 1)
            return out

        # bytes are sequences too; handle them before the generic branch.
        if isinstance(data, bytes | bytearray):
            return self.redaction_token

        if isinstance(data, Sequence):
            return [self.mask(item, _depth=_depth + 1) for item in data]

        if isinstance(data, set | frozenset):
            return [self.mask(item, _depth=_depth + 1) for item in data]

        # Numbers, bools, None and anything else opaque pass through: deal-level
        # figures are not PII and must stay readable in Langfuse.
        return data


_masker: Masker | None = None


def get_masker() -> Masker:
    """The process-wide masker, built from settings on first use."""
    global _masker
    if _masker is None:
        _masker = Masker.from_settings()
    return _masker


def reset_masker() -> None:
    """Drop the cached masker so new settings take effect. Tests use this."""
    global _masker
    _masker = None


def mask(*, data: Any, **_kwargs: Any) -> Any:
    """Langfuse ``MaskFunction`` entry point.

    The keyword-only ``data`` argument and the tolerated extra kwargs are
    dictated by the SDK's protocol.
    """
    return get_masker().mask(data)


def contains_pii(value: Any) -> bool:
    """True if any configured pattern still matches. Used by tests and asserts."""
    if isinstance(value, str):
        if any(pattern.search(value) for _, pattern in _PATTERNS):
            return True
        return any(_is_identifier_like(m.group()) for m in _GROUPED_DIGITS.finditer(value))
    if isinstance(value, Mapping):
        masker = get_masker()
        return any(
            (isinstance(k, str) and masker.is_sensitive_key(k) and v != masker.redaction_token)
            or contains_pii(v)
            for k, v in value.items()
        )
    if isinstance(value, bytes | bytearray):
        return False
    if isinstance(value, Sequence):
        return any(contains_pii(item) for item in value)
    return False
