"""PII masking tests (JOB-01).

Two rules are being defended here:

* borrower-level data must never survive masking, in any shape;
* deal-level terms — balances, thresholds, triggers — must survive *intact*,
  because redacting them would make the traces useless.
"""

from __future__ import annotations

import json

import pytest

from secai.config import PIISettings
from secai.telemetry.pii import Masker, contains_pii, mask, reset_masker


@pytest.fixture(autouse=True)
def _fresh_masker():
    reset_masker()
    yield
    reset_masker()


@pytest.fixture
def masker() -> Masker:
    return Masker.from_settings(PIISettings())


REDACTED = "[REDACTED]"


class TestSensitiveKeys:
    @pytest.mark.parametrize(
        "key",
        [
            "borrower_id",
            "borrower_name",
            "borrowerName",
            "BORROWER_ADDRESS",
            "account_number",
            "loan_number",
            "ssn",
            "national_id",
            "email",
            "phone",
            "obligor_name",
            "date_of_birth",
            "primary_borrower",
        ],
    )
    def test_borrower_level_keys_are_sensitive(self, masker: Masker, key: str) -> None:
        assert masker.is_sensitive_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "tranche_balance",
            "original_balance",
            "oc_threshold",
            "ic_threshold",
            "coupon_margin",
            "deal_name",
            "closing_date",
            "asset_class",
            "priority_of_payments",
            "cpr",
        ],
    )
    def test_deal_level_keys_are_not_sensitive(self, masker: Masker, key: str) -> None:
        # Deal terms are explicitly *not* PII (CLAUDE.md 6.5).
        assert masker.is_sensitive_key(key) is False


class TestPatterns:
    @pytest.mark.parametrize(
        "text",
        [
            "contact jane.doe@example.com for details",
            "SSN 123-45-6789 on file",
            "IBAN GB29NWBK60161331926819 confirmed",
            "account no. 1234567890123",
            "loan no. 4485-2938-1029-3847",
            "call +44 20 7946 0958",
        ],
    )
    def test_patterns_are_redacted(self, masker: Masker, text: str) -> None:
        masked = masker.mask_text(text)
        assert REDACTED in masked
        assert not contains_pii(masked)

    @pytest.mark.parametrize(
        "text",
        [
            "Class A original balance 250,000,000",
            "OC test threshold of 115.5%",
            "closing date 2024-03-15",
            "coupon SOFR + 1.35%",
            "WAL 4.2 years",
            "pool factor 0.8734",
        ],
    )
    def test_deal_figures_survive(self, masker: Masker, text: str) -> None:
        # A masker that eats tranche balances is worse than useless.
        assert masker.mask_text(text) == text


class TestFinancialFiguresSurvive:
    """Regression: the smoke run redacted a tranche balance.

    Langfuse serialises observation payloads to a JSON *string* before the mask
    hook runs, so the textual patterns see raw, unformatted integers. An
    over-broad "long digit run" rule ate ``250000000`` — the single most
    important field in the platform (JOB-04 grades critical-field F1 on it).
    """

    @pytest.mark.parametrize(
        "figure",
        [
            "50000000",  # 8 digits
            "250000000",  # 9 digits  <- the one that regressed
            "1250000000",  # 10 digits
            "25000000000",  # 11 digits
            "999999999999",  # 12 digits
        ],
    )
    def test_unformatted_balances_survive(self, masker: Masker, figure: str) -> None:
        assert masker.mask_text(figure) == figure
        assert not contains_pii(figure)

    def test_serialised_deal_document_survives(self, masker: Masker) -> None:
        document = json.dumps(
            {
                "deal_name": "SYNTH 2024-1",
                "closing_date": "2024-03-15",
                "tranches": [
                    {"class": "A", "original_balance": 250000000, "coupon_margin": 1.35},
                    {"class": "B", "original_balance": 50000000, "coupon_margin": 2.75},
                ],
                "oc_test_threshold": 115.5,
            }
        )
        assert masker.mask_text(document) == document

    def test_iso_dates_are_not_mistaken_for_account_numbers(self, masker: Masker) -> None:
        for date in ("2024-03-15", "2024 03 15", "1999-12-31"):
            assert masker.mask_text(date) == date

    @pytest.mark.parametrize(
        "identifier",
        [
            "4485-2938-1029-3847",
            "account no. 1234567890",
            "loan #A-55231099887",
            "card number 4485 2938 1029 3847",
        ],
    )
    def test_structured_identifiers_are_still_caught(self, masker: Masker, identifier: str) -> None:
        assert REDACTED in masker.mask_text(identifier)


class TestStructuralMasking:
    def test_nested_payload(self, masker: Masker) -> None:
        payload = {
            "deal_name": "SYNTH 2024-1",
            "tranches": [
                {"class": "A", "original_balance": 250_000_000, "coupon_margin": 1.35},
                {"class": "B", "original_balance": 50_000_000, "coupon_margin": 2.75},
            ],
            "loans": [
                {"borrower_name": "Jane Doe", "loan_number": "L-99887766", "balance": 312_500},
                {"borrower_name": "John Roe", "loan_number": "L-11223344", "balance": 415_000},
            ],
        }
        masked = masker.mask(payload)

        # Deal-level data intact.
        assert masked["deal_name"] == "SYNTH 2024-1"
        assert masked["tranches"][0]["original_balance"] == 250_000_000
        assert masked["tranches"][1]["coupon_margin"] == 2.75

        # Borrower-level data gone, but the record shape is preserved.
        for loan in masked["loans"]:
            assert loan["borrower_name"] == REDACTED
            assert loan["loan_number"] == REDACTED
            assert loan["balance"] in (312_500, 415_000)

        assert not contains_pii(masked)

    def test_pii_buried_in_free_text(self, masker: Masker) -> None:
        payload = {"note": "Escalated by adviser jane.doe@example.com re SSN 123-45-6789"}
        masked = masker.mask(payload)
        assert "jane.doe@example.com" not in masked["note"]
        assert "123-45-6789" not in masked["note"]

    def test_bytes_are_redacted_wholesale(self, masker: Masker) -> None:
        assert masker.mask(b"borrower blob") == REDACTED

    def test_scalars_pass_through(self, masker: Masker) -> None:
        assert masker.mask(42) == 42
        assert masker.mask(1.5) == 1.5
        assert masker.mask(None) is None
        assert masker.mask(True) is True

    def test_sets_become_lists(self, masker: Masker) -> None:
        assert sorted(masker.mask({"a", "b"})) == ["a", "b"]

    def test_deep_structures_are_truncated_not_recursed_forever(self, masker: Masker) -> None:
        payload: dict[str, object] = {"leaf": "ok"}
        for _ in range(40):
            payload = {"nested": payload}
        assert masker.mask(payload) is not None  # terminates

    def test_cyclic_structure_terminates(self, masker: Masker) -> None:
        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic
        assert masker.mask(cyclic) is not None


class TestMaskFunctionHook:
    def test_langfuse_hook_signature(self) -> None:
        # Langfuse calls this as mask(data=...) and may pass extra kwargs.
        result = mask(data={"borrower_id": "B-1", "tranche_balance": 100}, extra="ignored")
        assert result["borrower_id"] == REDACTED
        assert result["tranche_balance"] == 100


class TestConfigurability:
    def test_custom_redaction_token(self) -> None:
        masker = Masker.from_settings(PIISettings(redaction_token="<<gone>>"))
        assert masker.mask({"borrower_id": "B-1"})["borrower_id"] == "<<gone>>"

    def test_extra_configured_field(self) -> None:
        masker = Masker.from_settings(PIISettings(fields=("internal_ref",)))
        assert masker.is_sensitive_key("internal_ref") is True
