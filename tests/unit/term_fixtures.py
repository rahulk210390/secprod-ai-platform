"""A small synthetic deal, as a parsed document plus a correct raw extraction.

Every figure is invented. Printed page labels differ from PDF pages, as in the
real samples, so citation tests can tell the two apart.
"""

from __future__ import annotations

from typing import Any

from secai.jobs.term_extraction.assemble import Extraction
from secai.parsing.models import Block, BlockKind, ParsedDocument, Table
from secai.parsing.sections import chunk_by_section
from secai.parsing.waterfall import find_waterfalls
from secai.schemas.deal_terms import (
    RawDealFacts,
    RawTranche,
    RawTranches,
    RawTrigger,
    RawTriggers,
)

TRANCHE_TABLE = (
    "Class | Principal Amount | Interest Rate\n"
    "Class A-1 notes | $ 250,000,000 | 4.50%\n"
    "Class A-2 notes | $ 300,000,000 | SOFR + 0.45%\n"
    "Class B notes | $ 50,000,000 | 5.10%\n"
    "Total | $ 600,000,000 |"
)

# (kind, text, pdf page, printed label, level, marker)
_SPECS: list[tuple[BlockKind, str, int, str | None, int | None, str | None]] = [
    (BlockKind.HEADING, "SYNTHETIC AUTO OWNER TRUST 2099-1", 1, None, 1, None),
    (
        BlockKind.TEXT,
        "The trust's assets are motor vehicle retail installment sale contracts.",
        1,
        None,
        None,
        None,
    ),
    (
        BlockKind.TEXT,
        "The trust will pay interest and principal on the 15th day of each month.",
        1,
        None,
        None,
        None,
    ),
    (BlockKind.HEADING, "SUMMARY OF TERMS", 3, "1", 1, None),
    (BlockKind.TABLE, TRANCHE_TABLE, 3, "1", None, None),
    (BlockKind.HEADING, "Closing Date", 3, "1", 2, None),
    (
        BlockKind.TEXT,
        "The trust expects to issue the notes on or about March 18, 2026.",
        3,
        "1",
        None,
        None,
    ),
    (BlockKind.HEADING, "Clean Up Call", 4, "2", 2, None),
    (
        BlockKind.TEXT,
        "The servicer may purchase the receivables when the pool balance is "
        "10% or less of the initial pool balance.",
        4,
        "2",
        None,
        None,
    ),
    (BlockKind.HEADING, "Delinquency Trigger", 40, "38", 2, None),
    (
        BlockKind.TEXT,
        "The delinquency trigger will be 1.25% for the first 12 months and 2.50% thereafter.",
        40,
        "38",
        None,
        None,
    ),
    (BlockKind.HEADING, "DESCRIPTION OF THE NOTES", 50, "48", 1, None),
    (BlockKind.HEADING, "Priority of Payments", 50, "48", 2, None),
    (BlockKind.LIST_ITEM, "to the servicer, the servicing fee,", 50, "48", None, "(1)"),
    (BlockKind.LIST_ITEM, "to the noteholders, interest,", 50, "48", None, "(2)"),
    (BlockKind.LIST_ITEM, "to the noteholders, principal.", 51, "49", None, "(3)"),
]

TRANCHE_BLOCK = 4
FACT_BLOCKS = [0, 1, 2, 5, 6, 7, 8]
TRIGGER_BLOCKS = [9, 10]


def document() -> ParsedDocument:
    blocks = [
        Block(
            index=i,
            kind=kind,
            text=text,
            page=page,
            page_label=label,
            level=level,
            marker=marker,
            table_index=0 if kind is BlockKind.TABLE else None,
        )
        for i, (kind, text, page, label, level, marker) in enumerate(_SPECS)
    ]
    sections = chunk_by_section(blocks)
    table = Table(
        index=0,
        page=3,
        page_label="1",
        rows=[[cell.strip() for cell in line.split("|")] for line in TRANCHE_TABLE.splitlines()],
        source="docling",
    )
    return ParsedDocument(
        source="synthetic.pdf",
        sha256="0" * 64,
        parser="docling",
        page_count=51,
        blocks=blocks,
        tables=[table],
        sections=sections,
        waterfalls=find_waterfalls(blocks, [], sections),
        duration_s=0.0,
    )


def extraction() -> Extraction:
    """What a correct model would return: every value verbatim."""
    return Extraction(
        facts=[
            (
                RawDealFacts(
                    deal_name="SYNTHETIC AUTO OWNER TRUST 2099-1",
                    closing_date="March 18, 2026",
                    payment_frequency="monthly",
                    payment_frequency_text="15th day of each month",
                    cleanup_call="10% or less of the initial pool balance",
                    asset_class="auto_loan",
                    asset_class_evidence="motor vehicle retail installment sale contracts",
                ),
                FACT_BLOCKS,
            )
        ],
        tranches=RawTranches(
            tranches=[
                RawTranche(
                    class_name="Class A-1 notes",
                    original_balance="$250,000,000",
                    interest_rate="4.50%",
                ),
                RawTranche(
                    class_name="Class A-2 notes",
                    original_balance="$300,000,000",
                    interest_rate="SOFR + 0.45%",
                ),
                RawTranche(
                    class_name="Class B notes",
                    original_balance="$50,000,000",
                    interest_rate="5.10%",
                ),
            ],
            stated_total="$600,000,000",
        ),
        tranches_blocks=[TRANCHE_BLOCK],
        triggers=[
            (
                RawTriggers(
                    triggers=[
                        RawTrigger(
                            trigger_type="delinquency trigger",
                            metric="60+ day delinquencies",
                            threshold="1.25% for the first 12 months and 2.50% thereafter",
                            consequence="asset representations review",
                        )
                    ]
                ),
                TRIGGER_BLOCKS,
            )
        ],
    )


def gold() -> dict[str, Any]:
    return {
        "id": "synthetic",
        "deal_name": "Synthetic Auto Owner Trust 2099-1",
        "closing_date": "2026-03-18",
        "asset_class": "auto_loan",
        "payment_frequency": "monthly",
        "cleanup_call_pct": "10",
        "stated_total_balance": "600000000",
        "tranches": [
            {
                "class_key": "A-1",
                "original_balance": "250000000",
                "coupon_type": "fixed",
                "fixed_rate_pct": "4.50",
            },
            {
                "class_key": "A-2",
                "original_balance": "300000000",
                "coupon_type": "floating",
                "margin_pct": "0.45",
            },
            {
                "class_key": "B",
                "original_balance": "50000000",
                "coupon_type": "fixed",
                "fixed_rate_pct": "5.10",
            },
        ],
        "triggers": [{"trigger_type": "delinquency trigger", "thresholds_pct": ["1.25", "2.50"]}],
        "coverage_tests": [],
        "priority_of_payments": {"start_page": 50, "step_count": 3},
    }
