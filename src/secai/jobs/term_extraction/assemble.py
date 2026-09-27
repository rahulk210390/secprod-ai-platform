"""Raw model output + the parsed document → typed, cited :class:`DealTerms`.

Every number is parsed here by :mod:`normalise`, and every citation is found
here by :mod:`citations` in the document itself. A raw value that cannot be
parsed is dropped with an issue rather than guessed; a value whose quote cannot
be found stays, uncited, so validation can route the deal to review.

The priority of payments never goes through the model: it is JOB-03's
waterfall, whose step order is verified against gold.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from secai.jobs.term_extraction.citations import find_citation, normalise
from secai.jobs.term_extraction.normalise import (
    class_key,
    clean_class_name,
    parse_coupon,
    parse_date,
    parse_frequency,
    parse_money,
    parse_percent,
    parse_percents,
    parse_years,
)
from secai.parsing.models import Block, ParsedDocument, Waterfall
from secai.schemas.deal_terms import (
    AssetClass,
    Citation,
    CoverageTest,
    CoverageTestType,
    DealTerms,
    PaymentFrequency,
    PaymentStep,
    RawCoverageTests,
    RawDealFacts,
    RawTranches,
    RawTriggers,
    Tranche,
    TriggerEvent,
)
from secai.validation.deal_terms import Issue, Severity

# Interest-only classes carry a notional amount, not principal ("Class X-A"),
# and a residual class has none; neither belongs in the tranche sum.
_NON_PRINCIPAL: Final = re.compile(r"^(X(-|$)|R$)")
_SUMMARY: Final = re.compile(r"summary", re.IGNORECASE)
_BARE_MARGIN: Final = re.compile(r"^\s*(\+|plus\b)", re.IGNORECASE)
_DIGIT: Final = re.compile(r"\d")
_SENTENCE: Final = re.compile(r"(?<=[.;])\s+")
_NOT_A_THRESHOLD: Final = re.compile(
    r"\bvot(e|es|ed|ing)\b|quorum|holders? (of|evidencing|representing|holding)"
    r"|historical|ranged from|\bpeak\b|prior (securitized )?pools",
    re.IGNORECASE,
)
MIN_TRIGGER_NAME_CHARS: Final = 5


@dataclass
class Extraction:
    """Raw model outputs per target, with the block indices each one read."""

    # Several fact contexts; each field comes from the first that states it.
    facts: list[tuple[RawDealFacts, list[int]]] = field(default_factory=list)
    tranches: RawTranches | None = None
    tranches_blocks: list[int] = field(default_factory=list)
    triggers: list[tuple[RawTriggers, list[int]]] = field(default_factory=list)
    coverage: list[tuple[RawCoverageTests, list[int]]] = field(default_factory=list)


def _cite(
    citations: dict[str, Citation],
    key: str,
    quote: str | None,
    blocks: Sequence[Block],
    prefer: Sequence[int],
) -> None:
    citation = find_citation(quote, blocks, prefer=prefer)
    if citation is not None:
        citations[key] = citation


def recover_threshold(
    trigger_type: str, blocks: Sequence[Block], prefer: Sequence[int]
) -> tuple[str, list[Decimal]] | None:
    """The first sentence in the model's context that names the trigger and prints a percentage.

    Returns the sentence verbatim (so it can be cited) and its percentages.
    Voting and quorum sentences ("noteholders of at least 5% may demand a
    vote") and historical statistics ("a historical peak of 4.96%") mention
    trigger names too, so they are skipped.
    """
    name = normalise(trigger_type).strip(" '\"")
    if len(name) < MIN_TRIGGER_NAME_CHARS:
        return None
    for index in prefer:
        if not 0 <= index < len(blocks):
            continue
        for sentence in _SENTENCE.split(blocks[index].text):
            if name not in normalise(sentence) or _NOT_A_THRESHOLD.search(sentence):
                continue
            values = parse_percents(sentence)
            if values:
                return sentence.strip(), values
    return None


def main_waterfall(waterfalls: Sequence[Waterfall]) -> Waterfall | None:
    """The deal's legal priority of payments among JOB-03's waterfalls.

    Summary versions and post-acceleration waterfalls are skipped; of the rest,
    the one with most steps wins, and on a tie the later one (summaries come
    first in a prospectus, the legal text later).
    """

    def eligible(w: Waterfall) -> bool:
        text = " ".join([*w.section_path, w.title])
        return not _SUMMARY.search(text) and "acceleration" not in text.lower()

    candidates = [w for w in waterfalls if eligible(w)] or list(waterfalls)
    if not candidates:
        return None
    return max(candidates, key=lambda w: (len(w.steps), w.start_page))


def assemble(document: ParsedDocument, extraction: Extraction) -> tuple[DealTerms, list[Issue]]:
    blocks = document.blocks
    issues: list[Issue] = []
    citations: dict[str, Citation] = {}
    fields: dict[str, object] = {}

    # --- deal facts: first context that states a field wins ------------------
    unparsed_dates: list[str] = []
    for facts, prefer in extraction.facts:
        if "deal_name" not in fields and facts.deal_name:
            fields["deal_name"] = facts.deal_name.strip()
            _cite(citations, "deal_name", facts.deal_name, blocks, prefer)
        if "closing_date" not in fields and facts.closing_date:
            closing = parse_date(facts.closing_date)
            if closing is None:
                unparsed_dates.append(facts.closing_date)
            else:
                fields["closing_date"] = closing
                _cite(citations, "closing_date", facts.closing_date, blocks, prefer)
        if "payment_frequency" not in fields:
            frequency = (
                PaymentFrequency(facts.payment_frequency)
                if facts.payment_frequency
                else parse_frequency(facts.payment_frequency_text)
            )
            if frequency is not None:
                fields["payment_frequency"] = frequency
                _cite(citations, "payment_frequency", facts.payment_frequency_text, blocks, prefer)
        if "cleanup_call_pct" not in fields:
            cleanup = parse_percent(facts.cleanup_call)
            if cleanup is not None:
                fields["cleanup_call_pct"] = cleanup
                _cite(citations, "cleanup_call_pct", facts.cleanup_call, blocks, prefer)
    # The asset class is a judgement, so take it from the context whose
    # evidence is actually in the document; otherwise from the first context.
    for facts, prefer in extraction.facts:
        evidence = find_citation(facts.asset_class_evidence, blocks, prefer=prefer)
        if evidence is not None:
            fields["asset_class"] = AssetClass(facts.asset_class)
            citations["asset_class"] = evidence
            break
    else:
        if extraction.facts:
            fields["asset_class"] = AssetClass(extraction.facts[0][0].asset_class)
    if "closing_date" not in fields:
        issues.extend(
            Issue(code="unparsed", field="closing_date", message=text) for text in unparsed_dates
        )

    # --- tranches -----------------------------------------------------------
    tranches: list[Tranche] = []
    raw = extraction.tranches
    prefer = extraction.tranches_blocks
    if raw is not None:
        for item in raw.tranches:
            name = clean_class_name(item.class_name)
            if _NON_PRINCIPAL.match(class_key(name)):
                continue
            balance = parse_money(item.original_balance)
            # "+ 0.30%" is a margin whose index was dropped; parsed alone it would
            # pass for a fixed 0.30% coupon, and it would even be citable.
            coupon = (
                None if _BARE_MARGIN.match(item.interest_rate) else parse_coupon(item.interest_rate)
            )
            if balance is None or coupon is None:
                issues.append(
                    Issue(
                        code="unparsed",
                        field=f"tranche {name}",
                        message=f"balance={item.original_balance!r} rate={item.interest_rate!r}",
                    )
                )
                continue
            tranche_citations: dict[str, Citation] = {}
            _cite(tranche_citations, "original_balance", item.original_balance, blocks, prefer)
            _cite(tranche_citations, "coupon", item.interest_rate, blocks, prefer)
            _cite(tranche_citations, "rating", item.rating, blocks, prefer)
            _cite(tranche_citations, "wal_years", item.weighted_average_life, blocks, prefer)
            tranches.append(
                Tranche(
                    class_name=name,
                    original_balance=balance,
                    coupon_type=coupon.coupon_type,
                    fixed_rate_pct=coupon.fixed_rate_pct,
                    index=coupon.index_name,
                    margin_pct=coupon.margin_pct,
                    rating=item.rating,
                    wal_years=parse_years(item.weighted_average_life),
                    citations=tranche_citations,
                )
            )
        total = parse_money(raw.stated_total)
        if total is not None:
            fields["stated_total_balance"] = total
            _cite(citations, "stated_total_balance", raw.stated_total, blocks, prefer)

    # --- triggers -------------------------------------------------------------
    triggers: list[TriggerEvent] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for raw_triggers, prefer in extraction.triggers:
        for t in raw_triggers.triggers:
            threshold_text = t.threshold
            thresholds = parse_percents(threshold_text)
            if threshold_text and not thresholds and _DIGIT.search(threshold_text):
                # "0.0080" instead of "0.80%": a unit slip must not become a
                # trigger that silently has no threshold.
                issues.append(
                    Issue(
                        code="unparsed", field=f"trigger {t.trigger_type}", message=threshold_text
                    )
                )
            elif not thresholds:
                # The model named the trigger but described the limit instead of
                # quoting it. Code finds the number: the LLM identifies, it never
                # supplies figures (CLAUDE.md 2.4).
                recovered = recover_threshold(t.trigger_type, blocks, prefer)
                if recovered is not None:
                    threshold_text, thresholds = recovered
                    issues.append(
                        Issue(
                            code="threshold_from_context",
                            field=f"trigger {t.trigger_type}",
                            message=threshold_text[:200],
                            severity=Severity.WARNING,
                        )
                    )
            key = (t.trigger_type.strip().lower(), tuple(str(v) for v in thresholds))
            if key in seen:  # the same trigger read from two overlapping contexts
                continue
            seen.add(key)
            trigger_citations: dict[str, Citation] = {}
            _cite(trigger_citations, "trigger_type", t.trigger_type, blocks, prefer)
            if thresholds:
                _cite(trigger_citations, "thresholds_pct", threshold_text, blocks, prefer)
            triggers.append(
                TriggerEvent(
                    trigger_type=t.trigger_type.strip(),
                    metric=t.metric.strip(),
                    thresholds_pct=thresholds,
                    threshold_text=threshold_text,
                    consequence=t.consequence.strip(),
                    citations=trigger_citations,
                )
            )

    # --- coverage tests ------------------------------------------------------
    tests: list[CoverageTest] = []
    for raw_tests, prefer in extraction.coverage:
        for test in raw_tests.tests:
            threshold = parse_percent(test.threshold)
            if threshold is None:
                issues.append(Issue(code="unparsed", field="coverage_test", message=test.threshold))
                continue
            test_citations: dict[str, Citation] = {}
            _cite(test_citations, "threshold_pct", test.threshold, blocks, prefer)
            tests.append(
                CoverageTest(
                    tranche=clean_class_name(test.tranche),
                    test_type=CoverageTestType(test.test_type),
                    threshold_pct=threshold,
                    citations=test_citations,
                )
            )

    # --- priority of payments: JOB-03, not the model ---------------------------
    steps: list[PaymentStep] = []
    waterfall = main_waterfall(document.waterfalls)
    if waterfall is not None:
        steps = [
            PaymentStep(ordinal=s.ordinal, text=s.text, page=s.page, page_label=s.page_label)
            for s in waterfall.steps
        ]
    else:
        issues.append(
            Issue(
                code="no_waterfall",
                field="priority_of_payments",
                message="JOB-03 found no priority of payments",
                severity=Severity.WARNING,
            )
        )

    terms = DealTerms(
        tranches=tranches,
        triggers=triggers,
        coverage_tests=tests,
        priority_of_payments=steps,
        citations=citations,
        **fields,
    )
    return terms, issues
