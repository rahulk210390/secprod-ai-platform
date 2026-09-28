"""One deal's term extraction as a single traced job run (JOB-04).

    job.term_extraction
    ├── parse / chunk        JOB-03 (Docling output cached by file hash)
    ├── retrieve             choose the excerpts each question reads
    ├── llm.extract (xN)     one Langfuse generation per excerpt
    ├── validate             assemble, cite and check deterministically
    └── persist              write the DealTerms JSON

Scores on the trace: ``validation_pass``, ``needs_review``, ``latency_s``,
``cost_tokens``, and, when gold is supplied, ``field_f1`` and
``critical_field_f1``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from secai.config import Settings, get_settings
from secai.jobs.term_extraction.assemble import assemble
from secai.jobs.term_extraction.context import Context, Target, budget_chars, build_contexts
from secai.jobs.term_extraction.evaluate import EvalResult, evaluate
from secai.jobs.term_extraction.extract import (
    TermExtractor,
    cache_key,
    load_extraction,
    save_extraction,
)
from secai.llm import LLMClient
from secai.parsing.models import ParsedDocument
from secai.parsing.pipeline import parse_document
from secai.schemas.deal_terms import DealTerms
from secai.telemetry import (
    SPAN_PERSIST,
    SPAN_RETRIEVE,
    SPAN_VALIDATE,
    current_trace_id,
    job_span,
    job_trace,
    score_trace,
    set_span_attributes,
)
from secai.telemetry.attributes import (
    ATTR_ASSET_CLASS,
    SCORE_COST_TOKENS,
    SCORE_CRITICAL_FIELD_F1,
    SCORE_FIELD_F1,
    SCORE_LATENCY_S,
    SCORE_NEEDS_REVIEW,
    SCORE_VALIDATION_PASS,
)
from secai.validation.deal_terms import ValidationReport, citation_coverage, validate_deal_terms

JOB_NAME: Final = "term_extraction"


@dataclass
class TermExtractionResult:
    terms: DealTerms
    report: ValidationReport
    evaluation: EvalResult | None
    trace_id: str | None
    latency_s: float
    tokens: int
    llm_calls: int
    from_cache: bool
    prompt_version: str = ""
    contexts: dict[str, int] = field(default_factory=dict)

    @property
    def citation_rate(self) -> float:
        cited, populated = citation_coverage(self.terms)
        return cited / populated if populated else 1.0


def run_term_extraction(
    pdf: Path,
    *,
    deal_id: str | None = None,
    gold: dict[str, Any] | None = None,
    client: LLMClient | None = None,
    settings: Settings | None = None,
    parse_cache_dir: Path | None = None,
    extraction_cache_dir: Path | None = None,
    output_dir: Path | None = None,
    session_id: str | None = None,
    document: ParsedDocument | None = None,
) -> TermExtractionResult:
    """Extract, validate and (with ``gold``) score one deal, as one trace.

    Pass ``document`` when the PDF is already parsed; otherwise it is parsed
    here (from ``parse_cache_dir`` when the file is unchanged).
    """
    resolved = settings or get_settings()
    llm = client or LLMClient(settings=resolved)
    extractor = TermExtractor.create(llm, resolved)
    started = time.perf_counter()
    calls_before, tokens_before = llm.calls, llm.usage_total.get("total", 0)

    with job_trace(
        JOB_NAME,
        deal_id=deal_id or pdf.stem,
        doc_type="prospectus",
        prompt_version=extractor.prompt_version,
        model=extractor.model,
        session_id=session_id,
        tags=["job-04"],
        input_data={"source": pdf.name},
    ) as root:
        if document is None:
            document = parse_document(pdf, settings=resolved, cache_dir=parse_cache_dir)

        with job_span(SPAN_RETRIEVE) as span:
            # Each target's prompt differs in length, so each gets its own budget.
            contexts: dict[Target, list[Context]] = {
                target: build_contexts(
                    document,
                    target,
                    budget_chars(
                        resolved.vllm.max_model_len,
                        resolved.vllm.max_tokens,
                        extractor.prompt_chars(target),
                    ),
                )
                for target in Target
            }
            counts = {target.value: len(items) for target, items in contexts.items()}
            set_span_attributes(
                span,
                {f"secai.retrieve.{name}_contexts": count for name, count in counts.items()},
            )
            span.update(output=counts)

        cached_path = None
        if extraction_cache_dir is not None:
            key = cache_key(document.sha256, extractor.prompt_version, extractor.model)
            cached_path = extraction_cache_dir / f"{pdf.stem}.{key}.json"
        from_cache = cached_path is not None and cached_path.is_file()
        if from_cache and cached_path is not None:
            extraction, llm_issues = load_extraction(cached_path)
        else:
            extraction, llm_issues = extractor.run(contexts, document.source)
            if cached_path is not None:
                save_extraction(cached_path, extraction, llm_issues)

        with job_span(SPAN_VALIDATE) as span:
            terms, assembly_issues = assemble(document, extraction)
            report = validate_deal_terms(terms, extra=[*llm_issues, *assembly_issues])
            cited, populated = citation_coverage(terms)
            set_span_attributes(
                span,
                {
                    "secai.validate.passed": report.passed,
                    "secai.validate.errors": len(report.errors),
                    "secai.validate.cited_fields": cited,
                    "secai.validate.populated_fields": populated,
                },
            )
            span.update(
                output={
                    "passed": report.passed,
                    "issues": [f"{i.code}: {i.field}" for i in report.issues][:40],
                }
            )
        if terms.asset_class is not None:
            set_span_attributes(root, {ATTR_ASSET_CLASS: str(terms.asset_class)})

        evaluation = evaluate(terms, gold) if gold is not None else None

        with job_span(SPAN_PERSIST) as span:
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                target = output_dir / f"{pdf.stem}.deal_terms.json"
                target.write_text(terms.model_dump_json(indent=2), encoding="utf-8")
                span.update(output={"written": target.name})

        latency = time.perf_counter() - started
        tokens = llm.usage_total.get("total", 0) - tokens_before
        score_trace(SCORE_VALIDATION_PASS, report.passed)
        score_trace(SCORE_NEEDS_REVIEW, report.needs_review)
        score_trace(SCORE_LATENCY_S, round(latency, 3))
        score_trace(SCORE_COST_TOKENS, float(tokens))
        if evaluation is not None:
            score_trace(SCORE_FIELD_F1, round(evaluation.field_f1, 4))
            score_trace(SCORE_CRITICAL_FIELD_F1, round(evaluation.critical_field_f1, 4))

        root.update(
            output={
                "tranches": len(terms.tranches),
                "triggers": len(terms.triggers),
                "validation_pass": report.passed,
                "citations": f"{cited}/{populated}",
            }
        )
        trace_id = current_trace_id()

    return TermExtractionResult(
        terms=terms,
        report=report,
        evaluation=evaluation,
        trace_id=trace_id,
        latency_s=latency,
        tokens=tokens,
        llm_calls=llm.calls - calls_before,
        from_cache=from_cache,
        prompt_version=extractor.prompt_version,
        contexts=counts,
    )
