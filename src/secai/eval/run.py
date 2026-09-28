"""``make eval JOB=<id>``: run a job's gold set and write a report.

JOB-04 ships the minimal version (CLAUDE.md: "build a minimal eval inside
JOB-04, then generalise it here" in JOB-05). Only job 04 is wired up.

    uv run python -m secai.eval.run --job 04 [--fresh] [--only ford_auto_2025]

Scores are micro-averaged: true/false positives and negatives are summed
across deals before F1 is computed, so a deal with 12 tranches weighs more
than a deal with one. Exit status is 0 only when every threshold is met.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Final

from secai.config import REPO_ROOT, get_settings
from secai.jobs.term_extraction.evaluate import CRITICAL, EvalResult
from secai.jobs.term_extraction.pipeline import TermExtractionResult, run_term_extraction
from secai.telemetry import shutdown_telemetry

CRITICAL_F1_THRESHOLD: Final = 0.95
OVERALL_F1_THRESHOLD: Final = 0.90
CITATION_THRESHOLD: Final = 1.0

SAMPLES: Final = REPO_ROOT / "data" / "samples"
PARSE_CACHE: Final = SAMPLES / ".cache"
EXTRACTION_CACHE: Final = PARSE_CACHE / "extraction"


def _report(
    run_id: str,
    results: dict[str, TermExtractionResult],
    total: EvalResult,
    model: str,
    prompt_version: str,
) -> str:
    citations = [r.citation_rate for r in results.values()]
    citation_rate = min(citations) if citations else 1.0
    checks = [
        ("critical-field F1", total.critical_field_f1, CRITICAL_F1_THRESHOLD),
        ("overall field F1", total.field_f1, OVERALL_F1_THRESHOLD),
        ("fields with a citation (worst deal)", citation_rate, CITATION_THRESHOLD),
    ]
    lines = [
        f"# JOB-04 evaluation: {run_id}",
        "",
        f"- model: `{model}`",
        f"- prompt set: `{prompt_version}`",
        f"- deals: {len(results)}",
        "",
        "## Thresholds",
        "",
        "| Metric | Result | Threshold | |",
        "|---|---|---|---|",
    ]
    for name, value, threshold in checks:
        verdict = "pass" if value >= threshold else "**FAIL**"
        lines.append(f"| {name} | {value:.3f} | {threshold:.2f} | {verdict} |")
    lines += [
        "",
        "## Per deal",
        "",
        "| Deal | Critical F1 | Field F1 | Validation | Cited | LLM calls | Tokens | Seconds "
        "| Trace |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for deal, r in results.items():
        e = r.evaluation
        assert e is not None
        status = "pass" if r.report.passed else "needs_review"
        calls = f"{r.llm_calls}{' (cached)' if r.from_cache else ''}"
        lines.append(
            f"| {deal} | {e.critical_field_f1:.3f} | {e.field_f1:.3f} | {status} | "
            f"{r.citation_rate:.0%} | {calls} | {r.tokens} | {r.latency_s:.0f} | `{r.trace_id}` |"
        )
    lines += [
        "",
        "## Per field (micro-averaged)",
        "",
        "| Field | Critical | TP | FP | FN | P | R | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(total.per_field):
        c = total.per_field[name]
        lines.append(
            f"| {name} | {'yes' if name in CRITICAL else ''} | {c.tp} | {c.fp} | {c.fn} | "
            f"{c.precision:.2f} | {c.recall:.2f} | {c.f1:.2f} |"
        )
    lines += ["", "## Validation issues", ""]
    for deal, r in results.items():
        for issue in r.report.issues:
            lines.append(
                f"- {deal}: `{issue.code}` {issue.field} ({issue.severity.value}): {issue.message}"
            )
    lines += ["", "## Mismatches against gold", ""]
    lines += [f"- {m}" for m in total.mismatches] or ["- none"]
    return "\n".join(lines) + "\n"


def run_job_04(*, fresh: bool = False, only: list[str] | None = None) -> int:
    settings = get_settings()
    gold_dir = settings.storage.gold_dir / "term_extraction"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results: dict[str, TermExtractionResult] = {}
    total = EvalResult()
    prompt_version = ""

    for gold_path in sorted(gold_dir.glob("*.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        deal = gold["id"]
        if only and deal not in only:
            continue
        if not gold.get("reviewed"):
            print(f"[{deal}] gold is an unreviewed draft; skipped")
            continue
        pdf = SAMPLES / f"{deal}.pdf"
        if not pdf.is_file():
            print(f"[{deal}] {pdf.name} not fetched; run scripts/fetch_samples.py", file=sys.stderr)
            continue
        result = run_term_extraction(
            pdf,
            deal_id=deal,
            gold=gold,
            settings=settings,
            parse_cache_dir=PARSE_CACHE,
            extraction_cache_dir=None if fresh else EXTRACTION_CACHE,
            session_id=f"eval-04-{run_id}",
        )
        assert result.evaluation is not None
        results[deal] = result
        total.merge(result.evaluation)
        print(
            f"[{deal}] critical F1 {result.evaluation.critical_field_f1:.3f}, "
            f"field F1 {result.evaluation.field_f1:.3f}, "
            f"{'pass' if result.report.passed else 'needs_review'}, "
            f"{result.llm_calls} calls, {result.latency_s:.0f}s"
        )
        prompt_version = prompt_version or result.prompt_version

    if not results:
        print("no deals evaluated", file=sys.stderr)
        return 2

    report = _report(run_id, results, total, settings.vllm.model, prompt_version)
    settings.storage.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.storage.reports_dir / f"04_{run_id}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\ncritical F1 {total.critical_field_f1:.3f}  overall F1 {total.field_f1:.3f}")
    print(f"report: {path.relative_to(REPO_ROOT)}")
    met = (
        total.critical_field_f1 >= CRITICAL_F1_THRESHOLD
        and total.field_f1 >= OVERALL_F1_THRESHOLD
        and all(r.citation_rate >= CITATION_THRESHOLD for r in results.values())
    )
    return 0 if met else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a job's gold-set evaluation")
    parser.add_argument("--job", required=True, help="job id, e.g. 04")
    parser.add_argument("--fresh", action="store_true", help="ignore the extraction cache")
    parser.add_argument("--only", nargs="*", help="deal ids to evaluate")
    args = parser.parse_args(argv)
    try:
        if args.job.lstrip("0") == "4":
            return run_job_04(fresh=args.fresh, only=args.only)
        print(
            f"no evaluation for job {args.job} yet (JOB-05 generalises the harness)",
            file=sys.stderr,
        )
        return 2
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    raise SystemExit(main())
