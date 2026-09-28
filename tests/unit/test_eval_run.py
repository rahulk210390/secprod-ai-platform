"""The ``make eval JOB=04`` runner: report, thresholds and exit codes, with the pipeline faked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import term_fixtures as fx

from secai.config import reload_settings
from secai.eval import run as runner
from secai.jobs.term_extraction.assemble import assemble
from secai.jobs.term_extraction.evaluate import evaluate
from secai.jobs.term_extraction.pipeline import TermExtractionResult
from secai.schemas.deal_terms import DealTerms
from secai.validation.deal_terms import validate_deal_terms


def _result(terms: DealTerms, gold: dict[str, Any]) -> TermExtractionResult:
    return TermExtractionResult(
        terms=terms,
        report=validate_deal_terms(terms),
        evaluation=evaluate(terms, gold),
        trace_id="0" * 32,
        latency_s=1.5,
        tokens=1000,
        llm_calls=3,
        from_cache=False,
        prompt_version="set-test",
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gold_dir = tmp_path / "gold" / "term_extraction"
    gold_dir.mkdir(parents=True)
    samples = tmp_path / "samples"
    samples.mkdir()
    monkeypatch.setenv("SECAI_STORAGE_GOLD_DIR", str(tmp_path / "gold"))
    monkeypatch.setenv("SECAI_STORAGE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(runner, "SAMPLES", samples)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    reload_settings()
    return tmp_path


def _add_deal(workspace: Path, deal: str, *, reviewed: bool = True, fetched: bool = True) -> None:
    gold = fx.gold() | {"id": deal, "reviewed": reviewed}
    (workspace / "gold" / "term_extraction" / f"{deal}.json").write_text(json.dumps(gold))
    if fetched:
        (workspace / "samples" / f"{deal}.pdf").write_bytes(b"%PDF-1.4")


def _fake_pipeline(perfect: bool):  # type: ignore[no-untyped-def]
    def fake(pdf: Path, *, gold: dict[str, Any], **_: Any) -> TermExtractionResult:
        terms = assemble(fx.document(), fx.extraction())[0] if perfect else DealTerms()
        return _result(terms, gold)

    return fake


def test_all_thresholds_met_exits_zero_and_writes_report(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_deal(workspace, "deal_a")
    monkeypatch.setattr(runner, "run_term_extraction", _fake_pipeline(perfect=True))
    assert runner.run_job_04() == 0
    (report,) = (workspace / "reports").glob("04_*.md")
    text = report.read_text(encoding="utf-8")
    assert "| critical-field F1 | 1.000 | 0.95 | pass |" in text
    assert "| deal_a | 1.000 | 1.000 |" in text
    assert "`set-test`" in text


def test_missed_thresholds_exit_one(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_deal(workspace, "deal_a")
    monkeypatch.setattr(runner, "run_term_extraction", _fake_pipeline(perfect=False))
    assert runner.run_job_04() == 1
    text = next((workspace / "reports").glob("04_*.md")).read_text(encoding="utf-8")
    assert "**FAIL**" in text
    assert "missing_required" in text  # validation issues are listed


def test_drafts_unfetched_and_filtered_deals_are_skipped(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _add_deal(workspace, "draft", reviewed=False)
    _add_deal(workspace, "unfetched", fetched=False)
    _add_deal(workspace, "other")
    monkeypatch.setattr(runner, "run_term_extraction", _fake_pipeline(perfect=True))
    assert runner.run_job_04(only=["draft", "unfetched"]) == 2  # nothing left to evaluate
    out = capsys.readouterr()
    assert "unreviewed draft" in out.out
    assert "not fetched" in out.err


def test_main_dispatches_job_04_and_rejects_others(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(runner, "run_job_04", lambda **kw: calls.append(kw) or 0)
    assert runner.main(["--job", "04", "--fresh", "--only", "deal_a"]) == 0
    assert calls == [{"fresh": True, "only": ["deal_a"]}]
    assert runner.main(["--job", "06"]) == 2
