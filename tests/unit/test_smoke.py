"""Foundation smoke tests (JOB-00): the package imports and is wired up."""

from __future__ import annotations

import importlib

import pytest

import secai

SUBPACKAGES = [
    "secai.telemetry",
    "secai.llm",
    "secai.parsing",
    "secai.schemas",
    "secai.validation",
    "secai.eval",
    "secai.api",
    "secai.jobs",
    "secai.jobs.term_extraction",
    "secai.jobs.servicer_reports",
    "secai.jobs.loan_tape_mapping",
    "secai.jobs.trigger_surveillance",
    "secai.jobs.waterfall_codegen",
    "secai.jobs.commentary",
    "secai.jobs.deal_qa",
]


def test_version_is_exposed() -> None:
    assert secai.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(name) is not None
