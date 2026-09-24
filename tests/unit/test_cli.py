"""CLI tests (JOB-00)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from secai import __version__
from secai.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_command_emits_json() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["environment"] == "dev"
    assert "langfuse" in payload and "vllm" in payload


def test_config_masks_secrets_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-supersecret")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "sk-lf-supersecret" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["langfuse"]["secret_key"] == "***"
    assert payload["langfuse"]["public_key"] == "pk-lf-abc"


def test_config_show_secrets_reveals_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-supersecret")
    result = runner.invoke(app, ["config", "--show-secrets"])
    assert result.exit_code == 0
    assert "sk-lf-supersecret" in result.stdout
