"""Configuration tests (JOB-00)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from secai.config import (
    DatabaseSettings,
    Environment,
    LangfuseSettings,
    OTelSettings,
    PIISettings,
    Settings,
    StorageSettings,
    VLLMSettings,
    get_settings,
    reload_settings,
)


class TestDefaults:
    def test_root_defaults(self, settings: Settings) -> None:
        assert settings.environment is Environment.DEV
        assert settings.log_level == "INFO"
        # Guardrail: extraction must stay schema-constrained by default.
        assert settings.guided_decoding_required is True

    def test_nested_models_are_populated(self, settings: Settings) -> None:
        assert isinstance(settings.langfuse, LangfuseSettings)
        assert isinstance(settings.otel, OTelSettings)
        assert isinstance(settings.vllm, VLLMSettings)
        assert isinstance(settings.db, DatabaseSettings)
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.pii, PIISettings)

    def test_no_secret_has_a_real_default(self, settings: Settings) -> None:
        assert settings.langfuse.public_key == ""
        assert settings.langfuse.secret_key == ""
        assert settings.langfuse.auth == ""
        # vLLM's key is a required-but-ignored placeholder, not a credential.
        assert settings.vllm.api_key == "EMPTY"

    def test_storage_paths_point_into_the_repo(self, settings: Settings) -> None:
        assert isinstance(settings.storage.documents_dir, Path)
        assert settings.storage.documents_dir.parts[-2:] == ("data", "samples")
        assert settings.storage.gold_dir.parts[-2:] == ("data", "gold")
        assert settings.storage.reports_dir.parts[-2:] == ("eval", "reports")

    def test_storage_paths_can_be_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECAI_STORAGE_GOLD_DIR", "/srv/gold")
        assert reload_settings().storage.gold_dir == Path("/srv/gold")


class TestEnvOverrides:
    def test_root_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECAI_ENVIRONMENT", "prod")
        monkeypatch.setenv("SECAI_LOG_LEVEL", "DEBUG")
        s = reload_settings()
        assert s.environment is Environment.PROD
        assert s.log_level == "DEBUG"

    def test_nested_prefixes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "secai-worker")
        monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
        monkeypatch.setenv("SECAI_DB_ECHO", "true")
        s = reload_settings()
        assert s.langfuse.host == "https://cloud.langfuse.com"
        assert s.otel.service_name == "secai-worker"
        assert s.vllm.model == "Qwen/Qwen2.5-32B-Instruct-AWQ"
        assert s.db.echo is True

    def test_invalid_environment_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECAI_ENVIRONMENT", "banana")
        with pytest.raises(ValueError, match="environment"):
            reload_settings()


class TestLangfuse:
    def test_otel_endpoint_is_http_and_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse-web:3000/")
        lf = reload_settings().langfuse
        # Trailing slash must not produce a double slash; Langfuse rejects it.
        assert lf.otel_endpoint == "http://langfuse-web:3000/api/public/otel"

    def test_basic_auth_is_derived_from_the_key_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-xyz")
        lf = reload_settings().langfuse
        assert base64.b64decode(lf.basic_auth).decode() == "pk-lf-abc:sk-lf-xyz"

    def test_explicit_auth_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-xyz")
        monkeypatch.setenv("LANGFUSE_AUTH", "preset-value")
        assert reload_settings().langfuse.basic_auth == "preset-value"

    def test_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert reload_settings().langfuse.is_configured is False
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-xyz")
        assert reload_settings().langfuse.is_configured is True


class TestVLLM:
    def test_router_model_falls_back_to_the_main_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_MODEL", "big-model")
        assert reload_settings().vllm.effective_router_model == "big-model"

    def test_router_model_is_used_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_MODEL", "big-model")
        monkeypatch.setenv("VLLM_ROUTER_MODEL", "small-model")
        assert reload_settings().vllm.effective_router_model == "small-model"

    def test_deterministic_by_default(self, settings: Settings) -> None:
        # Extraction must be reproducible; sampling is opt-in, never default.
        assert settings.vllm.temperature == 0.0


class TestPII:
    def test_borrower_fields_are_listed(self, settings: Settings) -> None:
        for field in ("borrower_id", "borrower_name", "account_number", "loan_number"):
            assert field in settings.pii.fields

    def test_field_list_is_immutable(self, settings: Settings) -> None:
        assert isinstance(settings.pii.fields, tuple)


class TestCaching:
    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_reload_returns_a_new_instance(self) -> None:
        first = get_settings()
        assert reload_settings() is not first
