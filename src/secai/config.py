"""Central configuration.

All runtime configuration is read from environment variables (or a git-ignored
``.env`` file). Nothing is hard-coded, and no secret ever has a real default.

Settings are grouped into nested models that mirror the ``.env.example``
sections; :func:`get_settings` returns a cached, fully-populated tree.
"""

from __future__ import annotations

import base64
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
# Overridable so tests (and CI) can run against a known-empty environment
# instead of whatever a developer happens to have in their local .env.
ENV_FILE = Path(os.environ.get("SECAI_ENV_FILE", REPO_ROOT / ".env"))

_BASE_CONFIG = SettingsConfigDict(
    env_file=ENV_FILE,
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class Environment(StrEnum):
    """Deployment environment."""

    DEV = "dev"
    CI = "ci"
    STAGING = "staging"
    PROD = "prod"


class LangfuseSettings(BaseSettings):
    """Langfuse credentials and endpoint (SDK v4, OTel-native)."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LANGFUSE_")

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"
    # base64("public:secret") — consumed by the OTel Collector, not the SDK.
    auth: str = ""
    # Turn exporting off entirely (unit tests, offline CI).
    enabled: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def otel_endpoint(self) -> str:
        """The Langfuse OTLP receiver. HTTP only — it does not accept gRPC."""
        return f"{self.host.rstrip('/')}/api/public/otel"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def basic_auth(self) -> str:
        """``auth`` if set, otherwise derived from the key pair."""
        if self.auth:
            return self.auth
        raw = f"{self.public_key}:{self.secret_key}".encode()
        return base64.b64encode(raw).decode()

    @property
    def is_configured(self) -> bool:
        return bool(self.public_key and self.secret_key)


class OTelSettings(BaseSettings):
    """OpenTelemetry export settings (app → Collector)."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="OTEL_")

    service_name: str = "secai-app"
    exporter_otlp_endpoint: str = "http://localhost:4318"
    exporter_otlp_protocol: str = "http/protobuf"
    resource_attributes: str = "deployment.environment=dev"


class VLLMSettings(BaseSettings):
    """vLLM OpenAI-compatible server."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="VLLM_")

    base_url: str = "http://localhost:8000/v1"
    # Placeholder key: vLLM requires the header, not a real credential.
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    # Small model used by the router/classifier path.
    router_model: str = ""
    timeout_s: float = 120.0
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 4096

    @property
    def effective_router_model(self) -> str:
        return self.router_model or self.model


class DatabaseSettings(BaseSettings):
    """PostgreSQL for application state."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SECAI_DB_")

    url: str = "postgresql+psycopg://secai:secai@localhost:5433/secai"
    echo: bool = False
    pool_size: int = 5


class StorageSettings(BaseSettings):
    """Where source documents and derived artefacts live."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SECAI_STORAGE_")

    documents_dir: Path = REPO_ROOT / "data" / "samples"
    gold_dir: Path = REPO_ROOT / "data" / "gold"
    reports_dir: Path = REPO_ROOT / "eval" / "reports"


class PIISettings(BaseSettings):
    """Fields and patterns scrubbed before anything is exported (JOB-01)."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SECAI_PII_")

    redaction_token: str = "[REDACTED]"
    fields: tuple[str, ...] = (
        "borrower_id",
        "borrower_name",
        "borrower_address",
        "account_number",
        "loan_number",
        "ssn",
        "national_id",
        "email",
        "phone",
    )


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SECAI_")

    environment: Environment = Environment.DEV
    log_level: str = "INFO"
    # Guardrail: the LLM never computes financial numbers, so extraction is
    # always schema-constrained. Flipping this off is a deliberate test-only act.
    guided_decoding_required: bool = True

    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    otel: OTelSettings = Field(default_factory=OTelSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    pii: PIISettings = Field(default_factory=PIISettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment. Tests use this."""
    get_settings.cache_clear()
    return get_settings()
