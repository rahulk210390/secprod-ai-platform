"""Global test fixtures.

Point the settings loader at a non-existent .env *before* secai.config is
imported, so unit tests see a clean environment rather than the developer's
local secrets.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("SECAI_ENV_FILE", str(Path(__file__).parent / "fixtures" / "absent.env"))

import pytest

from secai.config import Settings, get_settings, reload_settings

# Every env var the settings tree can read; cleared between tests.
_MANAGED_PREFIXES = ("SECAI_", "LANGFUSE_", "OTEL_", "VLLM_")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove secai-related env vars and reset the settings cache.

    The cache is only *cleared*, never repopulated, so a test that sets env
    vars still gets them picked up by the first ``get_settings()`` call.
    """
    for key in list(os.environ):
        if key.startswith(_MANAGED_PREFIXES) and key != "SECAI_ENV_FILE":
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Freshly loaded settings for the current environment."""
    return reload_settings()
