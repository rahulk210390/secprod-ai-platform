"""Prompt templates, versioned in Langfuse with a local file fallback.

CLAUDE.md §5 (JOB-02) asks for Langfuse Prompt Management so prompt changes are
versioned and eval runs can be compared across them. Langfuse must not become a
hard dependency of a job run, though: if it is unreachable, extraction should
still work from the prompt committed alongside the code.

Resolution order:

1. Langfuse Prompt Management, by name and optional label;
2. ``prompts/<name>.md`` in the repo.

Whichever wins, the resolved version is stamped onto the span as
``secai.prompt_version``, so a trace always says which prompt produced it — the
eval harness in JOB-05 groups on exactly that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from secai.config import REPO_ROOT
from secai.telemetry.setup import get_client

logger = logging.getLogger(__name__)

PROMPT_DIR = REPO_ROOT / "prompts"

LOCAL_VERSION_PREFIX = "local"


@dataclass(frozen=True)
class Prompt:
    """A resolved prompt template and where it came from."""

    name: str
    template: str
    version: str
    source: str  # "langfuse" or "local"

    def render(self, **values: object) -> str:
        """Fill ``{placeholders}`` in the template.

        Raises:
            KeyError: a placeholder had no value. Failing loudly beats sending
                a half-rendered prompt to the model.
        """
        try:
            return self.template.format(**values)
        except KeyError as exc:
            raise KeyError(
                f"prompt {self.name!r} ({self.version}) needs a value for {exc}"
            ) from exc


def local_prompt_path(name: str) -> Path:
    return PROMPT_DIR / f"{name}.md"


def load_local(name: str) -> Prompt:
    """Read the prompt committed alongside the code.

    Raises:
        FileNotFoundError: no local fallback exists for this name.
    """
    path = local_prompt_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no local fallback for prompt {name!r}; expected {path}")
    template = path.read_text(encoding="utf-8").strip()
    return Prompt(
        name=name,
        template=template,
        # Content-addressed, so a local edit is visible in the trace.
        version=f"{LOCAL_VERSION_PREFIX}-{_short_hash(template)}",
        source="local",
    )


def get_prompt(name: str, *, label: str | None = None, use_langfuse: bool = True) -> Prompt:
    """Resolve a prompt, preferring Langfuse and falling back to the repo."""
    if use_langfuse:
        try:
            fetched = (
                get_client().get_prompt(name, label=label)
                if label
                else get_client().get_prompt(name)
            )
            template = (
                fetched.get_langchain_prompt() if hasattr(fetched, "get_langchain_prompt") else None
            )
            return Prompt(
                name=name,
                template=template if isinstance(template, str) else str(fetched.prompt),
                version=str(getattr(fetched, "version", "unknown")),
                source="langfuse",
            )
        except Exception:
            logger.warning(
                "could not fetch prompt %r from Langfuse; using the local copy",
                name,
                exc_info=True,
            )

    return load_local(name)


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
