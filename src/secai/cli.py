"""Command-line entry point.

Subcommands are registered by later jobs; JOB-00 ships only ``config`` so the
installed package is verifiably wired up end to end.
"""

from __future__ import annotations

import json

import typer

from secai import __version__
from secai.config import get_settings

app = typer.Typer(help="secai — securitised products AI platform", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def config(show_secrets: bool = False) -> None:
    """Print the resolved configuration as JSON (secrets masked by default)."""
    settings = get_settings()
    payload = settings.model_dump(mode="json")
    if not show_secrets:
        payload["langfuse"]["secret_key"] = "***" if settings.langfuse.secret_key else ""
        payload["langfuse"]["basic_auth"] = "***" if settings.langfuse.secret_key else ""
        payload["vllm"]["api_key"] = "***"
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    app()
