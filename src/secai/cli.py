"""Command-line entry point.

Subcommands are registered by later jobs; JOB-00 ships only ``config`` so the
installed package is verifiably wired up end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

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
        payload["sec_user_agent"] = "***" if settings.sec_user_agent else ""
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def parse(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF to parse")],
    out: Annotated[
        Path | None, typer.Option(help="write the full ParsedDocument JSON here")
    ] = None,
    deal_id: Annotated[str | None, typer.Option(help="recorded on the trace")] = None,
    asset_class: Annotated[str | None, typer.Option(help="recorded on the trace")] = None,
) -> None:
    """Parse a prospectus PDF and print a summary of its sections and waterfalls."""
    # Imported here: the parsing extra (Docling, pdfplumber) is optional.
    from secai.parsing import run_parsing_job
    from secai.telemetry import shutdown_telemetry

    try:
        document = run_parsing_job(path, deal_id=deal_id, asset_class=asset_class)
    finally:
        shutdown_telemetry()

    if out is not None:
        out.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    summary = {
        "source": document.source,
        "pages": document.page_count,
        "blocks": len(document.blocks),
        "tables": len(document.tables),
        "sections": len(document.sections),
        "canonical_sections": sorted({s.canonical for s in document.sections if s.canonical}),
        "waterfalls": [
            {
                "title": w.title[:100],
                "steps": len(w.steps),
                "pages": f"{w.start_page}-{w.end_page}",
                "form": w.form,
            }
            for w in document.waterfalls
        ],
        "duration_s": document.duration_s,
        "trace_id": document.trace_id,
    }
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
