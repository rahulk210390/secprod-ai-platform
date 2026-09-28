"""Fetch the public EDGAR prospectuses listed in data/samples/manifest.json.

EDGAR serves prospectuses as HTML, but the platform's real inputs are PDFs, so
each filing is printed to PDF with a headless Chromium browser (Edge or Chrome).
EDGAR marks its original page breaks in CSS, which the print honours, so the
PDF pages stay close to the printed prospectus.

    uv run python scripts/fetch_samples.py            # fetch what is missing
    uv run python scripts/fetch_samples.py --pin      # record hashes in the manifest
    uv run python scripts/fetch_samples.py --only ford_auto_2025

The SEC requires automated tools to declare themselves: SEC_USER_AGENT in .env
must be "<name> <contact email>". Requests are spaced well under the SEC's
10-per-second limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from secai.config import REPO_ROOT, get_settings

MANIFEST = REPO_ROOT / "data" / "samples" / "manifest.json"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
REQUEST_SPACING_S = 0.5

_BROWSER_CANDIDATES = (
    Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
)


def find_browser() -> Path | None:
    """A Chromium-based browser that supports ``--print-to-pdf``."""
    override = os.environ.get("SECAI_BROWSER")
    if override:
        return Path(override)
    for candidate in _BROWSER_CANDIDATES:
        if candidate.is_file():
            return candidate
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(client: httpx.Client, url: str, target: Path) -> None:
    response = client.get(url)
    response.raise_for_status()
    target.write_bytes(response.content)


def print_to_pdf(browser: Path, html: Path, pdf: Path) -> None:
    """Render ``html`` to ``pdf``. A throwaway profile avoids clashing with a running browser."""
    with tempfile.TemporaryDirectory(prefix="secai-print-") as profile:
        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={pdf}",
                html.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError(f"browser produced no PDF for {html.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="sample ids to fetch (default: all)")
    parser.add_argument("--pin", action="store_true", help="write missing hashes to the manifest")
    parser.add_argument("--force", action="store_true", help="re-download and re-render")
    args = parser.parse_args()

    user_agent = get_settings().sec_user_agent.strip()
    if not user_agent:
        print("SEC_USER_AGENT is not set in .env (format: '<name> <email>')", file=sys.stderr)
        return 1
    browser = find_browser()
    if browser is None:
        print("no Edge/Chrome found; set SECAI_BROWSER to a Chromium binary", file=sys.stderr)
        return 1

    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = set(args.only) if args.only else None
    failures = 0
    pinned = False

    with httpx.Client(
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=120.0,
        follow_redirects=True,
    ) as client:
        for sample in manifest["samples"]:
            sample_id = sample["id"]
            if wanted is not None and sample_id not in wanted:
                continue
            html = SAMPLES_DIR / f"{sample_id}.htm"
            pdf = SAMPLES_DIR / f"{sample_id}.pdf"

            if args.force or not html.is_file():
                print(f"[{sample_id}] downloading {sample['url']}")
                download(client, sample["url"], html)
                time.sleep(REQUEST_SPACING_S)

            actual = sha256(html)
            expected = sample.get("sha256")
            if expected is None:
                print(f"[{sample_id}] sha256 {actual} (not pinned)")
                if args.pin:
                    sample["sha256"] = actual
                    pinned = True
            elif expected != actual:
                print(f"[{sample_id}] HASH MISMATCH: expected {expected}, got {actual}")
                failures += 1
                continue

            if args.force or not pdf.is_file():
                print(f"[{sample_id}] printing to PDF")
                started = time.perf_counter()
                print_to_pdf(browser, html, pdf)
                print(
                    f"[{sample_id}] {pdf.stat().st_size / 1e6:.1f} MB in "
                    f"{time.perf_counter() - started:.0f}s"
                )
            else:
                print(f"[{sample_id}] present")

    if pinned:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"pinned hashes written to {MANIFEST.relative_to(REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
