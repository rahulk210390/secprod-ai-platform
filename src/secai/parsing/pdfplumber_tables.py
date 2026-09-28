"""pdfplumber as the table fallback.

Docling occasionally recognises a table region but returns an empty grid
(merged header cells, ruled layouts). The region's bounding box is still right,
so pdfplumber re-reads just that crop. Opening a PDF with pdfplumber is cheap;
reading every page is not, so this is only ever called per table.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pdfplumber
from pdfplumber.pdf import PDF


class TableFallback:
    """Lazily opened pdfplumber handle for one document."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pdf: PDF | None = None

    def __enter__(self) -> TableFallback:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def extract(self, page_no: int, bbox: tuple[float, float, float, float]) -> list[list[str]]:
        """The table inside ``bbox`` (left, top, right, bottom; top-left origin) on a 1-based page.

        Returns an empty list when pdfplumber finds nothing either.
        """
        if self._pdf is None:
            self._pdf = pdfplumber.open(self._path)
        page = self._pdf.pages[page_no - 1]
        left, top, right, bottom = bbox
        # Clamp: Docling's boxes can overhang the page by a fraction of a point.
        crop = page.crop(
            (max(left, 0), max(top, 0), min(right, page.width), min(bottom, page.height))
        )
        table = crop.extract_table()
        page.flush_cache()
        if not table:
            return []
        return [[(cell or "").strip() for cell in row] for row in table]
