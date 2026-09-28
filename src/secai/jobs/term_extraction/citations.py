"""Find where a quoted value appears in the document, and cite that page.

Citations are not taken from the LLM. The model returns values verbatim, and
this module searches the parsed document for that text: a value that is not in
the source cannot be cited, so validation can flag it as ungrounded. That turns
"100% of fields carry a page citation" from a prompt instruction into a
property of the code.

Matching tolerates what differs between a PDF's text and a model's copy of it:
spacing ("$ 320,400,000" vs "$320,400,000"), dash and quote styles, and case.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

from secai.parsing.models import Block
from secai.schemas.deal_terms import Citation

_TRANSLATE: Final = str.maketrans(
    {
        chr(0x2019): "'",
        chr(0x2018): "'",
        chr(0x201C): '"',
        chr(0x201D): '"',
        chr(0x2013): "-",
        chr(0x2014): "-",
        chr(0x00A0): " ",
    }
)
_SPACES = re.compile(r"\s+")
_AFTER_CURRENCY = re.compile(r"([$€£])\s+")
_BEFORE_PERCENT = re.compile(r"\s+%")
# Shorter quotes ("A", "5%") match too much of a 200-page document to cite.
MIN_QUOTE_CHARS: Final = 2


def normalise(text: str) -> str:
    text = text.translate(_TRANSLATE)
    text = _AFTER_CURRENCY.sub(r"\1", text)
    text = _BEFORE_PERCENT.sub("%", text)
    return _SPACES.sub(" ", text).strip().lower()


def find_citation(
    quote: str | None,
    blocks: Sequence[Block],
    *,
    prefer: Iterable[int] = (),
) -> Citation | None:
    """Cite the first block containing ``quote``, looking in ``prefer`` first.

    ``prefer`` is the block indices that were sent to the model, so a value
    that appears both in the extraction context and elsewhere is cited where
    the model actually read it.
    """
    if not quote:
        return None
    needle = normalise(quote)
    if len(needle) < MIN_QUOTE_CHARS:
        return None

    preferred = [blocks[i] for i in prefer if 0 <= i < len(blocks)]
    for block in (*preferred, *blocks):
        if needle in normalise(block.text):
            return Citation(quote=quote.strip(), page=block.page, page_label=block.page_label)
    return None
