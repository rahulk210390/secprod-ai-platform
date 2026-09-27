"""Builders for hand-made block sequences used across the parsing tests."""

from __future__ import annotations

from dataclasses import dataclass

from secai.parsing.models import Block, BlockKind


@dataclass(frozen=True)
class B:
    """A block spec: kind, text, and optional page/level/marker."""

    kind: BlockKind
    text: str
    page: int = 1
    level: int | None = None
    marker: str | None = None
    label: str | None = None


def head(text: str, level: int = 1, page: int = 1) -> B:
    return B(BlockKind.HEADING, text, page=page, level=level)


def para(text: str, page: int = 1, label: str | None = None) -> B:
    return B(BlockKind.TEXT, text, page=page, label=label)


def item(text: str, marker: str | None = None, page: int = 1) -> B:
    return B(BlockKind.LIST_ITEM, text, page=page, marker=marker)


def blocks(*specs: B) -> list[Block]:
    return [
        Block(
            index=index,
            kind=spec.kind,
            text=spec.text,
            page=spec.page,
            page_label=spec.label,
            level=spec.level,
            marker=spec.marker,
        )
        for index, spec in enumerate(specs)
    ]
