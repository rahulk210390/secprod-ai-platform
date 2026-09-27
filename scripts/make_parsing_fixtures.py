"""Generate the synthetic prospectus PDF used by the parsing tests.

    uv run python scripts/make_parsing_fixtures.py

Everything in it is invented. It reproduces the layouts the real samples use:
all-caps and title-case headings, a numbered waterfall that breaks across a
page, an ordinal-prose waterfall ("First, ... Second, ..."), a ruled table,
and printed page labels ("S-1") that differ from the PDF page numbers.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

from secai.config import REPO_ROOT

OUT = REPO_ROOT / "tests" / "fixtures" / "parsing" / "synthetic_prospectus.pdf"

# Page 1 is an unnumbered cover, so printed labels start at S-1 on PDF page 2.
LABEL_OFFSET = 1
# The numbered waterfall continues onto the next page after this step.
BREAK_AFTER_STEP = 5

LIST_STEPS = [
    "to the indenture trustee and the owner trustee, all amounts due, up to a maximum of "
    "$250,000 per year,",
    "to the servicer, all unpaid servicing fees,",
    "to the Class A noteholders, interest due on the Class A notes, pro rata,",
    "to the Class A noteholders, principal in an amount equal to the first priority "
    "principal payment, if any,",
    "to the Class B noteholders, interest due on the Class B notes,",
    "to the reserve account, the amount required to replenish it to its target balance,",
    "to the noteholders, sequentially by class, principal in an amount equal to the "
    "regular principal payment,",
    "to the holder of the residual interest, all remaining available funds.",
]

PROSE_STEPS = [
    ("First", "to the Class A-1 and Class A-2 certificates, pro rata, interest distributable"),
    ("Second", "to the Class A-1 certificates, principal until reduced to zero"),
    ("Third", "to the Class A-2 certificates, principal until reduced to zero"),
    ("Fourth", "to the Class B certificates, interest distributable"),
    ("Fifth", "to the Class B certificates, principal until reduced to zero"),
]

FILLER = (
    "This paragraph is synthetic filler describing the transaction in general terms. It "
    "contains no deal data and exists only to give the page realistic density so that list "
    "items fall where a real prospectus would place them. "
)


def _footer(canvas: object, doc: object) -> None:
    page = doc.page  # type: ignore[attr-defined]
    if page <= LABEL_OFFSET:
        return
    canvas.saveState()  # type: ignore[attr-defined]
    canvas.setFont("Helvetica", 9)  # type: ignore[attr-defined]
    canvas.drawCentredString(LETTER[0] / 2, 30, f"S-{page - LABEL_OFFSET}")  # type: ignore[attr-defined]
    canvas.restoreState()  # type: ignore[attr-defined]


def build(path: Path = OUT) -> Path:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    item = ParagraphStyle("Item", parent=body, leftIndent=24, firstLineIndent=-24)

    story: list[Flowable] = [
        Paragraph("Synthetic Auto Receivables Trust 2099-1", styles["Title"]),
        Spacer(1, 24),
        Paragraph("Prospectus. All names and figures are fictitious.", body),
        PageBreak(),
        Paragraph("SUMMARY OF TERMS", h1),
        Paragraph(FILLER * 3, body),
        Paragraph("Notes", h2),
    ]

    table = Table(
        [
            ["Class", "Original Balance", "Interest Rate", "Final Payment Date"],
            ["A-1", "$250,000,000", "4.50%", "June 2100"],
            ["A-2", "$300,000,000", "4.75%", "June 2102"],
            ["B", "$50,000,000", "5.10%", "June 2104"],
        ]
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story += [table, Spacer(1, 12), Paragraph(FILLER * 2, body)]

    story += [
        PageBreak(),
        Paragraph("DESCRIPTION OF THE NOTES", h1),
        Paragraph(FILLER * 6, body),
        Paragraph("Priority of Payments", h2),
        Paragraph(
            "On each payment date, the servicer will direct the indenture trustee to use "
            "available funds to make payments in the order of priority listed below:",
            body,
        ),
    ]
    for n, text in enumerate(LIST_STEPS, start=1):
        story.append(Paragraph(f"({n}) {text}", item))
        if n == BREAK_AFTER_STEP:
            # Real waterfalls run across page breaks; so does this one.
            story.append(PageBreak())
    story += [
        Paragraph(
            "If available funds are insufficient to cover items (1) through (5), the "
            "shortfall will be withdrawn from the reserve account.",
            body,
        ),
        Paragraph("Events of Default", h2),
        Paragraph(FILLER * 2, body),
        PageBreak(),
        Paragraph("DISTRIBUTIONS ON THE CERTIFICATES", h1),
        Paragraph(
            "On each distribution date, available funds will be distributed in the following "
            "order of priority: "
            + "; ".join(f"{word}, {text}" for word, text in PROSE_STEPS)
            + ".",
            body,
        ),
        Paragraph(FILLER, body),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title="Synthetic prospectus",
        author="secai tests",
        # Fixed metadata keeps the file byte-stable across regenerations.
        invariant=1,
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


if __name__ == "__main__":
    print(build())
