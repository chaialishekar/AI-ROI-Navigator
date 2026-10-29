"""Executive decision brief: the stage 4 business case as a two-page PDF.

The app gathers the numbers (value, ranges, readiness, alternatives, decision rule) into one dict;
this module only lays them out. Built with reportlab's standard fonts, so text is kept to the
Windows-1252 character set (no arrows or maths symbols).
"""

from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

INK = colors.HexColor("#1d2521")
TEAL = colors.HexColor("#0f6d74")
LIME = colors.HexColor("#d4ee5e")
SAGE = colors.HexColor("#eef2ea")
MUTED = colors.HexColor("#56615a")
RULE = colors.HexColor("#d5dccf")

PLAIN = {"→": "to", "≥": ">=", "≤": "<=", "“": '"', "”": '"', "’": "'", "−": "-"}


def clean(text) -> str:
    text = str(text)
    for a, b in PLAIN.items():
        text = text.replace(a, b)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("&lt;b&gt;", "<b>") \
        .replace("&lt;/b&gt;", "</b>")


S = {
    "eyebrow": ParagraphStyle("eyebrow", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=LIME),
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.white),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#c8d2c3")),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=TEAL, spaceBefore=10,
                        spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=13, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.5, leading=10, textColor=MUTED),
    "rec": ParagraphStyle("rec", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=INK),
    "kpi_l": ParagraphStyle("kpi_l", fontName="Helvetica-Bold", fontSize=6.8, leading=9, textColor=MUTED),
    "kpi_v": ParagraphStyle("kpi_v", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=INK),
    "kpi_n": ParagraphStyle("kpi_n", fontName="Helvetica", fontSize=7, leading=9, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK),
    "cell_b": ParagraphStyle("cell_b", fontName="Helvetica-Bold", fontSize=8, leading=10.5, textColor=INK),
    "head": ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED),
}
WIDTH = letter[0] - 1.3 * inch


def P(text, style="body") -> Paragraph:
    return Paragraph(clean(text), S[style])


def table(header: list[str], rows: list[list], widths: list[float], bold_rows=(), highlight_rows=()) -> Table:
    data = [[P(h.upper(), "head") for h in header]]
    for i, row in enumerate(rows):
        data.append([P(c, "cell_b" if i in bold_rows else "cell") for c in row])
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1)
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    for i in highlight_rows:
        style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#f2f8d8")))
    t.setStyle(TableStyle(style))
    return t


def header_band(d: dict) -> Table:
    inner = [P("INVESTMENT DECISION BRIEF  |  ILLUSTRATIVE CASE", "eyebrow"), Spacer(1, 4),
             P(d["title"], "title"), Spacer(1, 4),
             P(f"{d['client']}  |  Prepared by {d['author']}  |  {date.today():%B %Y}", "sub")]
    t = Table([[inner]], colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INK), ("LEFTPADDING", (0, 0), (-1, -1), 16),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 16), ("TOPPADDING", (0, 0), (-1, -1), 14),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    return t


def recommendation(d: dict) -> Table:
    t = Table([[ "", [P(f"Recommendation: {d['decision']}", "rec"), Spacer(1, 3), P(d['decision_reason'])]]],
              colWidths=[5, WIDTH - 5])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), LIME), ("BACKGROUND", (1, 0), (1, 0), SAGE),
                           ("LEFTPADDING", (1, 0), (1, 0), 12), ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    return t


def kpi_row(items: list[tuple[str, str, str]]) -> Table:
    cells = [[P(label.upper(), "kpi_l"), P(value, "kpi_v"), P(note, "kpi_n")] for label, value, note in items]
    t = Table([cells], colWidths=[WIDTH / len(items)] * len(items))
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 2, TEAL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.45 * inch,
                      "Illustrative case. Model results measured on public, synthetic machine data (AI4I 2020, UCI); plant costs "
                      "are assumptions; results after launch are simulated.")
    canvas.drawRightString(letter[0] - 0.65 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_brief(d: dict) -> bytes:
    """d holds pre-formatted strings and row lists; see business_case_brief() in app.py."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.75 * inch, title=d["title"], author=d["author"])
    def section(title, *flowables):
        return KeepTogether([P(title, "h"), *flowables])

    story = [header_band(d), Spacer(1, 12), recommendation(d), Spacer(1, 12), kpi_row(d["kpis"]),
             section("THE SITUATION", P(d["situation"])),
             section("OPTIONS CONSIDERED",
                     table(["Option", "Breakdowns caught", "False alarms a year", "Net savings a year", "Assessment"],
                           d["options"], [0.2, 0.14, 0.14, 0.15, 0.37], highlight_rows=[len(d["options"]) - 1])),
             section("HOW THE SAVINGS ARE WORKED OUT",
                     table(["Each year", "Amount", "Basis"], d["calc"], [0.32, 0.14, 0.54], bold_rows=[len(d["calc"]) - 1])),
             section("RANGE OF OUTCOMES",
                     table(["Case", "Net savings a year", "3-year net value", "Payback"], d["cases"],
                           [0.34, 0.22, 0.22, 0.22], highlight_rows=[1]),
                     Spacer(1, 4), P(d["driver"], "small")),
             section("READINESS GAPS AND RISKS",
                     table(["Risk", "Effect", "Mitigation", "Owner"], d["risks"], [0.24, 0.27, 0.35, 0.14])),
             section("DECISION RULE",
                     table(["Recommendation", "When"], d["rules"], [0.25, 0.75], highlight_rows=[d["rule_index"]])),
             section("PROPOSED PLAN AND DECISION GATES",
                     table(["When", "What happens", "Gate to pass"], d["plan"], [0.16, 0.48, 0.36])),
             section("EVIDENCE BEHIND THE NUMBERS",
                     P("  |  ".join(f"<b>{t}:</b> {item} ({src})" for item, t, src in d["evidence"]), "small"))]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
