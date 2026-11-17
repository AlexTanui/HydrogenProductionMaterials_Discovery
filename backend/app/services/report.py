"""Builds the Phase 1-3 benchmark report served by `GET /benchmarks/report`
(the dashboard's "Build report" action).

Deliberately read-only and cheap, the same discipline as
`app.services.calibration`: every number here already exists in a
`BenchmarkResult` row (itself read from `experiments/results/*.json` and
`reports/calibration/*.json`, never recomputed) - this module only
formats those rows into a PDF document. It never touches a checkpoint, a
gold file, or a training run, so calling it repeatedly is free and always
reflects whichever checkpoints are currently scored.

Mirrors `frontend/src/pages/Benchmarks.tsx`'s section structure (same
bake-off/accuracy/UQ/calibration/limitations split) so the downloaded
report and the live dashboard page tell the same story - but it is its
own implementation, not a port of the TSX, since the two run in
different languages and the report is read offline rather than browsed.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.schemas import BenchmarkResult

# Mirrors frontend/src/pages/Benchmarks.tsx's PHASE_LABELS - kept as its
# own copy rather than shared, since one is TypeScript and one is Python;
# if a phase key is added to app.core.config.PHASE_CHECKPOINTS, add its
# label in both places.
PHASE_LABELS: dict[str, str] = {
    "phase1_deterministic": "Phase 1 — MPNN",
    "phase1_schnet": "Phase 1 — SchNet",
    "phase1_painn": "Phase 1 — PaiNN",
    "phase1_physnet": "Phase 1 — PhysNet",
    "phase1_dimenet": "Phase 1 — DimeNet++",
    "phase2_blip_k1": "Phase 2 — BLIP (K=1)",
    "phase2_blip": "Phase 2 — BLIP (K=4, NLL-trained)",
    "phase3_graph_stochastic_base": "Phase 3 — Graph-space (K=1, base)",
    "phase3_graph_stochastic": "Phase 3 — Graph-space (K=4, NLL-trained)",
}

PHASE1_KEYS = ["phase1_deterministic", "phase1_schnet", "phase1_painn", "phase1_physnet", "phase1_dimenet"]

_INK = colors.HexColor("#1b2a3a")
_MUTED = colors.HexColor("#5b6b7a")
_ACCENT = colors.HexColor("#1f6f54")
_RULE = colors.HexColor("#d8dee3")
_HEADER_BG = colors.HexColor("#eef2f1")
_ROW_ALT = colors.HexColor("#f7f9f8")


def _label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, textColor=_INK, spaceAfter=4, leading=24,
        ),
        "Subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=12.5, textColor=_ACCENT, spaceAfter=2, leading=16,
        ),
        "Meta": ParagraphStyle(
            "ReportMeta", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, textColor=_MUTED, spaceAfter=14, leading=12,
        ),
        "H2": ParagraphStyle(
            "ReportH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=14, textColor=_INK, spaceBefore=18, spaceAfter=6,
        ),
        "H3": ParagraphStyle(
            "ReportH3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=11.5, textColor=_ACCENT, spaceBefore=12, spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "ReportBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, textColor=_INK, leading=14.5, spaceAfter=8,
        ),
        "Caption": ParagraphStyle(
            "ReportCaption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8.5, textColor=_MUTED, leading=11, spaceBefore=4, spaceAfter=10,
        ),
        "Note": ParagraphStyle(
            "ReportNote", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, textColor=_MUTED, leading=13, spaceAfter=8,
        ),
    }


_CELL_STYLE = ParagraphStyle(
    "ReportCell", fontName="Helvetica", fontSize=8.5, leading=11, textColor=_INK,
)
_CELL_HEADER_STYLE = ParagraphStyle(
    "ReportCellHeader", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=_INK,
)


def _cell(text: str, header: bool = False) -> Paragraph:
    # Wrapped in a Paragraph (rather than left as a plain string) so long
    # labels (e.g. "Phase 3 — Graph-space (K=4, NLL-trained)") wrap within
    # their own cell's width instead of overflowing into the next column -
    # reportlab's Table does not reliably wrap a bare multi-word string.
    style = _CELL_HEADER_STYLE if header else _CELL_STYLE
    return Paragraph(text.replace("\n", "<br/>"), style)


def _table(header: list[str], body: list[list[str]], col_widths: list[float]) -> Table:
    data = [[_cell(h, header=True) for h in header]] + [
        [_cell(v) for v in row] for row in body
    ]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, _ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), _ROW_ALT))
    table.setStyle(TableStyle(style))
    return table


def _phase1_table(rows: list[BenchmarkResult], styles: dict[str, ParagraphStyle]):
    phase1 = [r for r in rows if r.phase in PHASE1_KEYS]
    if not phase1:
        return Paragraph("No Phase 1 checkpoint has been trained or scored yet.", styles["Note"])
    body = [
        [r.model, _fmt(r.energy_mae), _fmt(r.energy_rmse), _fmt(r.force_mae), _fmt(r.force_rmse)]
        for r in phase1
    ]
    header = ["Model", "Energy MAE\n(kcal/mol)", "Energy RMSE", "Force MAE\n(kcal/mol/Å)", "Force RMSE"]
    return _table(header, body, [1.65 * inch, 1.2 * inch, 1.05 * inch, 1.25 * inch, 1.05 * inch])


def _phase23_table(rows: list[BenchmarkResult], styles: dict[str, ParagraphStyle]):
    phase23 = [r for r in rows if r.phase not in PHASE1_KEYS]
    if not phase23:
        return Paragraph("No Phase 2 or Phase 3 checkpoint has been trained or scored yet.", styles["Note"])
    body = [[_label(r.phase), r.model, _fmt(r.energy_mae), _fmt(r.force_mae)] for r in phase23]
    header = ["Phase", "Model", "Energy MAE", "Force MAE"]
    return _table(header, body, [2.5 * inch, 1.5 * inch, 1.1 * inch, 1.1 * inch])


def _uq_table(rows: list[BenchmarkResult], styles: dict[str, ParagraphStyle]):
    stochastic = [r for r in rows if r.ece is not None or r.uncertainty_correlation is not None]
    if not stochastic:
        return Paragraph(
            "No stochastic-phase results are available yet; Phase 1 carries no uncertainty "
            "estimate by construction.", styles["Note"],
        )
    header = ["Phase", "Energy\nECE", "Force\nECE", "Energy\nSpearman", "Force\nSpearman",
              "Energy\ncov@90%", "Force\ncov@90%"]
    body = [
        [
            _label(r.phase), _fmt(r.ece), _fmt(r.force_ece),
            _fmt(r.uncertainty_correlation), _fmt(r.force_uncertainty_correlation),
            _pct(r.energy_coverage_90), _pct(r.force_coverage_90),
        ]
        for r in stochastic
    ]
    return _table(header, body, [1.75 * inch, 0.75 * inch, 0.75 * inch, 0.8 * inch, 0.8 * inch, 0.75 * inch, 0.75 * inch])


def _calibration_table(rows: list[BenchmarkResult], styles: dict[str, ParagraphStyle]):
    calibrated = [r for r in rows if r.energy_ece_calibrated is not None]
    if not calibrated:
        return None
    header = ["Phase", "Energy ECE\n(raw → calibrated)", "Force ECE\n(raw → calibrated)",
              "Energy cov@90%\n(raw → calibrated)", "Force cov@90%\n(raw → calibrated)"]
    body = [
        [
            _label(r.phase),
            f"{_fmt(r.ece)} → {_fmt(r.energy_ece_calibrated)}",
            f"{_fmt(r.force_ece)} → {_fmt(r.force_ece_calibrated)}",
            f"{_pct(r.energy_coverage_90)} → {_pct(r.energy_coverage_90_calibrated)}",
            f"{_pct(r.force_coverage_90)} → {_pct(r.force_coverage_90_calibrated)}",
        ]
        for r in calibrated
    ]
    return _table(header, body, [1.6 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])


def build_phase_report(rows: list[BenchmarkResult]) -> bytes:
    """Renders the full PDF document. `rows` is whatever `GET /benchmarks`
    would return right now (see `get_benchmark_results` in
    `app.api.benchmarks`) - this function is pure formatting, so the same
    `rows` always produces the same report.
    """
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.85 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        title="Phase 1-3 Benchmark Report",
        author="Hydrogen Production Materials Discovery",
    )

    generated_at = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    story = [
        Paragraph("Hydrogen Production Materials Discovery", styles["Title"]),
        Paragraph("Adaptive Graph-Space Uncertainty Modelling — Phase 1–3 Benchmark Report", styles["Subtitle"]),
        Paragraph(f"Generated {generated_at} from the project's live benchmark data", styles["Meta"]),
        Paragraph(
            "This report summarizes model accuracy and predictive-uncertainty quality across "
            "the project's three-phase evaluation on the MD17 molecular-dynamics benchmark: a "
            "deterministic baseline (Phase 1), weight-space stochasticity via multiple "
            "stochastic forward passes (Phase 2), and representation-space stochasticity "
            "(Phase 3). Every figure below is read directly from a scored checkpoint's own "
            "results file, not recomputed for this report, so it always matches the live "
            "dashboard.",
            styles["Body"],
        ),
    ]

    story.append(Paragraph("1. Phase 1 — Deterministic Model Comparison", styles["H2"]))
    story.append(Paragraph(
        "Five message-passing architectures — MPNN, SchNet, PaiNN, PhysNet, and DimeNet++ — "
        "were trained under one shared protocol (identical data, loss, and training budget) "
        "to identify the most accurate backbone before any uncertainty-estimation mechanism "
        "was introduced. Energy is reported in kcal/mol and force in kcal/mol/Å.",
        styles["Body"],
    ))
    story.append(_phase1_table(rows, styles))
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Phase 2 and Phase 3 — Accuracy", styles["H2"]))
    story.append(Paragraph(
        "Both later phases extend the same backbone used in Phase 1's deterministic row, "
        "rather than introducing a new architecture, so any difference in results is "
        "attributable to where stochasticity lives in the model — in the weights (Phase 2) "
        "or in the node and edge representations (Phase 3) — rather than to unrelated "
        "architectural choices. This is a different question from the Phase 1 bake-off above, "
        "so the two are not ranked directly against one another.",
        styles["Body"],
    ))
    story.append(_phase23_table(rows, styles))
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Uncertainty Quantification", styles["H2"]))
    story.append(Paragraph(
        "Expected calibration error (ECE) closer to zero indicates better-calibrated "
        "uncertainty; a Spearman correlation closer to one means predicted uncertainty "
        "actually tracks prediction error; coverage at the 90% level below 90% indicates "
        "overconfidence, and above 90% indicates overcautiousness. Energy and force are "
        "reported separately throughout, since on a given checkpoint the two are not on a "
        "comparable scale — pooling them would misrepresent whichever quantity the pooled "
        "figure is not.",
        styles["Body"],
    ))
    story.append(_uq_table(rows, styles))
    story.append(Spacer(1, 4))

    calibration_table = _calibration_table(rows, styles)
    if calibration_table is not None:
        story.append(Paragraph("4. Post-Hoc Calibration", styles["H2"]))
        story.append(Paragraph(
            "A single conformal scale factor, fit offline on each model's held-out validation "
            "split, applied at test time. Figures are shown as raw → calibrated, as a "
            "correction layered on top of a model's own uncertainty estimate rather than a "
            "different model.",
            styles["Body"],
        ))
        story.append(calibration_table)
        story.append(Spacer(1, 4))

    if any(r.phase.startswith("phase3_graph_stochastic") for r in rows):
        story.append(Paragraph("5. Resolved Finding — Phase 3 Force-Uncertainty Calibration", styles["H2"]))
        story.append(Paragraph(
            "The Phase 3 figures above come from the currently deployed checkpoint, which "
            "predates a fix described below. Post-hoc calibration substantially improves this "
            "checkpoint's energy calibration error, but only partially corrects its force "
            "calibration error. The underlying cause was architectural rather than numerical: "
            "the deployed mechanism injects stochasticity additively into node and edge "
            "representations, which has no first-order coupling to atomic position, whereas "
            "the Phase 2 mechanism injects stochasticity multiplicatively on the message "
            "itself, which does. Because this is a property of where noise enters the forward "
            "pass rather than a miscalibrated scalar, no post-hoc correction could fully close "
            "the gap on this checkpoint.",
            styles["Body"],
        ))
        story.append(Paragraph(
            "A structural revision — applying multiplicative noise at the message level, "
            "before aggregation — has since been implemented and scored against the same "
            "protocol. It corrects the underlying mechanism, restoring real first-order force "
            "coupling, though it trades the original undercoverage for a milder overcoverage "
            "rather than resolving the discrepancy outright on every metric. This finding has "
            "been root-caused, fixed, and formally closed; the revised checkpoint has not yet "
            "replaced the one reported above, pending review by the Phase 3 architecture's "
            "owner before it is deployed. The figures in this report will update once that "
            "review lands and the new checkpoint is in production.",
            styles["Body"],
        ))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Report generated automatically from the project's benchmarking dashboard. All MAE, "
        "RMSE, and calibration figures reflect the most recently scored checkpoint for each "
        "phase; no number in this report is recomputed or estimated independently of that "
        "scoring run.",
        styles["Caption"],
    ))

    doc.build(story)
    return buffer.getvalue()
