"""
report_generator.py — Generates the final HTML CEO Report.

Produces one beautiful, plain-English HTML file saved to outputs/.
The report is self-contained (all charts embedded as inline HTML/JS via Plotly).
It is readable by a CEO who knows nothing about the code.
"""

import logging
import math
import numbers
import os
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import plotly.io as pio

from .metrics_calculator import AllMetrics
from .panel_generator import Panel, PanelGenerator
from .utils import (
    fmt_currency, fmt_pct, fmt_multiple, fmt_number,
    outputs_dir, timestamp_str, delta_arrow, delta_color, safe_divide,
)
from .metric_interpreter import (
    build_interpretation_table_html,
    color_legend_html,
    interpret_metric,
    build_badge_html,
    classify_metric,
    METRIC_RULES,
)

logger = logging.getLogger("dashboard.report")

if TYPE_CHECKING:
    from .first_principles.schemas import FirstPrinciplesReport, QuestionResult

# ── Color constants ───────────────────────────────────────────────────────────
COLOR_STRONG  = "#27ae60"
COLOR_OK      = "#f39c12"
COLOR_WEAK    = "#e74c3c"
COLOR_NEUTRAL = "#7f8c8d"
COLOR_ACCENT  = "#2980b9"
COLOR_BG      = "#f9fafb"
COLOR_CARD    = "#ffffff"
COLOR_TEXT    = "#2c3e50"


# ── Recommendation styling ────────────────────────────────────────────────────
REC_STYLES = {
    "Strong Buy":    {"bg": "#eafaf1", "border": "#27ae60", "text": "#1a7a47"},
    "Buy":           {"bg": "#eafaf1", "border": "#27ae60", "text": "#1a7a47"},
    "Buy More":      {"bg": "#eafaf1", "border": "#27ae60", "text": "#1a7a47"},
    "Add Slowly":    {"bg": "#eafaf1", "border": "#27ae60", "text": "#1a7a47"},
    "Hold":          {"bg": "#fef9e7", "border": "#f39c12", "text": "#9a6107"},
    "Hold / Monitor":{"bg": "#fef9e7", "border": "#f39c12", "text": "#9a6107"},
    "Optional Trim": {"bg": "#fef5e7", "border": "#e67e22", "text": "#935116"},
    "Trim":          {"bg": "#fdecea", "border": "#e74c3c", "text": "#922b21"},
    "Sell Partial":  {"bg": "#fdecea", "border": "#e74c3c", "text": "#922b21"},
    "Sell":          {"bg": "#fdecea", "border": "#e74c3c", "text": "#922b21"},
    "Exit":          {"bg": "#f9ebea", "border": "#c0392b", "text": "#7b241c"},
    "Avoid":         {"bg": "#fdecea", "border": "#e74c3c", "text": "#922b21"},
    "Sell or Trim":  {"bg": "#fdecea", "border": "#e74c3c", "text": "#922b21"},
    # First-principles specific
    "Avoid/Trim":    {"bg": "#fdecea", "border": "#e74c3c", "text": "#922b21"},
    "Go":            {"bg": "#eafaf1", "border": "#27ae60", "text": "#1a7a47"},
    "Monitor":       {"bg": "#fef9e7", "border": "#f39c12", "text": "#9a6107"},
}


def _normalize_report_view(report_view: str) -> str:
    view = str(report_view or "present").strip().lower()
    return view if view in {"present", "past", "future", "master"} else "present"


def _fig_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an inline HTML div (no external dependencies)."""
    try:
        return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})
    except Exception as e:
        return f'<div style="color:red;padding:10px">Chart unavailable: {e}</div>'


def _metric_row(label: str, value: str, delta: str = "", color: str = COLOR_TEXT, delta_col: str = COLOR_NEUTRAL) -> str:
    """Single metric row for the metrics table."""
    delta_html = f' <span style="color:{delta_col};font-size:12px">{delta}</span>' if delta else ""
    return f"""
        <tr>
          <td style="padding:6px 12px;color:{COLOR_NEUTRAL};font-size:13px;border-bottom:1px solid #ecf0f1">{label}</td>
          <td style="padding:6px 12px;color:{color};font-weight:600;font-size:13px;border-bottom:1px solid #ecf0f1">
            {value}{delta_html}
          </td>
        </tr>"""


def _bullet_list(bullets: List[str], color: str = COLOR_TEXT) -> str:
    if not bullets:
        return '<p style="color:#7f8c8d;font-style:italic">No signals identified.</p>'
    items = "".join(
        f'<li style="margin:6px 0;color:{color};font-size:14px">{b}</li>'
        for b in bullets if b
    )
    return f'<ul style="padding-left:20px;margin:10px 0">{items}</ul>'


def _card(title: str, content: str, border_color: str = COLOR_ACCENT, icon: str = "") -> str:
    """HTML card container."""
    return f"""
    <div style="background:{COLOR_CARD};border-left:4px solid {border_color};
                border-radius:8px;padding:20px 24px;margin:16px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <h3 style="margin:0 0 12px 0;color:{COLOR_TEXT};font-size:15px;font-weight:700">
        {icon} {title}
      </h3>
      {content}
    </div>"""


def _data_quality_banner(metrics: AllMetrics) -> str:
    """
    Render a data-quality banner beneath the header.
    Shows a stale-filing warning if > 120 days, plus decision confidence.
    """
    flags = getattr(metrics, "decision_confidence_flags", [])
    confidence = getattr(metrics, "decision_confidence", 100.0)

    # Staleness badge
    staleness_badge = ""
    days_old = None
    label = metrics.staleness_label or ""
    import re
    m = re.search(r"last filing (\d+) days ago", label)
    if m:
        days_old = int(m.group(1))
    if days_old is not None and days_old > 120:
        staleness_badge = f"""
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:#fff3cd;border:1px solid #f0ad4e;
                     color:#856404;border-radius:6px;
                     padding:4px 12px;font-size:12px;font-weight:700">
          ⚠️ Stale filing data — last filing {days_old} days ago. Verify before action.
        </span>"""

    # Decision confidence badge
    if confidence >= 80:
        conf_color, conf_bg, conf_border = "#145a32", "#d5f5e3", "#27ae60"
        conf_icon = "✅"
    elif confidence >= 60:
        conf_color, conf_bg, conf_border = "#7d6608", "#fef9e7", "#f39c12"
        conf_icon = "🟡"
    elif confidence >= 40:
        conf_color, conf_bg, conf_border = "#a04000", "#fdebd0", "#e67e22"
        conf_icon = "🟠"
    else:
        conf_color, conf_bg, conf_border = "#922b21", "#fdecea", "#e74c3c"
        conf_icon = "🔴"

    flags_html = ""
    if flags:
        items = "".join(
            f'<li style="margin:3px 0;font-size:12px">{f}</li>'
            for f in flags
        )
        flags_html = f'<ul style="margin:6px 0 0 16px;padding:0;color:#5d6d7e">{items}</ul>'

    confidence_badge = f"""
    <div style="display:inline-flex;align-items:flex-start;gap:6px;
                background:{conf_bg};border:1px solid {conf_border};
                color:{conf_color};border-radius:6px;
                padding:6px 12px;font-size:12px">
      <div>
        <span style="font-weight:700">{conf_icon} Decision Confidence: {confidence:.0f}%</span>
        {flags_html}
      </div>
    </div>"""

    if not staleness_badge and confidence >= 80 and not flags:
        return ""  # nothing to show — data is clean

    return f"""
    <div style="max-width:1100px;margin:0 auto 0;padding:8px 20px;
                display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start">
      {staleness_badge}
      {confidence_badge}
    </div>"""


def _score_badge(score: float) -> str:
    """Colored score badge using 5-tier system."""
    if score >= 8.0:
        color, label = "#27ae60", "Excellent"
    elif score >= 6.5:
        color, label = "#52be80", "Good"
    elif score >= 5.0:
        color, label = "#f39c12", "Neutral"
    elif score >= 3.5:
        color, label = "#e67e22", "Weak"
    else:
        color, label = "#e74c3c", "Bad"
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:700">{score:.1f}/10</span>'
        f'&nbsp;<span style="font-size:11px;color:{color};font-weight:600">{label}</span>'
    )


def _recommendation_box(rec: str, explanation: str) -> str:
    style = REC_STYLES.get(rec, REC_STYLES["Hold"])
    return f"""
    <div style="background:{style['bg']};border:2px solid {style['border']};
                border-radius:12px;padding:24px 28px;margin:20px 0;text-align:center">
      <div style="font-size:32px;font-weight:900;color:{style['text']};letter-spacing:1px;margin-bottom:8px">
        {rec}
      </div>
      <div style="font-size:14px;color:{COLOR_TEXT};line-height:1.6;max-width:700px;margin:0 auto">
        {explanation}
      </div>
    </div>"""


def _decision_audit_card_html(panel: Panel) -> str:
    """
    Render the Decision Audit as a structured grid card.
    Shows each verdict field in a two-column table, followed by the one-sentence Why.
    """
    audit = panel.raw_data
    if not audit:
        return ""

    # Colour-code each verdict value
    def _verdict_color(label: str, value: str) -> str:
        lv = value.lower()
        if label in ("Company Verdict",):
            return {"good": COLOR_STRONG, "average": COLOR_OK, "weak": COLOR_WEAK}.get(lv, COLOR_TEXT)
        if label in ("Valuation Verdict",):
            return {"cheap": COLOR_STRONG, "fair": COLOR_OK,
                    "expensive": "#e67e22", "very expensive": COLOR_WEAK}.get(lv, COLOR_TEXT)
        if label in ("Portfolio Weight Verdict",):
            return {"underweight": COLOR_ACCENT, "near target": COLOR_STRONG,
                    "overweight": "#e67e22", "severely overweight": COLOR_WEAK}.get(lv, COLOR_TEXT)
        if label in ("Unrealized Gain/Loss",):
            return {"gain": COLOR_STRONG, "loss": COLOR_WEAK, "neutral": COLOR_NEUTRAL}.get(lv, COLOR_TEXT)
        if label in ("Final Action",):
            style = REC_STYLES.get(value, {})
            return style.get("text", COLOR_TEXT)
        return COLOR_TEXT

    rows_html = ""
    field_map = [
        ("Company Verdict",          audit.get("company_verdict")),
        ("Valuation Verdict",        audit.get("valuation_verdict")),
    ]
    if audit.get("weight_verdict") is not None:
        field_map.append(("Portfolio Weight Verdict", audit.get("weight_verdict")))
    if audit.get("gain_loss_verdict") is not None:
        field_map.append(("Unrealized Gain/Loss",    audit.get("gain_loss_verdict")))
    field_map += [
        ("Primary Driver",   audit.get("primary_driver")),
        ("Secondary Driver", audit.get("secondary_driver")),
        ("Final Action",     audit.get("final_action")),
    ]

    for label, value in field_map:
        if value is None:
            continue
        color = _verdict_color(label, str(value))
        rows_html += f"""
        <tr>
          <td style="padding:8px 14px;color:{COLOR_NEUTRAL};font-size:13px;
                     border-bottom:1px solid #ecf0f1;font-weight:600;width:44%">{label}</td>
          <td style="padding:8px 14px;color:{color};font-size:13px;
                     border-bottom:1px solid #ecf0f1;font-weight:700">{value}</td>
        </tr>"""

    why_text = audit.get("why", "")
    why_html = f"""
    <div style="background:#f4f6f7;border-left:4px solid {COLOR_ACCENT};border-radius:0 6px 6px 0;
                padding:12px 16px;margin-top:14px;font-size:14px;color:{COLOR_TEXT};line-height:1.6">
      <strong style="color:{COLOR_ACCENT}">Why:</strong> {why_text}
    </div>""" if why_text else ""

    content = f"""
    <table style="width:100%;border-collapse:collapse;margin-bottom:4px">
      {rows_html}
    </table>
    {why_html}"""

    return _card("Decision Audit — Why This Action?", content, border_color=COLOR_ACCENT, icon="🔍")


def generate_report(
    metrics: AllMetrics,
    panels: Dict[str, Panel],
    recommendation: str,
    recommendation_explanation: str,
    charts: Dict[str, go.Figure],
    open_in_browser: bool = True,
    peer_metrics: Optional[Dict[str, AllMetrics]] = None,
    notes: str = "",
    report_view: str = "present",
    first_principles_report: Optional["FirstPrinciplesReport"] = None,
) -> Path:
    """
    Generate the full CEO HTML report and save to outputs/.

    Parameters
    ----------
    metrics           : AllMetrics object for the primary ticker
    panels            : dict of panel name → Panel object
    recommendation    : recommendation string (e.g. "Buy")
    recommendation_explanation : one-paragraph explanation
    charts            : dict of chart name → Plotly Figure
    open_in_browser   : auto-open the report after saving
    peer_metrics      : optional dict of ticker → AllMetrics for peer comparison
    notes             : user notes to include in the report
    """
    view = _normalize_report_view(report_view)
    os.makedirs(outputs_dir(), exist_ok=True)
    mode = (metrics.mode or "new").strip().lower() or "new"
    filename = f"ceo_report_{metrics.ticker}_{mode}_{view}_{timestamp_str()}.html"
    output_path = outputs_dir() / filename

    html = _build_html(
        metrics=metrics,
        panels=panels,
        recommendation=recommendation,
        recommendation_explanation=recommendation_explanation,
        charts=charts,
        peer_metrics=peer_metrics,
        notes=notes,
        report_view=view,
        first_principles_report=first_principles_report,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"CEO Report saved to: {output_path}")
    print(f"\n✅  CEO Report saved → {output_path}")

    if open_in_browser:
        try:
            webbrowser.open(f"file://{output_path.resolve()}")
        except Exception as e:
            logger.warning(f"Could not auto-open browser: {e}")

    return output_path


def generate_master_report(
    new_metrics: AllMetrics,
    new_panels: Dict[str, Panel],
    new_recommendation: str,
    new_recommendation_explanation: str,
    new_charts: Dict[str, go.Figure],
    existing_metrics: AllMetrics,
    existing_panels: Dict[str, Panel],
    existing_recommendation: str,
    existing_recommendation_explanation: str,
    existing_charts: Dict[str, go.Figure],
    first_principles_report: Optional["FirstPrinciplesReport"] = None,
    notes: str = "",
    open_in_browser: bool = True,
    peer_metrics: Optional[Dict[str, AllMetrics]] = None,
) -> Path:
    """
    Generate the master CEO HTML report combining New Position and Existing Holding
    across Present, Past, and Future views into a single self-contained HTML file.
    """
    os.makedirs(outputs_dir(), exist_ok=True)
    ticker = new_metrics.ticker
    filename = f"ceo_report_{ticker}_master_{timestamp_str()}.html"
    output_path = outputs_dir() / filename

    html = _build_master_html(
        new_metrics=new_metrics,
        new_panels=new_panels,
        new_recommendation=new_recommendation,
        new_recommendation_explanation=new_recommendation_explanation,
        new_charts=new_charts,
        existing_metrics=existing_metrics,
        existing_panels=existing_panels,
        existing_recommendation=existing_recommendation,
        existing_recommendation_explanation=existing_recommendation_explanation,
        existing_charts=existing_charts,
        first_principles_report=first_principles_report,
        notes=notes,
        peer_metrics=peer_metrics,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Master CEO Report saved to: {output_path}")
    print(f"\n✅  Master CEO Report saved → {output_path}")

    if open_in_browser:
        try:
            webbrowser.open(f"file://{output_path.resolve()}")
        except Exception as e:
            logger.warning(f"Could not auto-open browser: {e}")

    return output_path


def _build_html(
    metrics: AllMetrics,
    panels: Dict[str, Panel],
    recommendation: str,
    recommendation_explanation: str,
    charts: Dict[str, go.Figure],
    peer_metrics: Optional[Dict[str, AllMetrics]] = None,
    notes: str = "",
    report_view: str = "present",
    first_principles_report: Optional["FirstPrinciplesReport"] = None,
) -> str:
    view = _normalize_report_view(report_view)
    if view in {"past", "future"}:
        return _build_first_principles_html(
            metrics=metrics,
            panels=panels,
            recommendation=recommendation,
            recommendation_explanation=recommendation_explanation,
            notes=notes,
            report_view=view,
            first_principles_report=first_principles_report,
        )
    return _build_present_html(
        metrics=metrics,
        panels=panels,
        recommendation=recommendation,
        recommendation_explanation=recommendation_explanation,
        charts=charts,
        peer_metrics=peer_metrics,
        notes=notes,
    )



def _present_inner_content(
    metrics: AllMetrics,
    panels: Dict[str, Panel],
    recommendation: str,
    recommendation_explanation: str,
    charts: Dict[str, go.Figure],
    peer_metrics: Optional[Dict[str, AllMetrics]] = None,
    notes: str = "",
    gen_time: Optional[str] = None,
) -> str:
    """Inner body content for the Present view — excludes header, data_quality_banner, and container div."""
    m = metrics
    if gen_time is None:
        gen_time = datetime.now().strftime("%B %d, %Y %H:%M")

    current_price_str = fmt_currency(m.q6.current_price) if m.q6.current_price else "N/A"
    market_cap_str = fmt_currency(m.q1.market_cap) if m.q1.market_cap else "N/A"
    sector_str = m.q1.sector or "N/A"
    cap_cat_str = m.q1.cap_category or "N/A"

    exec_summary = f"""
    <h2 style="color:{COLOR_TEXT};font-size:18px;margin:0 0 8px">Executive Summary</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px">
      <table style="width:100%;border-collapse:collapse">
        {_metric_row("Current Price", current_price_str)}
        {_metric_row("Market Cap", market_cap_str)}
        {_metric_row("Sector", sector_str)}
        {_metric_row("Cap Category", cap_cat_str)}
        {_metric_row("Industry", m.q1.industry or "N/A")}
        {_metric_row("Country", m.q1.country or "N/A")}
      </table>
      <table style="width:100%;border-collapse:collapse">
        {_metric_row("Overall Score", f"{m.overall_score:.1f} / 10")}
        {_metric_row("TTM Revenue", fmt_currency(m.q1.ttm_revenue))}
        {_metric_row("Operating Margin", fmt_pct(m.q3.operating_margin))}
        {_metric_row("FCF Yield", fmt_pct(m.q4.fcf_yield))}
        {_metric_row("P/E Ratio", fmt_multiple(m.q6.pe_ratio))}
        {_metric_row("Debt/EBITDA", fmt_multiple(m.q5.debt_to_ebitda))}
      </table>
    </div>"""

    exec_card = _card("Company Snapshot", exec_summary, border_color=COLOR_ACCENT, icon="📊")
    rec_box = _recommendation_box(recommendation, recommendation_explanation)

    decision_audit_html = ""
    if "decision_audit" in panels and panels["decision_audit"]:
        decision_audit_html = _decision_audit_card_html(panels["decision_audit"])

    score_rows = "".join(
        _metric_row(k, _score_badge(v))
        for k, v in m.scores.items()
    )
    score_table = f'<table style="width:100%;border-collapse:collapse">{score_rows}</table>'
    score_card = _card("Fundamental Scores — 7 Questions", score_table, border_color=COLOR_ACCENT, icon="🎯")

    legend_html = color_legend_html()
    ceo_summary = _build_ceo_decision_summary(m, recommendation, recommendation_explanation)

    scorecard_chart_html = ""
    if "scorecard_bars" in charts:
        scorecard_chart_html = _fig_to_html(charts["scorecard_bars"])
    if "scorecard_radar" in charts:
        scorecard_chart_html += _fig_to_html(charts["scorecard_radar"])

    q_sections = _build_7q_sections(m, charts)

    panel_sections = []
    if "bullish" in panels:
        panel_sections.append(_panel_to_html(panels["bullish"], border=COLOR_STRONG, icon="✅"))
    if "hold" in panels:
        panel_sections.append(_panel_to_html(panels["hold"], border=COLOR_OK, icon="⏸️"))
    if "bearish" in panels:
        panel_sections.append(_panel_to_html(panels["bearish"], border=COLOR_WEAK, icon="⚠️", is_warning=True))
    if "dividends" in panels:
        panel_sections.append(_panel_to_html(panels["dividends"], border=COLOR_ACCENT, icon="💰"))
    if "portfolio" in panels and panels["portfolio"]:
        panel_sections.append(_panel_to_html(panels["portfolio"], border="#8e44ad", icon="📁"))
    if "ml" in panels and panels["ml"]:
        panel_sections.append(_panel_to_html(panels["ml"], border="#2c3e50", icon="🤖", is_ml=True))

    peer_section = ""
    if peer_metrics:
        peer_section = _build_peer_section(m, peer_metrics)

    price_chart_html = ""
    if "price_history" in charts:
        price_chart_html = f"""
        <div style="margin:24px 0">
          <h2 style="color:{COLOR_TEXT};font-size:18px;margin:0 0 8px">Price History</h2>
          {_fig_to_html(charts["price_history"])}
        </div>"""

    notes_html = ""
    if notes or m.notes:
        note_text = notes or m.notes
        notes_html = _card(
            "Analyst Notes",
            f'<p style="font-size:14px;color:{COLOR_TEXT};line-height:1.7;white-space:pre-wrap">{note_text}</p>',
            border_color=COLOR_NEUTRAL,
            icon="📝",
        )

    footer = f"""
    <div style="text-align:center;color:{COLOR_NEUTRAL};font-size:12px;padding:32px 16px;
                border-top:1px solid #ecf0f1;margin-top:40px">
      <p>Stock Fundamentals CEO Dashboard — Generated {gen_time}</p>
      <p>{m.staleness_label}</p>
      <p style="margin-top:8px;font-style:italic">
        This report is for informational purposes only and does not constitute financial advice.
        All data sourced from public filings via yfinance. Verify critical figures independently.
      </p>
    </div>"""

    return "\n".join([
        exec_card,
        rec_box,
        decision_audit_html,
        score_card,
        legend_html,
        scorecard_chart_html,
        price_chart_html,
        q_sections,
        ceo_summary,
        *panel_sections,
        peer_section,
        notes_html,
        footer,
    ])


def _build_present_html(
    metrics: AllMetrics,
    panels: Dict[str, Panel],
    recommendation: str,
    recommendation_explanation: str,
    charts: Dict[str, go.Figure],
    peer_metrics: Optional[Dict[str, AllMetrics]] = None,
    notes: str = "",
) -> str:
    m = metrics
    mode_label = "Existing Holding" if m.mode == "existing" else "New Position Analysis"
    gen_time = datetime.now().strftime("%B %d, %Y %H:%M")

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>'

    header = f"""
    <div style="background:linear-gradient(135deg,#1a252f,#2c3e50);
                padding:32px 40px;color:white;border-radius:0 0 12px 12px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:13px;text-transform:uppercase;letter-spacing:2px;opacity:0.7;margin-bottom:4px">
            Stock Fundamentals CEO Dashboard
          </div>
          <div style="font-size:42px;font-weight:900;letter-spacing:2px">{m.ticker}</div>
          <div style="font-size:18px;opacity:0.85;margin-top:2px">{m.company_name}</div>
          <div style="font-size:12px;opacity:0.6;margin-top:8px">{mode_label}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:13px;opacity:0.7">Generated</div>
          <div style="font-size:15px;font-weight:600">{gen_time}</div>
          <div style="font-size:11px;opacity:0.6;margin-top:4px">{m.staleness_label}</div>
          {_overall_score_widget(m.overall_score)}
        </div>
      </div>
    </div>"""

    data_quality_banner = _data_quality_banner(metrics)
    inner = _present_inner_content(
        metrics=metrics,
        panels=panels,
        recommendation=recommendation,
        recommendation_explanation=recommendation_explanation,
        charts=charts,
        peer_metrics=peer_metrics,
        notes=notes,
        gen_time=gen_time,
    )

    body = "\n".join([
        header,
        data_quality_banner,
        '<div style="max-width:1100px;margin:0 auto;padding:0 20px">',
        inner,
        "</div>",
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CEO Report — {m.ticker} — {datetime.now().strftime('%Y-%m-%d')}</title>
  {plotly_cdn}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      background: {COLOR_BG};
      color: {COLOR_TEXT};
      margin: 0;
      padding: 0;
      line-height: 1.5;
    }}
    table {{ border-collapse: collapse; }}
    @media (max-width: 700px) {{
      div[style*="grid-template-columns:1fr 1fr"] {{
        display: block !important;
      }}
    }}
    .question-section {{
      background: {COLOR_CARD};
      border-radius: 8px;
      padding: 20px 24px;
      margin: 16px 0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }}
    .question-header {{
      font-size: 16px;
      font-weight: 700;
      color: {COLOR_TEXT};
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 2px solid #ecf0f1;
    }}
    .question-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {COLOR_NEUTRAL};
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>"""

_SIGNAL_COLORS = {
    "Strong": COLOR_STRONG,
    "Watch": COLOR_OK,
    "Weak": COLOR_WEAK,
    "Insufficient": COLOR_NEUTRAL,
}


# question_id -> (guide_metric_key, source_key_or_path, display_format)
_FP_GUIDE_ROW_SPECS: Dict[str, List[Tuple[str, Any, str]]] = {
    "P1": [
        ("real_owner_earnings_cagr_5y", "real_owner_earnings_cagr_5y", "pct"),
        ("real_fcf_per_share_cagr_5y", "real_fcf_per_share_cagr_5y", "pct"),
        ("real_revenue_per_share_cagr_5y", "real_revenue_per_share_cagr_5y", "pct"),
        ("cumulative_real_owner_earnings_change", "cumulative_real_owner_earnings_change", "pct"),
    ],
    "P2": [
        ("median_incremental_roic_5y", "median_incremental_roic_5y", "pct"),
        ("incremental_roic_years_below_watch", "incremental_roic_years_below_watch", "number"),
    ],
    "P3": [
        ("accrual_ratio_latest", "accrual_ratio_latest", "pct"),
        ("cfo_to_ni_latest", "cfo_to_ni_latest", "multiple"),
    ],
    "P4": [
        ("revenue_cagr_5y", "revenue_cagr_5y", "pct"),
        ("share_cagr_5y", "share_cagr_5y", "pct"),
        ("dilution_adjusted_growth", "dilution_adjusted_growth", "pct"),
        ("net_debt_trend", "net_debt_trend", "pct"),
    ],
    "P5": [
        ("fcf_payout_coverage_latest", "fcf_payout_coverage_latest", "multiple"),
        ("payout_to_fcf_5y", "payout_to_fcf_5y", "multiple"),
    ],
    "P6": [
        ("min_fcf_margin", "min_fcf_margin", "pct"),
        ("interest_coverage_trough", "interest_coverage_trough", "multiple"),
        ("recovery_speed_years", "recovery_speed_years", "number"),
    ],
    "F1": [
        ("forward_incremental_roic_spread", "forward_incremental_roic_spread", "pct"),
        ("base_incremental_roic", "base_incremental_roic", "pct"),
    ],
    "F2": [
        ("scenario_bear_growth", ("growth_assumptions", "bear"), "pct"),
        ("scenario_base_growth", ("growth_assumptions", "base"), "pct"),
        ("scenario_bull_growth", ("growth_assumptions", "bull"), "pct"),
        ("bear_drawdown", "bear_drawdown", "pct"),
    ],
    "F3": [
        ("operating_margin_cv", "operating_margin_cv", "multiple"),
        ("operating_margin_latest", "operating_margin_latest", "pct"),
        ("historical_margin_p90", "historical_margin_p90", "pct"),
        ("required_margin_base_case", "required_margin_base_case", "pct"),
    ],
    "F4": [
        ("stress_interest_coverage", "stress_interest_coverage", "multiple"),
        ("stress_net_debt_ebitda_proxy", "stress_net_debt_ebitda_proxy", "multiple"),
        ("liquidity_runway_current_ratio", "liquidity_runway_current_ratio", "multiple"),
    ],
    "F5": [
        ("implied_growth_reverse_dcf", "implied_growth_reverse_dcf", "pct"),
        ("historical_growth_p60", "historical_growth_p60", "pct"),
        ("historical_growth_p90", "historical_growth_p90", "pct"),
    ],
    "F6": [
        ("expected_return_no_multiple_expansion", "expected_return_no_multiple_expansion", "pct"),
        ("cash_yield", "cash_yield", "pct"),
        ("real_growth", "real_growth", "pct"),
        ("dilution_friction", "dilution_friction", "pct"),
    ],
}


def _signal_badge(signal: str) -> str:
    color = _SIGNAL_COLORS.get(signal, COLOR_NEUTRAL)
    return (
        f'<span style="background:{color};color:white;padding:4px 10px;border-radius:999px;'
        f'font-size:12px;font-weight:700;letter-spacing:0.2px">{escape(signal)}</span>'
    )


def _coverage_badge(coverage_ratio: float) -> str:
    ratio = coverage_ratio if isinstance(coverage_ratio, (int, float)) else float("nan")
    if not math.isfinite(ratio):
        label = "Coverage N/A"
        color = COLOR_NEUTRAL
    else:
        label = f"Coverage {ratio:.0%}"
        color = COLOR_STRONG if ratio >= 0.85 else COLOR_OK if ratio >= 0.70 else COLOR_WEAK
    return (
        f'<span style="border:1px solid {color};color:{color};padding:3px 9px;border-radius:999px;'
        f'font-size:12px;font-weight:600;background:white">{label}</span>'
    )


def _format_metric_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        vf = float(value)
        if not math.isfinite(vf):
            return "N/A"
        if abs(vf) >= 1000:
            return f"{vf:,.2f}"
        if abs(vf) >= 1:
            return f"{vf:.4f}"
        return f"{vf:.4%}"
    if isinstance(value, dict):
        return escape(", ".join(f"{k}: {_format_metric_value(v)}" for k, v in value.items()))
    if isinstance(value, list):
        return escape(", ".join(_format_metric_value(v) for v in value))
    return escape(str(value))


def _metric_display_name(metric_key: str) -> str:
    rule = METRIC_RULES.get(metric_key, {})
    return str(rule.get("display_name") or metric_key.replace("_", " ").title())


def _coerce_scalar(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        fv = float(value)
        if math.isfinite(fv):
            return fv
    return None


def _extract_metric_source(metrics: Dict[str, Any], source: Any) -> Any:
    if isinstance(source, str):
        return metrics.get(source)
    if isinstance(source, tuple) and len(source) == 2:
        container = metrics.get(source[0])
        if isinstance(container, dict):
            return container.get(source[1])
    return None


def _format_scalar_for_guide(value: float, kind: str) -> str:
    if kind == "pct":
        return fmt_pct(value)
    if kind == "multiple":
        return fmt_multiple(value)
    return f"{value:.2f}"


def _build_first_principles_interpretation_entries(question: "QuestionResult") -> List[Tuple[str, str, float]]:
    qid = str(getattr(question, "question_id", "")).upper()
    specs = _FP_GUIDE_ROW_SPECS.get(qid, [])
    metrics = getattr(question, "metrics", {}) or {}
    entries: List[Tuple[str, str, float]] = []
    for metric_key, source, kind in specs:
        raw = _extract_metric_source(metrics, source)
        scalar = _coerce_scalar(raw)
        if scalar is None:
            continue
        entries.append((metric_key, _format_scalar_for_guide(scalar, kind), scalar))
    return entries


def _build_technical_metrics_details(rows: Sequence[Tuple[str, str]]) -> str:
    if not rows:
        return ""
    body = "".join(
        "<tr>"
        f"<td style=\"padding:7px 10px;border-bottom:1px solid #ecf0f1;color:{COLOR_NEUTRAL};font-size:12px\">{escape(str(label))}</td>"
        f"<td style=\"padding:7px 10px;border-bottom:1px solid #ecf0f1;color:{COLOR_TEXT};font-size:12px;font-weight:600\">{escape(str(value))}</td>"
        "</tr>"
        for label, value in rows
    )
    return f"""
    <details style="margin-top:10px">
      <summary style="cursor:pointer;color:{COLOR_ACCENT};font-weight:600;font-size:13px">Technical metrics (expand)</summary>
      <div style="margin-top:8px;overflow-x:auto">
        <table style="width:100%;border-collapse:collapse">
          <thead>
            <tr>
              <th style="text-align:left;padding:7px 10px;background:#f4f6f7;font-size:12px;color:{COLOR_NEUTRAL}">Metric</th>
              <th style="text-align:left;padding:7px 10px;background:#f4f6f7;font-size:12px;color:{COLOR_NEUTRAL}">Value</th>
            </tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </details>
    """


def _build_interpretation_guide_details(interpretation_table_html: str) -> str:
    if not interpretation_table_html:
        return ""
    return f"""
    <details style="margin-top:12px">
      <summary style="cursor:pointer;color:{COLOR_ACCENT};font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:0.7px">
        Metric Interpretation Guide
      </summary>
      <div style="margin-top:6px">
        {interpretation_table_html}
      </div>
    </details>
    """


def _build_present_technical_rows(
    context_items: Sequence[Tuple[str, str]],
    metric_entries: Sequence[Tuple[str, str, Any]],
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    seen = set()
    for label, value in context_items:
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        key = ("ctx", label)
        if key in seen:
            continue
        seen.add(key)
        rows.append((label, value_str))

    for metric_key, fmt_value, _raw_value in metric_entries:
        label = _metric_display_name(metric_key)
        key = ("metric", label)
        if key in seen:
            continue
        seen.add(key)
        rows.append((label, str(fmt_value)))
    return rows


def _first_principles_chart_html(question: "QuestionResult") -> str:
    payload = getattr(question, "chart_payload", None)
    if payload is None or not getattr(payload, "series", None):
        return '<div style="font-size:13px;color:#7f8c8d;font-style:italic">Chart data unavailable.</div>'

    series_list = [s for s in payload.series if getattr(s, "x", None) and getattr(s, "y", None)]
    if not series_list:
        return '<div style="font-size:13px;color:#7f8c8d;font-style:italic">Chart data unavailable.</div>'

    fig = go.Figure()
    if payload.chart_type == "bar":
        for s in series_list:
            fig.add_trace(go.Bar(name=s.name, x=s.x, y=s.y))
    else:
        for s in series_list:
            fig.add_trace(go.Scatter(name=s.name, x=s.x, y=s.y, mode="lines+markers"))

    y_values: List[float] = []
    for s in series_list:
        for val in s.y:
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                y_values.append(fv)

    yaxis: Dict[str, object] = {"title": payload.y_label}
    if y_values:
        max_abs = max(abs(v) for v in y_values)
        if payload.unit in {"ratio"} and max_abs <= 1.5:
            yaxis["tickformat"] = ".0%"

    for ref in getattr(payload, "reference_lines", []):
        try:
            value = float(ref.value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        fig.add_hline(
            y=value,
            line_width=1.2,
            line_dash="dot",
            line_color="#95a5a6",
            annotation_text=ref.label,
            annotation_position="top right",
        )

    fig.update_layout(
        title={"text": payload.title, "font": {"size": 13}},
        margin={"l": 36, "r": 20, "t": 36, "b": 34},
        height=290,
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor="#fbfcfd",
        legend={"orientation": "h", "y": -0.25},
        xaxis={"title": payload.x_label},
        yaxis=yaxis,
    )
    return _fig_to_html(fig)


def _first_principles_question_card(question: "QuestionResult", sector: Optional[str] = None) -> str:
    signal = getattr(question, "signal", "Insufficient")
    color = _SIGNAL_COLORS.get(signal, COLOR_NEUTRAL)
    falsified = bool(getattr(question, "falsified", False))
    falsification_reason = getattr(question, "falsification_reason", None)
    coverage = getattr(question, "coverage", None)
    coverage_ratio = float(getattr(coverage, "coverage_ratio", float("nan")))
    takeaway = getattr(question, "takeaway", "") or getattr(question, "decision_reason", "")

    metrics = getattr(question, "metrics", {}) or {}
    technical_rows: List[Tuple[str, str]] = []
    for key in sorted(metrics.keys()):
        technical_rows.append((str(key), _format_metric_value(metrics[key])))

    falsification_text = "All principle checks passed."
    if falsified:
        falsification_text = f"Principle concern: {escape(falsification_reason or 'rule triggered')}"

    guide_entries = _build_first_principles_interpretation_entries(question)
    if guide_entries:
        guide_content = build_interpretation_table_html(guide_entries, sector)
    else:
        guide_content = '<div style="font-size:13px;color:#7f8c8d;font-style:italic">Guide data unavailable.</div>'
    guide_html = _build_interpretation_guide_details(guide_content)

    technical_html = _build_technical_metrics_details(technical_rows)

    content = f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap">
      <div style="flex:1;min-width:240px">
        <div style="font-size:16px;font-weight:800;color:{COLOR_TEXT}">{escape(question.question_id)}: {escape(question.question)}</div>
        <div style="margin-top:6px;font-size:14px;color:{COLOR_TEXT};line-height:1.55">
          <strong style="color:{color}">Main takeaway:</strong> {escape(takeaway)}
        </div>
      </div>
      {_signal_badge(signal)}
    </div>
    <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <span style="font-size:12px;color:{COLOR_NEUTRAL}">{falsification_text}</span>
      {_coverage_badge(coverage_ratio)}
    </div>
    <div style="margin-top:12px">{_first_principles_chart_html(question)}</div>
    {guide_html}
    {technical_html}
    """
    return _card(f"{question.question_id} Signal Card", content, border_color=color, icon="🧭")


# ── First-principles dashboard helper functions ───────────────────────────────

def _fp_hard_fail_banner(falsification_reasons: List[str]) -> str:
    """Amber thesis-concern card when one or more questions flagged a principle concern."""
    if not falsification_reasons:
        return ""
    items = "".join(
        f'<li style="margin:4px 0;font-size:13px">{escape(r)}</li>'
        for r in falsification_reasons
    )
    return f"""
    <div style="background:#fef9e7;border:2px solid #d35400;border-radius:8px;
                padding:16px 20px;margin:0 0 16px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
        <span style="font-size:20px">⚠️</span>
        <strong style="font-size:15px;color:#784212">Thesis At Risk — Principle Concerns Identified</strong>
      </div>
      <ul style="margin:0;padding-left:20px;color:#935116">{items}</ul>
      <div style="margin-top:8px;font-size:12px;color:#d35400;font-style:italic">
        One or more core investment principles are not currently supported by the data.
        Review the flagged questions and consider whether the investment thesis still holds.
      </div>
    </div>"""


def _fp_thesis_health_card(
    questions: list,
    hard_fail: bool,
    falsification_reasons: List[str],
    view: str,
    synthesis_counts: Dict[str, int],
) -> str:
    """Q7-style thesis health synthesis for past/future FP views.

    Answers the same three questions as Q7 in the present view:
    - What supports the thesis
    - What needs watching
    - What would change my mind
    """
    view_label = "past" if view == "past" else "forward"
    strong_qs  = [q for q in questions if str(getattr(q, "signal", "")) == "Strong"]
    watch_qs   = [q for q in questions if str(getattr(q, "signal", "")) == "Watch"]
    weak_qs    = [q for q in questions if str(getattr(q, "signal", "")) == "Weak"]
    flagged_qs = [q for q in questions if bool(getattr(q, "falsified", False))]

    # Thesis label
    if hard_fail or len(weak_qs) >= 2:
        label = "Thesis At Risk"
        label_color = "#d35400"
    elif len(watch_qs) > 0 or len(weak_qs) == 1:
        label = "Thesis Watch"
        label_color = "#f39c12"
    elif len(strong_qs) >= 4:
        label = "Thesis Intact"
        label_color = "#27ae60"
    else:
        label = "Thesis Watch"
        label_color = "#f39c12"

    # What supports
    if strong_qs:
        support_parts = []
        for q in strong_qs[:3]:
            qid = escape(str(getattr(q, "question_id", "")))
            short = escape(str(getattr(q, "takeaway", "") or getattr(q, "decision_reason", ""))[:120])
            support_parts.append(f"<strong style='color:#27ae60'>{qid}</strong> — {short}")
        supports_html = "<br>".join(support_parts)
    else:
        supports_html = f'<span style="color:#7f8c8d;font-style:italic">No {view_label} questions are currently Strong.</span>'

    # What needs watching
    watch_parts = []
    for q in watch_qs[:3]:
        qid = escape(str(getattr(q, "question_id", "")))
        short = escape(str(getattr(q, "takeaway", "") or getattr(q, "decision_reason", ""))[:100])
        watch_parts.append(f"<strong style='color:#f39c12'>{qid}</strong> — {short}")
    if flagged_qs and not hard_fail:
        for q in flagged_qs[:2]:
            qid = escape(str(getattr(q, "question_id", "")))
            reason = escape(str(getattr(q, "falsification_reason", "") or ""))[:100]
            watch_parts.append(f"<strong style='color:#d35400'>{qid}</strong> — Principle concern: {reason}")
    watching_html = (
        "<br>".join(watch_parts)
        if watch_parts
        else f'<span style="color:#7f8c8d;font-style:italic">No {view_label} questions currently flagged for monitoring.</span>'
    )

    # What would change my mind
    if weak_qs or flagged_qs:
        change_parts = []
        for q in (flagged_qs or weak_qs)[:3]:
            qid = escape(str(getattr(q, "question_id", "")))
            reason = (
                escape(str(getattr(q, "falsification_reason", "") or ""))
                or escape(str(getattr(q, "takeaway", "") or getattr(q, "decision_reason", ""))[:100])
            )
            change_parts.append(f"<strong style='color:#e74c3c'>{qid}</strong> — {reason}")
        change_html = "<br>".join(change_parts)
    else:
        change_html = (
            f'<span style="color:#7f8c8d">Watch for any {view_label} question weakening to Weak, '
            f'or coverage dropping on key data fields — that would shift this to Avoid/Trim.</span>'
        )

    def _section(title: str, body: str) -> str:
        return (
            f'<div style="margin:10px 0">'
            f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;'
            f'color:#7f8c8d;margin-bottom:4px">{title}</div>'
            f'<div style="font-size:13px;color:#2c3e50;line-height:1.55">{body}</div>'
            f'</div>'
        )

    content = f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
      <div>
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#7f8c8d">Thesis Status</div>
        <div style="font-size:22px;font-weight:900;color:{label_color}">{label}</div>
        <div style="font-size:12px;color:#7f8c8d">
          {synthesis_counts.get('Strong',0)} Strong · 
          {synthesis_counts.get('Watch',0)} Watch · 
          {synthesis_counts.get('Weak',0)} Weak · 
          {synthesis_counts.get('Insufficient',0)} N/A
        </div>
      </div>
    </div>
    <div style="border-top:1px solid #ecf0f1;padding-top:12px">
      {_section("What supports the thesis", supports_html)}
      {_section("What needs watching", watching_html)}
      {_section("What would change my mind", change_html)}
    </div>
    """
    return _card(
        f"Q7 — Thesis Health ({view_label.capitalize()} View)",
        content,
        border_color="#8e44ad",
        icon="🔍",
    )


def _fp_staleness_warning(staleness_label: str) -> str:
    """Warns prominently when filing data is old — critical for FP analysis."""
    import re
    m = re.search(r"last filing (\d+) days ago", staleness_label or "")
    if not m:
        return ""
    days = int(m.group(1))
    if days <= 120:
        return ""
    months = days // 30
    return f"""
    <div style="background:#fff3cd;border:2px solid #f0ad4e;border-radius:8px;
                padding:14px 18px;margin:0 0 16px;display:flex;gap:12px;align-items:flex-start">
      <span style="font-size:20px;flex-shrink:0">⚠️</span>
      <div>
        <strong style="font-size:14px;color:#856404">
          Stale Fundamental Data — Last Filing ~{months} Months Ago ({days} Days)
        </strong>
        <div style="font-size:13px;color:#856404;margin-top:4px">
          First-principles signals rely on point-in-time SEC filings. Data this old may predate
          material business changes. <strong>Verify with the latest 10-K/10-Q before acting.</strong>
        </div>
      </div>
    </div>"""


def _fp_signal_scorecard(questions: list) -> str:
    """Compact table of all P/F signal cards: ID → question short-form → signal badge → one-line takeaway."""
    if not questions:
        return ""

    # Short labels per question_id
    _SHORT_Q = {
        "P1": "Did per-share owner earnings compound in real terms?",
        "P2": "Did incremental capital earn attractive returns?",
        "P3": "Is earnings quality high (accruals low)?",
        "P4": "Was growth funded internally (not dilution/leverage)?",
        "P5": "Did capital allocation create value?",
        "P6": "Did the business survive stress periods?",
        "F1": "Is reinvestment runway still attractive?",
        "F2": "What is the 5Y owner-earnings distribution?",
        "F3": "Can margins sustain the base-case scenario?",
        "F4": "Can the balance sheet survive a stress scenario?",
        "F5": "What growth is implied by today's price?",
        "F6": "What is expected return without multiple expansion?",
    }

    rows = ""
    for q in questions:
        qid = str(getattr(q, "question_id", ""))
        signal = str(getattr(q, "signal", "Insufficient"))
        color = _SIGNAL_COLORS.get(signal, COLOR_NEUTRAL)
        takeaway = escape(str(getattr(q, "takeaway", "") or getattr(q, "decision_reason", ""))[:120])
        if len(getattr(q, "takeaway", "") or "") > 120:
            takeaway += "…"
        short_q = _SHORT_Q.get(qid, escape(str(getattr(q, "question", "")))[:80])
        falsified = bool(getattr(q, "falsified", False))
        fail_icon = ' <span title="Principle concern" style="color:#d35400">⚡</span>' if falsified else ""

        coverage = getattr(q, "coverage", None)
        cov_ratio = float(getattr(coverage, "coverage_ratio", float("nan")))
        cov_color = COLOR_STRONG if cov_ratio >= 0.85 else COLOR_OK if cov_ratio >= 0.70 else COLOR_WEAK
        cov_str = f"{cov_ratio:.0%}" if math.isfinite(cov_ratio) else "N/A"

        rows += f"""
        <tr style="border-bottom:1px solid #ecf0f1;vertical-align:middle">
          <td style="padding:8px 12px;font-weight:700;color:{color};font-size:13px;white-space:nowrap">
            {escape(qid)}{fail_icon}
          </td>
          <td style="padding:8px 12px;font-size:12px;color:{COLOR_NEUTRAL};max-width:240px">{short_q}</td>
          <td style="padding:8px 12px;text-align:center">
            <span style="background:{color};color:white;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700">{escape(signal)}</span>
          </td>
          <td style="padding:8px 12px;font-size:12px;color:{COLOR_TEXT};max-width:320px">{takeaway}</td>
          <td style="padding:8px 12px;text-align:center">
            <span style="font-size:11px;color:{cov_color};font-weight:600">{cov_str}</span>
          </td>
        </tr>"""

    content = f"""
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;min-width:640px">
        <thead>
          <tr style="background:#f4f6f7;border-bottom:2px solid #d5d8dc">
            <th style="padding:8px 12px;text-align:left;font-size:12px;color:{COLOR_NEUTRAL};width:50px">ID</th>
            <th style="padding:8px 12px;text-align:left;font-size:12px;color:{COLOR_NEUTRAL}">Question</th>
            <th style="padding:8px 12px;text-align:center;font-size:12px;color:{COLOR_NEUTRAL}">Signal</th>
            <th style="padding:8px 12px;text-align:left;font-size:12px;color:{COLOR_NEUTRAL}">Key Finding</th>
            <th style="padding:8px 12px;text-align:center;font-size:12px;color:{COLOR_NEUTRAL}">Coverage</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""
    return _card("Signal Scorecard — All Questions at a Glance", content, border_color=COLOR_ACCENT, icon="📋")


def _fp_investor_decision_card(
    questions: list,
    decision: str,
    hard_fail: bool,
    view: str,
    metrics: "AllMetrics",
    synthesis_counts: Dict[str, int],
) -> str:
    """Answers the 7 investor decision questions in a structured card."""

    # ── What is the company? ──────────────────────────────────────────────────
    company_name = escape(getattr(metrics, "company_name", "") or metrics.ticker)
    sector = escape(metrics.q1.sector or "N/A")
    price_str = fmt_currency(metrics.q6.current_price) if metrics.q6.current_price else "N/A"
    market_cap_str = fmt_currency(metrics.q1.market_cap) if metrics.q1.market_cap else "N/A"
    company_html = (
        f'<strong>{company_name}</strong> ({escape(metrics.ticker)}) — '
        f'{sector} · Price {price_str} · Market Cap {market_cap_str}'
    )

    # ── What is the recommendation? ──────────────────────────────────────────
    style = REC_STYLES.get(decision, REC_STYLES["Hold"])
    rec_html = (
        f'<span style="background:{style["border"]};color:white;padding:4px 14px;'
        f'border-radius:6px;font-weight:900;font-size:16px;letter-spacing:0.5px">'
        f'{escape(decision)}</span>'
    )

    # ── Why? ─────────────────────────────────────────────────────────────────
    strong = synthesis_counts.get("Strong", 0)
    watch = synthesis_counts.get("Watch", 0)
    weak = synthesis_counts.get("Weak", 0)
    insuff = synthesis_counts.get("Insufficient", 0)
    total = strong + watch + weak + insuff
    view_label = "past" if view == "past" else "forward"
    if hard_fail:
        why_text = (
            f"At least one {view_label} first-principles question has flagged a principle concern, "
            f"indicating that a core investment assumption is not currently supported by the data. "
            f"This drives the recommendation to Avoid/Trim."
        )
    elif weak >= 2:
        why_text = (
            f"Two or more {view_label} questions are Weak ({weak}/{total}), indicating that "
            f"core investment principles are not being met. The Avoid/Trim recommendation reflects "
            f"this pattern of deteriorating fundamentals."
        )
    elif strong >= 4 and weak == 0:
        why_text = (
            f"{strong} of {total} {view_label} questions are Strong with zero Weak signals, "
            f"indicating the investment thesis is robustly supported across all key dimensions."
        )
    elif insuff >= 3:
        why_text = (
            f"{insuff} of {total} {view_label} questions have Insufficient data coverage. "
            f"The Monitor recommendation reflects that the thesis cannot be fully confirmed or denied "
            f"without more complete data."
        )
    else:
        why_text = (
            f"The {view_label} scan shows {strong} Strong, {watch} Watch, {weak} Weak, and "
            f"{insuff} Insufficient signals out of {total} total. The mix is not compelling enough "
            f"for a Go decision but not weak enough for Avoid."
        )

    # ── Main strength ─────────────────────────────────────────────────────────
    strong_questions = [q for q in questions if str(getattr(q, "signal", "")) == "Strong"]
    strength_html = '<span style="color:#7f8c8d;font-style:italic">No strong signals identified.</span>'
    if strong_questions:
        sq = strong_questions[0]
        sq_takeaway = escape(str(getattr(sq, "takeaway", "") or getattr(sq, "decision_reason", ""))[:200])
        strength_html = (
            f'<strong style="color:{COLOR_STRONG}">{escape(sq.question_id)}:</strong> '
            f'{sq_takeaway}'
        )

    # ── Main risk ─────────────────────────────────────────────────────────────
    risk_questions = [q for q in questions if bool(getattr(q, "falsified", False))] or \
                     [q for q in questions if str(getattr(q, "signal", "")) == "Weak"]
    risk_html = '<span style="color:#7f8c8d;font-style:italic">No significant risk signals identified.</span>'
    if risk_questions:
        rq = risk_questions[0]
        rq_takeaway = escape(str(getattr(rq, "takeaway", "") or getattr(rq, "decision_reason", ""))[:200])
        icon = "⚡" if bool(getattr(rq, "falsified", False)) else "⚠️"
        strength_html_color = COLOR_WEAK
        risk_html = (
            f'<strong style="color:{COLOR_WEAK}">{escape(rq.question_id)} {icon}:</strong> '
            f'{rq_takeaway}'
        )

    # ── Missing / stale data ──────────────────────────────────────────────────
    insuff_questions = [q for q in questions if str(getattr(q, "signal", "")) == "Insufficient"]
    missing_parts = []
    for q in insuff_questions:
        cov = getattr(q, "coverage", None)
        missing_fields = list(getattr(cov, "missing_fields", []))
        if missing_fields:
            missing_parts.append(
                f'<strong style="color:{COLOR_NEUTRAL}">{escape(q.question_id)}:</strong> '
                f'missing {escape(", ".join(missing_fields[:4]))}'
                + (" + more" if len(missing_fields) > 4 else "")
            )
        else:
            missing_parts.append(f'<strong style="color:{COLOR_NEUTRAL}">{escape(q.question_id)}:</strong> low coverage')

    import re
    days_old = None
    m_stale = re.search(r"last filing (\d+) days ago", metrics.staleness_label or "")
    if m_stale:
        days_old = int(m_stale.group(1))
    if days_old and days_old > 120:
        stale_note = (
            f'<div style="margin-top:6px;font-size:12px;color:#856404">'
            f'⚠️ Underlying filing data is <strong>{days_old} days old</strong> '
            f'(~{days_old // 30} months). All signals based on this stale data.</div>'
        )
    else:
        stale_note = ""

    if missing_parts:
        missing_html = "<br>".join(missing_parts) + stale_note
    elif stale_note:
        missing_html = stale_note
    else:
        missing_html = '<span style="color:#27ae60;font-style:italic">All required data fields are available.</span>'

    # ── What would change the action? ────────────────────────────────────────
    if decision in ("Go",):
        flip_html = (
            f'<span style="color:{COLOR_WEAK}">This becomes Monitor if</span>: '
            f'any {view_label} question weakens to Watch or below, coverage drops on key fields, '
            f'or macro conditions change the hurdle rate assumptions.'
        )
    elif decision in ("Avoid/Trim", "Avoid"):
        flip_html = (
            f'<span style="color:{COLOR_OK}">This becomes Monitor if</span>: '
            f'flagged principle concerns are resolved (e.g., next filing restores data coverage), '
            f'weak signal count drops below 2, or valuation improves materially. '
            f'<span style="color:{COLOR_STRONG}">This becomes Go if</span>: '
            f'≥4 questions reach Strong with zero Weak.'
        )
    else:  # Monitor
        flip_html = (
            f'<span style="color:{COLOR_STRONG}">This becomes Go if</span>: '
            f'≥4 of {total} {view_label} questions reach Strong with zero Weak signals. '
            f'<span style="color:{COLOR_WEAK}">This becomes Avoid/Trim if</span>: '
            f'≥2 questions become Weak, or a principle concern is flagged.'
        )

    def _row(label: str, content: str, label_color: str = COLOR_NEUTRAL) -> str:
        return f"""
        <tr style="border-bottom:1px solid #f0f3f4">
          <td style="padding:10px 14px;font-size:13px;font-weight:700;color:{label_color};
                     white-space:nowrap;vertical-align:top;width:28%">{label}</td>
          <td style="padding:10px 14px;font-size:13px;color:{COLOR_TEXT};line-height:1.5">{content}</td>
        </tr>"""

    table_body = (
        _row("🏢 The Company", company_html)
        + _row("🎯 Recommendation", rec_html)
        + _row("💡 Why", why_text)
        + _row("✅ Main Strength", strength_html)
        + _row("⚠️ Main Risk", risk_html)
        + _row("📭 Missing / Stale Data", missing_html)
        + _row("🔄 What Would Change This", flip_html)
    )

    content = f"""
    <table style="width:100%;border-collapse:collapse">
      {table_body}
    </table>"""
    return _card(
        "Investment Decision Summary",
        content,
        border_color=style["border"],
        icon="🧠",
    )


def _fp_header_score_widget(synthesis_counts: Dict[str, int], hard_fail: bool) -> str:
    """Signal-count widget for the past/future report header (replaces present-view score)."""
    strong = synthesis_counts.get("Strong", 0)
    watch = synthesis_counts.get("Watch", 0)
    weak = synthesis_counts.get("Weak", 0)
    insuff = synthesis_counts.get("Insufficient", 0)
    total = strong + watch + weak + insuff

    badges = ""
    for label, count, color in [
        ("Strong", strong, "#27ae60"),
        ("Watch", watch, "#f39c12"),
        ("Weak", weak, "#e74c3c"),
        ("N/A", insuff, "#7f8c8d"),
    ]:
        if count > 0:
            badges += (
                f'<span style="display:inline-block;background:{color};color:white;'
                f'border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;margin:1px 2px">'
                f'{count} {label}</span>'
            )
    fail_badge = ""
    if hard_fail:
        fail_badge = (
            '<div style="margin-top:4px">'
            '<span style="background:#d35400;color:white;border-radius:4px;'
            'padding:2px 8px;font-size:11px;font-weight:700">⚠️ Thesis At Risk</span>'
            '</div>'
        )
    return f"""
    <div style="margin-top:10px;text-align:right">
      <div style="font-size:10px;opacity:0.7;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
        Signals ({total} questions)
      </div>
      <div>{badges}</div>
      {fail_badge}
    </div>"""



def _compute_fp_questions_state(
    first_principles_report: Optional["FirstPrinciplesReport"],
    view: str,
    recommendation: str,
) -> Tuple[list, str, Dict[str, int], bool, List[str]]:
    """Compute question state for a given FP view (past or future).

    Returns (questions, decision, synthesis_counts, hard_fail, falsification_reasons).
    """
    questions: list = []
    decision = recommendation
    synthesis_counts: Dict[str, int] = {"Strong": 0, "Watch": 0, "Weak": 0, "Insufficient": 0}
    hard_fail = False
    falsification_reasons: List[str] = []

    if first_principles_report is not None:
        all_questions = list(getattr(first_principles_report, "questions", []) or [])
        prefix = "P" if view == "past" else "F"
        questions = [q for q in all_questions if str(getattr(q, "question_id", "")).upper().startswith(prefix)]
        questions = sorted(questions, key=lambda q: str(getattr(q, "question_id", "")))
        for q in questions:
            sig = str(getattr(q, "signal", "Insufficient"))
            synthesis_counts[sig] = synthesis_counts.get(sig, 0) + 1
            if bool(getattr(q, "falsified", False)):
                hard_fail = True
                reason = str(getattr(q, "falsification_reason", "") or "")
                if reason:
                    falsification_reasons.append(f"{getattr(q, 'question_id', '?')}: {reason}")

        strong_count = synthesis_counts.get("Strong", 0)
        weak_count = synthesis_counts.get("Weak", 0)
        if hard_fail or weak_count >= 2:
            decision = "Avoid/Trim"
        elif strong_count >= 4 and weak_count == 0:
            decision = "Go"
        else:
            decision = "Monitor"

    return questions, decision, synthesis_counts, hard_fail, falsification_reasons


def _fp_inner_content(
    metrics: AllMetrics,
    recommendation: str,
    recommendation_explanation: str,
    notes: str,
    first_principles_report: Optional["FirstPrinciplesReport"],
    report_view: str,
    gen_time: Optional[str] = None,
) -> str:
    """Inner body content for a FP view — excludes outer header and container div."""
    m = metrics
    if gen_time is None:
        gen_time = datetime.now().strftime("%B %d, %Y %H:%M")
    view = _normalize_report_view(report_view)
    view_title = "Past" if view == "past" else "Future"

    questions, decision, synthesis_counts, hard_fail, falsification_reasons = _compute_fp_questions_state(
        first_principles_report, view, recommendation
    )

    hard_fail_banner = _fp_hard_fail_banner(falsification_reasons) if hard_fail else ""
    thesis_health_card = _fp_thesis_health_card(
        questions, hard_fail, falsification_reasons, view, synthesis_counts,
    )
    staleness_warning = _fp_staleness_warning(m.staleness_label or "")
    data_quality_banner_html = _data_quality_banner(metrics)

    hf_cell = (
        "<strong style='color:#d35400'>Thesis At Risk</strong>"
        if hard_fail else "Thesis Intact"
    )
    snapshot = _card(
        "Company Snapshot",
        f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <table style="width:100%;border-collapse:collapse">
            {_metric_row("Current Price", fmt_currency(m.q6.current_price) if m.q6.current_price else "N/A")}
            {_metric_row("Market Cap", fmt_currency(m.q1.market_cap) if m.q1.market_cap else "N/A")}
            {_metric_row("Sector", m.q1.sector or "N/A")}
            {_metric_row("Cap Category", m.q1.cap_category or "N/A")}
          </table>
          <table style="width:100%;border-collapse:collapse">
            {_metric_row("View", view_title + " First-Principles")}
            {_metric_row("Data Freshness", m.staleness_label or "N/A")}
            {_metric_row("Questions Assessed", str(len(questions)))}
            {_metric_row("Thesis Health", hf_cell)}
          </table>
        </div>
        """,
        border_color=COLOR_ACCENT,
        icon="📊",
    )

    decision_card = _fp_investor_decision_card(
        questions=questions,
        decision=decision,
        hard_fail=hard_fail,
        view=view,
        metrics=m,
        synthesis_counts=synthesis_counts,
    )

    rec_box = _recommendation_box(decision, recommendation_explanation)
    scorecard = _fp_signal_scorecard(questions)

    if not questions:
        question_cards = _card(
            f"{view_title} Question Cards",
            '<p style="font-size:14px;color:#7f8c8d">No first-principles payload available for this view.</p>',
            border_color=COLOR_NEUTRAL,
            icon="🧭",
        )
    else:
        question_cards = "".join(_first_principles_question_card(q, sector=m.q1.sector) for q in questions)

    notes_html = ""
    if notes or m.notes:
        note_text = notes or m.notes
        notes_html = _card(
            "Analyst Notes",
            f'<p style="font-size:14px;color:{COLOR_TEXT};line-height:1.7;white-space:pre-wrap">{escape(note_text)}</p>',
            border_color=COLOR_NEUTRAL,
            icon="📝",
        )

    footer = f"""
    <div style="text-align:center;color:{COLOR_NEUTRAL};font-size:12px;padding:32px 16px;
                border-top:1px solid #ecf0f1;margin-top:40px">
      <p>Stock Fundamentals CEO Dashboard — Generated {gen_time}</p>
      <p>{m.staleness_label}</p>
      <p style="margin-top:8px;font-style:italic">
        This report is for informational purposes only and does not constitute financial advice.
        Data sources: SEC EDGAR, FRED, Kenneth French Data Library, and public market data.
      </p>
    </div>"""

    detail_divider = (
        f'<h2 style="font-size:17px;font-weight:700;color:{COLOR_TEXT};'
        f'margin:36px 0 16px;padding-top:12px;border-top:2px solid #ecf0f1">'
        f'📂 Detailed Analysis — Individual Question Cards</h2>'
    )

    return "\n".join([
        hard_fail_banner,
        staleness_warning,
        data_quality_banner_html,
        snapshot,
        thesis_health_card,
        decision_card,
        rec_box,
        scorecard,
        detail_divider,
        question_cards,
        notes_html,
        footer,
    ])


def _build_first_principles_html(
    metrics: AllMetrics,
    panels: Dict[str, Panel],
    recommendation: str,
    recommendation_explanation: str,
    notes: str,
    report_view: str,
    first_principles_report: Optional["FirstPrinciplesReport"],
) -> str:
    m = metrics
    mode_label = "Existing Holding" if m.mode == "existing" else "New Position Analysis"
    gen_time = datetime.now().strftime("%B %d, %Y %H:%M")
    view = _normalize_report_view(report_view)
    view_title = "Past" if view == "past" else "Future"

    # Need synthesis_counts and hard_fail for the header score widget
    _, _, synthesis_counts, hard_fail, _ = _compute_fp_questions_state(
        first_principles_report, view, recommendation
    )

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>'
    header = f"""
    <div style="background:linear-gradient(135deg,#1a252f,#2c3e50);
                padding:32px 40px;color:white;border-radius:0 0 12px 12px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:13px;text-transform:uppercase;letter-spacing:2px;opacity:0.7;margin-bottom:4px">
            Stock Fundamentals CEO Dashboard
          </div>
          <div style="font-size:42px;font-weight:900;letter-spacing:2px">{m.ticker}</div>
          <div style="font-size:18px;opacity:0.85;margin-top:2px">{m.company_name}</div>
          <div style="font-size:12px;opacity:0.75;margin-top:8px">{mode_label} · First-Principles {view_title} View</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:13px;opacity:0.7">Generated</div>
          <div style="font-size:15px;font-weight:600">{gen_time}</div>
          <div style="font-size:11px;opacity:0.6;margin-top:4px">{m.staleness_label}</div>
          {_fp_header_score_widget(synthesis_counts, hard_fail)}
        </div>
      </div>
    </div>"""

    inner = _fp_inner_content(
        metrics=metrics,
        recommendation=recommendation,
        recommendation_explanation=recommendation_explanation,
        notes=notes,
        first_principles_report=first_principles_report,
        report_view=view,
        gen_time=gen_time,
    )

    body = "\n".join([
        header,
        '<div style="max-width:1100px;margin:0 auto;padding:0 20px">',
        inner,
        "</div>",
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CEO Report — {m.ticker} — {view_title} — {datetime.now().strftime('%Y-%m-%d')}</title>
  {plotly_cdn}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      background: {COLOR_BG};
      color: {COLOR_TEXT};
      margin: 0;
      padding: 0;
      line-height: 1.5;
    }}
    table {{ border-collapse: collapse; }}
    @media (max-width: 700px) {{
      div[style*="grid-template-columns:1fr 1fr"] {{
        display: block !important;
      }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>"""

def _overall_score_widget(score: float) -> str:
    if score >= 8.0:
        color = "#27ae60"
    elif score >= 6.5:
        color = "#52be80"
    elif score >= 5.0:
        color = "#f39c12"
    elif score >= 3.5:
        color = "#e67e22"
    else:
        color = "#e74c3c"
    return f"""
    <div style="margin-top:12px;text-align:center">
      <div style="font-size:11px;opacity:0.7;text-transform:uppercase;letter-spacing:1px">Overall Score</div>
      <div style="font-size:36px;font-weight:900;color:{color}">{score:.1f}</div>
      <div style="font-size:11px;opacity:0.5">out of 10</div>
    </div>"""


def _build_7q_sections(m: AllMetrics, charts: Dict[str, go.Figure]) -> str:
    """Build the 7-question detail sections with metric interpretation guides."""
    sector = m.q1.sector
    sections = []

    # ── Q1 ─────────────────────────────────────────────────────────────────────
    q1_ctx = [
        ("Sector / Industry", f"{m.q1.sector} / {m.q1.industry}"),
        ("Cap Category", m.q1.cap_category),
        ("Country", m.q1.country),
        ("Employees", f"{m.q1.employee_count:,}" if m.q1.employee_count else "N/A"),
    ]
    if m.mode == "existing" and m.q1.revenue_delta_pct is not None:
        q1_ctx.append(("Revenue Change Since Purchase", fmt_pct(m.q1.revenue_delta_pct)))

    q1_metric_entries = [
        ("market_cap",       fmt_currency(m.q1.market_cap),       m.q1.market_cap),
        ("ttm_revenue",      fmt_currency(m.q1.ttm_revenue),      m.q1.ttm_revenue),
        ("enterprise_value", fmt_currency(m.q1.enterprise_value), m.q1.enterprise_value),
    ]
    q1_interp = build_interpretation_table_html(q1_metric_entries, sector)
    q1_technical_rows = _build_present_technical_rows(q1_ctx, q1_metric_entries)

    sections.append(_q_section(
        number=1,
        question="Is this business at a meaningful scale for my portfolio, and has that scale improved or deteriorated since I bought?",
        score=m.scores.get("Q1_Scale"),
        chart_html=_fig_to_html(charts["q1_revenue"]) if "q1_revenue" in charts else "",
        border_color=COLOR_ACCENT,
        why_matters="Scale determines whether the business has the revenue base, brand, and distribution to sustain competitive advantages. Larger companies typically have lower risk of failure and better access to capital.",
        main_takeaway=_q1_takeaway(m),
        interpretation_table=q1_interp,
        context_items=q1_ctx,
        technical_rows=q1_technical_rows,
    ))

    # ── Q2 ─────────────────────────────────────────────────────────────────────
    q2_ctx = [
        ("EPS (TTM)", fmt_multiple(m.q2.eps_ttm)),
        ("Growth Trend", m.q2.growth_trend or "N/A"),
    ]
    if m.mode == "existing" and m.q2.revenue_cagr_at_purchase is not None:
        q2_ctx.append(("Revenue CAGR at Purchase", fmt_pct(m.q2.revenue_cagr_at_purchase)))

    q2_metric_entries = [
        ("revenue_1yr_cagr",       fmt_pct(m.q2.revenue_1yr_cagr),       m.q2.revenue_1yr_cagr),
        ("revenue_3yr_cagr",       fmt_pct(m.q2.revenue_3yr_cagr),       m.q2.revenue_3yr_cagr),
        ("revenue_5yr_cagr",       fmt_pct(m.q2.revenue_5yr_cagr),       m.q2.revenue_5yr_cagr),
        ("earnings_1yr_growth",    fmt_pct(m.q2.earnings_1yr_growth),    m.q2.earnings_1yr_growth),
        ("forward_revenue_growth", fmt_pct(m.q2.forward_revenue_growth), m.q2.forward_revenue_growth),
    ]
    q2_interp = build_interpretation_table_html(q2_metric_entries, sector)
    q2_technical_rows = _build_present_technical_rows(q2_ctx, q2_metric_entries)

    sections.append(_q_section(
        number=2,
        question="Is the business growing at a rate that will compound my portfolio value, and has growth accelerated, stayed stable, or slowed since I bought?",
        score=m.scores.get("Q2_Growth"),
        chart_html=_fig_to_html(charts["q2_revenue_cagr"]) if "q2_revenue_cagr" in charts else "",
        border_color=COLOR_STRONG,
        why_matters="Growth is the engine of long-term returns. A company that grows revenue and earnings consistently will typically see its stock price follow. Slowing growth can compress multiples and erode portfolio value even if the business remains profitable.",
        main_takeaway=_q2_takeaway(m),
        interpretation_table=q2_interp,
        context_items=q2_ctx,
        technical_rows=q2_technical_rows,
    ))

    # ── Q3 ─────────────────────────────────────────────────────────────────────
    q3_ctx = [("Margin Trend", m.q3.margin_trend or "N/A")]
    if m.mode == "existing":
        if m.q3.gross_margin_at_purchase is not None:
            q3_ctx.append(("Gross Margin at Purchase", fmt_pct(m.q3.gross_margin_at_purchase)))
        if m.q3.operating_margin_at_purchase is not None:
            q3_ctx.append(("Op. Margin at Purchase", fmt_pct(m.q3.operating_margin_at_purchase)))

    chart_html = ""
    if "q3_margins" in charts: chart_html += _fig_to_html(charts["q3_margins"])
    if "q3_roic_gauge" in charts: chart_html += _fig_to_html(charts["q3_roic_gauge"])

    q3_metric_entries = [
        ("gross_margin",     fmt_pct(m.q3.gross_margin),     m.q3.gross_margin),
        ("operating_margin", fmt_pct(m.q3.operating_margin), m.q3.operating_margin),
        ("net_margin",       fmt_pct(m.q3.net_margin),       m.q3.net_margin),
        ("ebitda_margin",    fmt_pct(m.q3.ebitda_margin),    m.q3.ebitda_margin),
        ("roe",              fmt_pct(m.q3.roe),              m.q3.roe),
        ("roa",              fmt_pct(m.q3.roa),              m.q3.roa),
        ("roic",             fmt_pct(m.q3.roic),             m.q3.roic),
    ]
    q3_interp = build_interpretation_table_html(q3_metric_entries, sector)
    q3_technical_rows = _build_present_technical_rows(q3_ctx, q3_metric_entries)

    sections.append(_q_section(
        number=3,
        question="Are profits high-quality, stable, and ideally expanding, and have margins held or improved since I bought?",
        score=m.scores.get("Q3_Profitability"),
        chart_html=chart_html,
        border_color=COLOR_STRONG,
        why_matters="Profitability tells us whether the company converts revenue into real earnings efficiently. Strong and expanding margins usually signal pricing power, brand strength, or operational leverage — characteristics of a durable business.",
        main_takeaway=_q3_takeaway(m),
        interpretation_table=q3_interp,
        context_items=q3_ctx,
        technical_rows=q3_technical_rows,
    ))

    # ── Q4 ─────────────────────────────────────────────────────────────────────
    # Absolute dollar values go in context strip; ratios go in interpretation table
    q4_ctx = [
        ("Operating Cash Flow", fmt_currency(m.q4.operating_cash_flow)),
        ("Free Cash Flow", fmt_currency(m.q4.free_cash_flow)),
        ("FCF per Share", fmt_currency(m.q4.fcf_per_share)),
        ("CapEx", fmt_currency(m.q4.capital_expenditure)),
        ("Buybacks (TTM)", fmt_currency(m.q4.buybacks_ttm)),
    ]
    if m.mode == "existing" and m.q4.fcf_at_purchase is not None:
        q4_ctx.append(("FCF at Purchase", fmt_currency(m.q4.fcf_at_purchase)))

    chart_html = ""
    if "q4_cashflow_waterfall" in charts: chart_html += _fig_to_html(charts["q4_cashflow_waterfall"])
    elif "q4_cashflow_bars" in charts: chart_html += _fig_to_html(charts["q4_cashflow_bars"])
    if "q4_fcf_yield_gauge" in charts: chart_html += _fig_to_html(charts["q4_fcf_yield_gauge"])

    capex_intensity = safe_divide(m.q4.capital_expenditure, m.q1.ttm_revenue)
    q4_metric_entries = [
        ("fcf_margin",      fmt_pct(m.q4.fcf_margin),  m.q4.fcf_margin),
        ("fcf_yield",       fmt_pct(m.q4.fcf_yield),   m.q4.fcf_yield),
        ("capex_intensity", fmt_pct(capex_intensity),   capex_intensity),
        ("roic",            fmt_pct(m.q4.roic),         m.q4.roic),
    ]
    q4_interp = build_interpretation_table_html(q4_metric_entries, sector)
    q4_technical_rows = _build_present_technical_rows(q4_ctx, q4_metric_entries)

    sections.append(_q_section(
        number=4,
        question="Is the company generating reliable free cash flow that can be returned to owners or reinvested at high returns, and has cash flow strength improved since I bought?",
        score=m.scores.get("Q4_CashFlow"),
        chart_html=chart_html,
        border_color=COLOR_ACCENT,
        why_matters="Free cash flow is the lifeblood of a business. It funds dividends, buybacks, debt repayment, and growth reinvestment. Companies that generate strong FCF consistently are compounders; those that burn cash depend on external financing.",
        main_takeaway=_q4_takeaway(m),
        interpretation_table=q4_interp,
        context_items=q4_ctx,
        technical_rows=q4_technical_rows,
    ))

    # ── Q5 ─────────────────────────────────────────────────────────────────────
    q5_ctx = [
        ("Total Debt", fmt_currency(m.q5.total_debt)),
        ("Cash & Equivalents", fmt_currency(m.q5.cash_and_equivalents)),
        ("Net Debt", fmt_currency(m.q5.net_debt)),
        ("Financial Risk Trend", m.q5.financial_risk_trend or "N/A"),
    ]
    if m.mode == "existing" and m.q5.debt_to_ebitda_at_purchase is not None:
        q5_ctx.append(("Debt/EBITDA at Purchase", fmt_multiple(m.q5.debt_to_ebitda_at_purchase)))

    chart_html = ""
    if "q5_debt_ebitda_gauge" in charts: chart_html += _fig_to_html(charts["q5_debt_ebitda_gauge"])
    if "q5_balance_sheet_bars" in charts: chart_html += _fig_to_html(charts["q5_balance_sheet_bars"])

    q5_metric_entries = [
        ("debt_to_ebitda",    fmt_multiple(m.q5.debt_to_ebitda),    m.q5.debt_to_ebitda),
        ("debt_to_equity",    fmt_multiple(m.q5.debt_to_equity),    m.q5.debt_to_equity),
        ("current_ratio",     fmt_multiple(m.q5.current_ratio),     m.q5.current_ratio),
        ("quick_ratio",       fmt_multiple(m.q5.quick_ratio),       m.q5.quick_ratio),
        ("interest_coverage", fmt_multiple(m.q5.interest_coverage), m.q5.interest_coverage),
    ]
    q5_interp = build_interpretation_table_html(q5_metric_entries, sector)
    q5_technical_rows = _build_present_technical_rows(q5_ctx, q5_metric_entries)

    sections.append(_q_section(
        number=5,
        question="Is the balance sheet strong enough to survive shocks without hurting my portfolio, and has financial risk increased or decreased since I bought?",
        score=m.scores.get("Q5_BalanceSheet"),
        chart_html=chart_html,
        border_color=COLOR_OK,
        why_matters="A strong balance sheet is the difference between a company that can survive a recession and one that cannot. Excessive debt amplifies losses in downturns, limits strategic flexibility, and can force dilutive equity raises.",
        main_takeaway=_q5_takeaway(m),
        interpretation_table=q5_interp,
        context_items=q5_ctx,
        technical_rows=q5_technical_rows,
    ))

    # ── Q6 ─────────────────────────────────────────────────────────────────────
    q6_ctx = [
        ("Current Price", fmt_currency(m.q6.current_price)),
        ("DCF Intrinsic Value", fmt_currency(m.q6.intrinsic_value_dcf)),
    ]
    if m.mode == "existing":
        if m.q6.price_at_purchase is not None:
            q6_ctx.append(("Price at Purchase", fmt_currency(m.q6.price_at_purchase)))
        if m.q6.unrealized_gain_loss_pct is not None:
            q6_ctx.append(("Price Return Since Purchase", fmt_pct(m.q6.unrealized_gain_loss_pct)))
        q6_ctx.append(("Valuation vs. Purchase", m.q6.valuation_change or "N/A"))

    chart_html = ""
    if "q6_pe_gauge" in charts: chart_html += _fig_to_html(charts["q6_pe_gauge"])
    if "q6_ev_ebitda_gauge" in charts: chart_html += _fig_to_html(charts["q6_ev_ebitda_gauge"])
    if "q6_valuation_bars" in charts: chart_html += _fig_to_html(charts["q6_valuation_bars"])

    q6_metric_entries = [
        ("pe_ratio",         fmt_multiple(m.q6.pe_ratio),       m.q6.pe_ratio),
        ("forward_pe",       fmt_multiple(m.q6.forward_pe),     m.q6.forward_pe),
        ("peg_ratio",        fmt_multiple(m.q6.peg_ratio),      m.q6.peg_ratio),
        ("ev_ebitda",        fmt_multiple(m.q6.ev_ebitda),      m.q6.ev_ebitda),
        ("price_to_sales",   fmt_multiple(m.q6.price_to_sales), m.q6.price_to_sales),
        ("price_to_book",    fmt_multiple(m.q6.price_to_book),  m.q6.price_to_book),
        ("price_to_fcf",     fmt_multiple(m.q6.price_to_fcf),  m.q6.price_to_fcf),
        ("fcf_yield",        fmt_pct(m.q6.fcf_yield),           m.q6.fcf_yield),
        ("margin_of_safety", fmt_pct(m.q6.margin_of_safety),   m.q6.margin_of_safety),
    ]
    q6_interp = build_interpretation_table_html(q6_metric_entries, sector)
    q6_technical_rows = _build_present_technical_rows(q6_ctx, q6_metric_entries)

    sections.append(_q_section(
        number=6,
        question="Is the current price cheap enough to provide a margin of safety for my portfolio, and is the stock cheaper or more expensive than when I bought?",
        score=m.scores.get("Q6_Valuation"),
        chart_html=chart_html,
        border_color=COLOR_WEAK,
        why_matters="A good business bought at a bad price can still be a bad investment. Valuation determines your starting return, your margin of error if the business underperforms, and how much upside remains if the business executes well.",
        main_takeaway=_q6_takeaway(m),
        interpretation_table=q6_interp,
        context_items=q6_ctx,
        technical_rows=q6_technical_rows,
    ))

    # ── Q7 ─────────────────────────────────────────────────────────────────────
    q7_ctx = [
        ("Thesis Status",   _thesis_label(m.scores.get("Q7_Thesis"))),
        ("Growth Signal",   m.q2.growth_trend or "Unknown"),
        ("Margin Signal",   m.q3.margin_trend or "Unknown"),
        ("Balance Sheet",   m.q5.financial_risk_trend or "Unknown"),
        ("Valuation Risk",  m.q5.financial_risk_trend or "Unknown"),
        ("Revenue Momentum",    m.q2.growth_trend),
        ("ROIC Trend",          m.q3.roic_trend),
    ]
    if m.q4.share_dilution_pct is not None:
        sign = "+" if m.q4.share_dilution_pct >= 0 else ""
        q7_ctx.append(("Share Count Change", f"{sign}{m.q4.share_dilution_pct:.1%} YoY"))
    if m.q4.fcf_conversion_ratio is not None:
        q7_ctx.append(("FCF Conversion", f"{m.q4.fcf_conversion_ratio:.0%} of net income"))
    if m.mode == "existing":
        q7_ctx.append(("Valuation vs. Purchase", m.q6.valuation_change or "Unknown"))
        if m.portfolio.position_weight is not None:
            q7_ctx.append(("Portfolio Weight", fmt_pct(m.portfolio.position_weight)))

    sections.append(_q_section(
        number=7,
        question="Is the investment thesis still intact, and what would make me wrong?",
        score=m.scores.get("Q7_Thesis"),
        chart_html="",
        border_color="#8e44ad",
        why_matters=(
            "The investment thesis is the single reason you own or are considering buying a stock. "
            "This question synthesises Q2–Q6 to tell you whether that reason still holds — "
            "or whether something has changed that should make you reconsider."
        ),
        main_takeaway=_q7_takeaway(m),
        interpretation_table="",
        context_items=q7_ctx,
        technical_rows=[],
    ))

    return "\n".join(sections)


def _context_strip(items: list) -> str:
    """Compact key-value info strip for non-ratable descriptive fields."""
    if not items:
        return ""
    cells = "".join(
        f'<div style="display:inline-block;background:#f4f6f7;border-radius:6px;'
        f'padding:5px 12px;margin:3px 4px;font-size:12px">'
        f'<span style="color:#7f8c8d">{label}:</span> '
        f'<span style="color:#2c3e50;font-weight:600">{value}</span>'
        f'</div>'
        for label, value in items if value and value != "N/A" and value != "Unknown"
    )
    return f'<div style="margin:8px 0 4px">{cells}</div>' if cells else ""


def _q_section(
    number: int,
    question: str,
    score: Optional[float],
    chart_html: str = "",
    border_color: str = "#2980b9",
    why_matters: str = "",
    main_takeaway: str = "",
    interpretation_table: str = "",
    context_items: list = None,
    technical_rows: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    score_badge = _score_badge(score) if score is not None else ""

    why_html = (
        f'<div style="background:#f4f6f7;border-left:3px solid #aab7b8;'
        f'padding:8px 12px;margin:10px 0 6px;font-size:13px;color:#566573;'
        f'border-radius:0 4px 4px 0">'
        f'<strong>Why this matters:</strong> {why_matters}</div>'
    ) if why_matters else ""

    ctx_html = _context_strip(context_items or [])

    takeaway_html = ""
    if main_takeaway:
        if score is None:
            tk_color = COLOR_NEUTRAL
        else:
            tk_color = "#27ae60" if score >= 7 else "#e67e22" if score >= 4 else "#e74c3c"
        takeaway_html = (
            f'<div style="background:#fdfefe;border:1px solid {tk_color};'
            f'border-radius:6px;padding:8px 14px;margin:12px 0 0;font-size:13px;'
            f'color:#2c3e50">'
            f'<strong style="color:{tk_color}">Main takeaway:</strong> {main_takeaway}</div>'
        )

    guide_html = _build_interpretation_guide_details(interpretation_table) if interpretation_table else ""
    technical_html = _build_technical_metrics_details(list(technical_rows or []))

    # Chart sits to the right of context info if both exist; full-width otherwise
    top_row = ""
    if chart_html and ctx_html:
        top_row = f"""
        <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:16px;align-items:start;margin-top:4px">
          <div>{ctx_html}</div>
          <div>{chart_html}</div>
        </div>"""
    elif chart_html:
        top_row = f'<div style="margin-top:4px">{chart_html}</div>'
    elif ctx_html:
        top_row = f'<div style="margin-top:4px">{ctx_html}</div>'

    return f"""
    <div class="question-section" style="border-left:4px solid {border_color}">
      <div class="question-label">Question {number}</div>
      <div class="question-header">
        {question}
        &nbsp;{score_badge}
      </div>
      {why_html}
      {top_row}
      {guide_html}
      {technical_html}
      {takeaway_html}
    </div>"""


def _panel_to_html(panel: Panel, border: str, icon: str = "",
                   is_warning: bool = False, is_ml: bool = False) -> str:
    bullet_color = COLOR_WEAK if is_warning else COLOR_TEXT
    charts_html = "".join(_fig_to_html(fig) for fig in panel.charts)
    disclaimer = (
        '<p style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;'
        'padding:10px;font-size:12px;color:#856404;margin-top:12px">'
        '⚠️ ML PREDICTIONS: All figures in this panel are estimates based on historical patterns. '
        'They are NOT financial advice and are NOT guaranteed.</p>'
    ) if is_ml else ""
    bullets_html = _bullet_list(panel.bullets, color=bullet_color)
    return f"""
    <div style="background:{COLOR_CARD};border-left:4px solid {border};
                border-radius:8px;padding:20px 24px;margin:16px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <h2 style="margin:0 0 12px 0;color:{COLOR_TEXT};font-size:16px;font-weight:700">
        {icon} {panel.title}
      </h2>
      {disclaimer}
      {bullets_html}
      {charts_html}
    </div>"""


def _build_peer_section(primary: AllMetrics, peers: Dict[str, AllMetrics]) -> str:
    all_tickers = [primary.ticker] + list(peers.keys())
    all_metrics = {primary.ticker: primary, **peers}

    # Compare key metrics
    metric_map = {
        "P/E": lambda m: m.q6.pe_ratio,
        "EV/EBITDA": lambda m: m.q6.ev_ebitda,
        "Operating Margin": lambda m: m.q3.operating_margin,
        "FCF Yield": lambda m: m.q4.fcf_yield,
        "Debt/EBITDA": lambda m: m.q5.debt_to_ebitda,
        "Revenue Growth": lambda m: m.q2.revenue_1yr_cagr,
    }
    rows = ""
    for metric_name, getter in metric_map.items():
        vals = {t: getter(all_metrics[t]) for t in all_tickers}
        row_cells = "".join(
            f'<td style="padding:6px 12px;text-align:center;font-size:13px">'
            f'{"N/A" if v is None else (fmt_pct(v) if "Margin" in metric_name or "Yield" in metric_name or "Growth" in metric_name else fmt_multiple(v))}</td>'
            for v in vals.values()
        )
        rows += f'<tr><td style="padding:6px 12px;color:{COLOR_NEUTRAL};font-size:13px;font-weight:600">{metric_name}</td>{row_cells}</tr>'

    header_cells = "".join(
        f'<th style="padding:8px 12px;background:#ecf0f1;font-size:13px">{t}</th>'
        for t in all_tickers
    )
    table = f"""
    <table style="width:100%;border-collapse:collapse">
      <thead><tr><th style="padding:8px 12px;background:#ecf0f1;text-align:left;font-size:13px">Metric</th>{header_cells}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

    return _card("Peer Comparison", table, border_color=COLOR_ACCENT, icon="🔍")


# ── Section takeaway helpers ─────────────────────────────────────────────────

def _q1_takeaway(m: AllMetrics) -> str:
    cap = m.q1.cap_category or "Unknown"
    rev = fmt_currency(m.q1.ttm_revenue)
    sector = m.q1.sector or "Unknown"
    return (
        f"{m.company_name} is a {cap} company in {sector} with {rev} in trailing revenue. "
        f"{'Revenue has grown since purchase.' if m.mode == 'existing' and (m.q1.revenue_delta_pct or 0) > 0 else ''}"
    ).strip()


def _q2_takeaway(m: AllMetrics) -> str:
    trend = m.q2.growth_trend or "Stable"
    score = m.scores.get("Q2_Growth", 5)
    cagr = m.q2.revenue_1yr_cagr
    cagr_str = fmt_pct(cagr) if cagr is not None else "unknown"
    if score >= 7:
        return f"Growth is strong at {cagr_str} (1yr), with a {trend.lower()} trend. This is a key positive for the investment case."
    elif score >= 4:
        return f"Growth is moderate at {cagr_str} (1yr) with a {trend.lower()} trend. Adequate, but not a standout driver."
    else:
        return f"Growth is weak at {cagr_str} (1yr) and {trend.lower()}. Slowing growth is a risk that could compress the valuation multiple."


def _q3_takeaway(m: AllMetrics) -> str:
    score = m.scores.get("Q3_Profitability", 5)
    op_m = fmt_pct(m.q3.operating_margin)
    trend = m.q3.margin_trend or "Stable"
    if score >= 7:
        return f"Profitability is excellent with an operating margin of {op_m} and {trend.lower()} margins. Strong margins usually indicate a durable competitive advantage."
    elif score >= 4:
        return f"Profitability is acceptable with {op_m} operating margin. Margins are {trend.lower()}. Monitor for further compression."
    else:
        return f"Profitability is weak ({op_m} operating margin, {trend.lower()} trend). This is a quality problem that must be resolved before valuation matters."


def _q4_takeaway(m: AllMetrics) -> str:
    score = m.scores.get("Q4_CashFlow", 5)
    fcf_y = fmt_pct(m.q4.fcf_yield)
    fcf_m = fmt_pct(m.q4.fcf_margin)
    if score >= 7:
        return f"Cash flow is strong — FCF margin of {fcf_m} and FCF yield of {fcf_y} indicate the business is a reliable cash compounder."
    elif score >= 4:
        return f"Cash flow is adequate — FCF margin of {fcf_m} and FCF yield of {fcf_y}. Sufficient, but watch for capital-allocation quality."
    else:
        return f"Cash flow is weak — FCF margin of {fcf_m} and FCF yield of {fcf_y}. The business may need external capital to fund growth, which is a risk."


def _q5_takeaway(m: AllMetrics) -> str:
    score = m.scores.get("Q5_BalanceSheet", 5)
    d_e = fmt_multiple(m.q5.debt_to_ebitda)
    trend = m.q5.financial_risk_trend or "Stable"
    if score >= 7:
        return f"Balance sheet is strong — Debt/EBITDA of {d_e}, {trend.lower()} financial risk. The company has ample capacity to absorb shocks."
    elif score >= 4:
        return f"Balance sheet is adequate — Debt/EBITDA of {d_e} with {trend.lower()} financial risk. Manageable, but leaves limited buffer in a downturn."
    else:
        return f"Balance sheet is strained — Debt/EBITDA of {d_e} and {trend.lower()} financial risk. High leverage constrains strategic options and amplifies downside."


def _q6_takeaway(m: AllMetrics) -> str:
    score = m.scores.get("Q6_Valuation", 5)
    pe = fmt_multiple(m.q6.pe_ratio)
    mos = fmt_pct(m.q6.margin_of_safety)
    if score >= 7:
        return f"Valuation looks attractive — P/E of {pe} with a margin of safety of {mos} based on DCF. The price may not fully reflect the business quality."
    elif score >= 4:
        return f"Valuation is fair — P/E of {pe} with a margin of safety of {mos}. The price is reasonable but offers limited discount to intrinsic value."
    else:
        return f"Valuation is expensive — P/E of {pe} with a margin of safety of {mos}. A premium price means execution must be near-perfect. Risk/reward is skewed."


def _thesis_label(score: Optional[float]) -> str:
    """Map Q7 score to a plain-English thesis-health label."""
    if score is None:
        return "Thesis Unknown"
    if score >= 7.0:
        return "Thesis Intact"
    if score >= 5.0:
        return "Thesis Watch"
    if score >= 3.0:
        return "Thesis At Risk"
    return "Thesis Broken"


def _q7_takeaway(m: AllMetrics) -> str:
    """Generate a plain-English thesis-health summary answering three questions."""
    sc = m.scores
    pillar_names = {
        "Q2_Growth":        "growth",
        "Q3_Profitability": "profitability",
        "Q4_CashFlow":      "free cash flow",
        "Q5_BalanceSheet":  "balance sheet",
        "Q6_Valuation":     "valuation",
    }
    strong   = [pillar_names[k] for k in pillar_names if sc.get(k, 5) >= 7]
    watching = [pillar_names[k] for k in pillar_names if 4 <= sc.get(k, 5) < 7]
    weak     = [pillar_names[k] for k in pillar_names if sc.get(k, 5) < 4]

    # — 4 additional thesis signals
    if m.q2.growth_trend == "Decelerating":
        watching.append("revenue momentum (decelerating)")
    elif m.q2.growth_trend == "Accelerating" and "growth" not in strong:
        strong.append("revenue momentum (accelerating)")

    if m.q3.roic_trend == "Deteriorating":
        watching.append("return on capital (eroding)")
    elif m.q3.roic_trend == "Improving":
        strong.append("return on capital (improving)")

    if m.q4.share_dilution_pct is not None:
        if m.q4.share_dilution_pct > 0.03:
            watching.append(f"share dilution (+{m.q4.share_dilution_pct:.1%} shares YoY)")
        elif m.q4.share_dilution_pct > 0.01:
            watching.append(f"mild share dilution (+{m.q4.share_dilution_pct:.1%} YoY)")
        elif m.q4.share_dilution_pct < -0.02:
            strong.append(f"buybacks reducing share count ({m.q4.share_dilution_pct:.1%} YoY)")

    if m.q4.fcf_conversion_ratio is not None:
        if m.q4.fcf_conversion_ratio < 0.5:
            weak.append(f"earnings quality (FCF only {m.q4.fcf_conversion_ratio:.0%} of net income)")
        elif m.q4.fcf_conversion_ratio < 0.7:
            watching.append(f"earnings quality (FCF {m.q4.fcf_conversion_ratio:.0%} of net income — monitor)")

    # — Existing-holding adjustments
    if m.mode == "existing":
        if m.q6.valuation_change in ("Much More Expensive", "More Expensive"):
            watching.append("valuation (higher than at purchase)")
        if m.portfolio.position_weight is not None and m.portfolio.position_weight > 0.15:
            watching.append("portfolio concentration")

    label = _thesis_label(sc.get("Q7_Thesis"))

    supports_str = (
        f"{', '.join(strong).capitalize()} {'are' if len(strong) > 1 else 'is'} scoring well."
        if strong else "No individual pillar is currently in the top tier."
    )
    watching_str = (
        f"Keep an eye on {', '.join(watching)}."
        if watching else "No areas of immediate concern."
    )
    change_mind_str = (
        f"The thesis would come under real pressure if {' and '.join(weak)} deteriorate further."
        if weak else (
            "Watch for growth deceleration, margin compression, ROIC erosion, or share dilution "
            "accelerating — any of these could shift the risk/reward unfavourably."
        )
    )

    return (
        f"<strong>Thesis status: {label}.</strong><br>"
        f"<strong>What supports the thesis:</strong> {supports_str}<br>"
        f"<strong>What needs watching:</strong> {watching_str}<br>"
        f"<strong>What would change my mind:</strong> {change_mind_str}"
    )


# ── CEO Decision Summary ─────────────────────────────────────────────────────

def _build_ceo_decision_summary(
    m: AllMetrics,
    recommendation: str,
    recommendation_explanation: str,
) -> str:
    """Generate a plain-English CEO-style decision summary."""
    scores = m.scores
    sector = m.q1.sector or "Unknown"
    ticker = m.ticker
    name = m.company_name

    # Identify strong and weak pillars
    strong = [k for k, v in scores.items() if v >= 7]
    weak = [k for k, v in scores.items() if v < 4]
    neutral = [k for k, v in scores.items() if 4 <= v < 7]

    def _label(key: str) -> str:
        labels = {
            "Q1_Scale": "scale",
            "Q2_Growth": "growth",
            "Q3_Profitability": "profitability",
            "Q4_CashFlow": "cash flow",
            "Q5_BalanceSheet": "balance sheet strength",
            "Q6_Valuation": "valuation",
            "Q7_Thesis": "thesis health",
        }
        return labels.get(key, key)

    strong_str = ", ".join(_label(k) for k in strong) if strong else "none standing out"
    weak_str   = ", ".join(_label(k) for k in weak)   if weak   else "none"

    # What type of problem is it?
    problem_type = []
    if "Q2_Growth" in weak:       problem_type.append("a growth problem")
    if "Q3_Profitability" in weak: problem_type.append("a profitability problem")
    if "Q5_BalanceSheet" in weak:  problem_type.append("a balance sheet problem")
    if "Q6_Valuation" in weak:     problem_type.append("a valuation problem")
    if not problem_type:
        problem_type_str = "no major structural problem — the main debate is price vs. quality."
    else:
        problem_type_str = " and ".join(problem_type) + "."

    # Valuation context
    val_score = scores.get("Q6_Valuation", 5)
    if val_score >= 7:
        priced_in = "The current price may not be fully pricing in the business quality — potential upside exists."
        buy_trigger = "The current price already offers a reasonable entry point relative to fundamentals."
        wait_trigger = "Patience is not required — valuation is already attractive."
        sell_trigger = "If the business fundamentals deteriorate significantly without a price correction."
    elif val_score >= 4:
        priced_in = "The market appears to have priced in steady performance. Little margin of error at current levels."
        buy_trigger = f"A pullback to more attractive valuation levels or a fundamental catalyst (margin expansion, accelerating growth) would improve the risk/reward."
        wait_trigger = "Consider waiting for a better entry price if no time pressure exists."
        sell_trigger = "If growth decelerates materially or margins contract, the multiple could compress and compound the loss."
    else:
        priced_in = "Strong future performance is already priced in. The stock is expensive relative to current fundamentals."
        buy_trigger = "Only if you have very high conviction in sustained above-consensus growth and margin expansion."
        wait_trigger = "Waiting for a significant price correction or re-rating event is prudent."
        sell_trigger = "If growth disappoints or competition intensifies — a multiple de-rating from expensive levels can be severe."

    # Main buy / avoid logic
    overall = m.overall_score
    if overall >= 7.5:
        avoid_str = "The main risk is not business quality but price. Avoid if you require a large margin of safety or are near-term oriented."
    elif overall >= 5.5:
        avoid_str = "Avoid if any of the weak areas deteriorate further or if valuation expands without improving fundamentals."
    else:
        avoid_str = f"Avoid until the core problem ({problem_type_str[:-1]}) is resolved. Do not pay a fair price for a below-average business."

    # Compose sections
    what_strong = f"<b>What looks strong:</b> {name} scores well on {strong_str}." if strong else f"<b>What looks strong:</b> No area scores in the top tier currently."
    what_weak   = f"<b>What looks weak:</b> The main concerns are {weak_str}." if weak else "<b>What looks weak:</b> No major weaknesses identified."
    priced_html = f"<b>What is priced in:</b> {priced_in}"
    buy_html    = f"<b>What would make me buy / add:</b> {buy_trigger}"
    wait_html   = f"<b>What would make me wait:</b> {wait_trigger}"
    sell_html   = f"<b>What would make me sell or avoid:</b> {sell_trigger} {avoid_str}"
    problem_html = f"<b>Root cause:</b> This is primarily {problem_type_str}"

    items = [what_strong, what_weak, priced_html, buy_html, wait_html, sell_html, problem_html]
    items_html = "".join(
        f'<li style="margin:10px 0;font-size:14px;color:#2c3e50;line-height:1.6">{item}</li>'
        for item in items
    )

    rec_style = REC_STYLES.get(recommendation, REC_STYLES["Hold"])
    summary_score_display = f'<span style="font-size:28px;font-weight:900;color:{rec_style["text"]}">{m.overall_score:.1f}/10</span>'

    return f"""
    <div style="background:#ffffff;border:2px solid {rec_style['border']};border-radius:12px;
                padding:24px 28px;margin:24px 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.08)">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  flex-wrap:wrap;gap:12px;margin-bottom:16px">
        <div>
          <div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;
                      color:#7f8c8d;margin-bottom:4px">CEO Decision Summary</div>
          <div style="font-size:20px;font-weight:800;color:#2c3e50">
            {ticker} — {name}
          </div>
          <div style="font-size:13px;color:#566573;margin-top:2px">{sector}</div>
        </div>
        <div style="text-align:right">
          {summary_score_display}
          <div style="font-size:12px;color:#7f8c8d">Overall Score</div>
          <div style="margin-top:6px">
            <span style="background:{rec_style['bg']};color:{rec_style['text']};
                         border:1px solid {rec_style['border']};padding:4px 14px;
                         border-radius:16px;font-size:14px;font-weight:700">
              {recommendation}
            </span>
          </div>
        </div>
      </div>
      <div style="font-size:14px;color:#566573;line-height:1.7;
                  border-bottom:1px solid #ecf0f1;padding-bottom:14px;margin-bottom:14px">
        {recommendation_explanation}
      </div>
      <ul style="list-style:none;padding:0;margin:0">
        {items_html}
      </ul>
    </div>"""


# ── Delta helpers (display strings) ─────────────────────────────────────────

def _delta_str(pct: Optional[float]) -> str:
    if pct is None: return ""
    arrow = "▲" if pct >= 0 else "▼"
    color = COLOR_STRONG if pct >= 0 else COLOR_WEAK
    return f'<span style="color:{color}">{arrow} {fmt_pct(abs(pct))}</span>'


def _margin_delta(now: Optional[float], prior: Optional[float]) -> str:
    if now is None or prior is None: return ""
    delta = (now or 0) - (prior or 0)
    return _delta_str(delta)


def _fcf_delta(now: Optional[float], at_buy: Optional[float]) -> str:
    if now is None or at_buy is None: return ""
    pct = (now - at_buy) / abs(at_buy) if at_buy != 0 else None
    return _delta_str(pct)


def _debt_delta(now: Optional[float], at_buy: Optional[float]) -> str:
    """For debt, lower is better — flip the arrow."""
    if now is None or at_buy is None: return ""
    delta = (now or 0) - (at_buy or 0)
    arrow = "▲" if delta >= 0 else "▼"
    color = COLOR_WEAK if delta > 0 else COLOR_STRONG  # more debt = worse
    return f'<span style="color:{color}">{arrow} {abs(delta):.1f}x vs. purchase</span>'


def _pe_delta(now: Optional[float], at_buy: Optional[float]) -> str:
    if now is None or at_buy is None: return ""
    delta = (now or 0) - (at_buy or 0)
    arrow = "▲" if delta >= 0 else "▼"
    color = COLOR_WEAK if delta > 0 else COLOR_STRONG  # higher P/E = more expensive
    return f'<span style="color:{color}">{arrow} {abs(delta):.1f}x vs. purchase</span>'


def _evev_delta(now: Optional[float], at_buy: Optional[float]) -> str:
    return _pe_delta(now, at_buy)



def _build_master_html(
    new_metrics: AllMetrics,
    new_panels: Dict[str, Panel],
    new_recommendation: str,
    new_recommendation_explanation: str,
    new_charts: Dict[str, go.Figure],
    existing_metrics: AllMetrics,
    existing_panels: Dict[str, Panel],
    existing_recommendation: str,
    existing_recommendation_explanation: str,
    existing_charts: Dict[str, go.Figure],
    first_principles_report: Optional["FirstPrinciplesReport"] = None,
    notes: str = "",
    peer_metrics: Optional[Dict[str, AllMetrics]] = None,
) -> str:
    """Build the master HTML combining both ownership modes × all three timeline views."""
    m = new_metrics  # ticker/company name are the same for both modes
    gen_time = datetime.now().strftime("%B %d, %Y %H:%M")
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>'

    # FP decisions for overview table
    _, past_decision, _, _, _ = _compute_fp_questions_state(first_principles_report, "past", "Monitor")
    _, future_decision, _, _, _ = _compute_fp_questions_state(first_principles_report, "future", "Monitor")

    # FP explanation string for rec_box in FP panels
    fp_explanation = ""
    if first_principles_report is not None:
        s = first_principles_report.synthesis
        fp_explanation = (
            f"First-principles signal mix: Strong {s.strong_count}, "
            f"Watch {s.watch_count}, Weak {s.weak_count}, "
            f"Insufficient {s.insufficient_count}."
        )

    # ── Master Header ─────────────────────────────────────────────────────────
    master_header = f"""
    <div style="background:linear-gradient(135deg,#1a252f,#2c3e50);
                padding:32px 40px;color:white;border-radius:0 0 12px 12px;margin-bottom:0">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:13px;text-transform:uppercase;letter-spacing:2px;opacity:0.7;margin-bottom:4px">
            Master CEO Report
          </div>
          <div style="font-size:42px;font-weight:900;letter-spacing:2px">{escape(m.ticker)}</div>
          <div style="font-size:18px;opacity:0.85;margin-top:2px">{escape(m.company_name)}</div>
          <div style="font-size:12px;opacity:0.6;margin-top:8px">{gen_time}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;opacity:0.6">{escape(m.staleness_label)}</div>
        </div>
      </div>
    </div>"""

    # ── Decision badge helper (inline) ────────────────────────────────────────
    def _rec_badge_mini(rec: str) -> str:
        s = REC_STYLES.get(rec, REC_STYLES["Hold"])
        return (
            f'<span style="background:{s["bg"]};color:{s["text"]};'
            f'border:1px solid {s["border"]};padding:3px 10px;border-radius:12px;'
            f'font-size:12px;font-weight:700;white-space:nowrap">{escape(rec)}</span>'
        )

    # ── Decision Overview Table ───────────────────────────────────────────────
    overview_section = f"""
    <div style="max-width:1100px;margin:0 auto;padding:0 20px">
      <div style="background:#ffffff;border-radius:8px;padding:20px 24px;margin:16px 0;
                  box-shadow:0 1px 4px rgba(0,0,0,0.08);overflow-x:auto">
        <h3 style="margin:0 0 14px 0;color:{COLOR_TEXT};font-size:15px;font-weight:700">
          📋 Decision Overview
        </h3>
        <table style="width:100%;border-collapse:collapse;min-width:420px">
          <thead>
            <tr style="background:#f4f6f7">
              <th style="padding:10px 14px;text-align:left;font-size:13px;color:{COLOR_NEUTRAL}">Mode</th>
              <th style="padding:10px 14px;text-align:center;font-size:13px;color:{COLOR_NEUTRAL}">Present</th>
              <th style="padding:10px 14px;text-align:center;font-size:13px;color:{COLOR_NEUTRAL}">Past</th>
              <th style="padding:10px 14px;text-align:center;font-size:13px;color:{COLOR_NEUTRAL}">Future</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #ecf0f1">
              <td style="padding:10px 14px;font-size:13px;font-weight:600;color:{COLOR_TEXT}">New Position</td>
              <td style="padding:10px 14px;text-align:center">{_rec_badge_mini(new_recommendation)}</td>
              <td style="padding:10px 14px;text-align:center">{_rec_badge_mini(past_decision)}</td>
              <td style="padding:10px 14px;text-align:center">{_rec_badge_mini(future_decision)}</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;font-size:13px;font-weight:600;color:{COLOR_TEXT}">Existing Holding</td>
              <td style="padding:10px 14px;text-align:center">{_rec_badge_mini(existing_recommendation)}</td>
              <td style="padding:10px 14px;text-align:center">{_rec_badge_mini(past_decision)}</td>
              <td style="padding:10px 14px;text-align:center">{_rec_badge_mini(future_decision)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>"""

    # ── Sticky Controls ───────────────────────────────────────────────────────
    _btn = (
        f"padding:6px 14px;border:1px solid #ddd;border-radius:6px;cursor:pointer;"
        f"font-size:13px;font-weight:600;background:#f9fafb;color:{COLOR_TEXT};"
        f"transition:background 0.15s,color 0.15s,border-color 0.15s"
    )
    controls = f"""
    <div id="master-controls" style="background:#ffffff;border-bottom:2px solid #ecf0f1;
                padding:12px 20px;position:sticky;top:0;z-index:100;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
      <div style="max-width:1100px;margin:0 auto;display:flex;gap:24px;align-items:center;flex-wrap:wrap">
        <div style="display:flex;align-items:center;gap:8px">
          <span style="font-size:12px;font-weight:600;color:{COLOR_NEUTRAL};text-transform:uppercase;letter-spacing:0.8px">Ownership</span>
          <div id="ownership-sel" style="display:flex;gap:4px">
            <button class="sel-btn" data-group="ownership" data-val="existing" onclick="setOwnership('existing')" style="{_btn}">Existing Holding</button>
            <button class="sel-btn" data-group="ownership" data-val="new" onclick="setOwnership('new')" style="{_btn}">New Position</button>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <span style="font-size:12px;font-weight:600;color:{COLOR_NEUTRAL};text-transform:uppercase;letter-spacing:0.8px">Timeline</span>
          <div id="timeline-sel" style="display:flex;gap:4px">
            <button class="sel-btn" data-group="timeline" data-val="present" onclick="setTimeline('present')" style="{_btn}">Present</button>
            <button class="sel-btn" data-group="timeline" data-val="past" onclick="setTimeline('past')" style="{_btn}">Past</button>
            <button class="sel-btn" data-group="timeline" data-val="future" onclick="setTimeline('future')" style="{_btn}">Future</button>
          </div>
        </div>
      </div>
    </div>"""

    # ── Build inner content for all 6 panels ─────────────────────────────────
    fp_rec = first_principles_report.synthesis.decision if first_principles_report is not None else "Monitor"

    new_present_inner = _present_inner_content(
        new_metrics, new_panels, new_recommendation, new_recommendation_explanation,
        new_charts, peer_metrics, notes, gen_time,
    )
    existing_present_inner = _present_inner_content(
        existing_metrics, existing_panels, existing_recommendation, existing_recommendation_explanation,
        existing_charts, peer_metrics, notes, gen_time,
    )
    new_past_inner = _fp_inner_content(
        new_metrics, fp_rec, fp_explanation, notes, first_principles_report, "past", gen_time,
    )
    new_future_inner = _fp_inner_content(
        new_metrics, fp_rec, fp_explanation, notes, first_principles_report, "future", gen_time,
    )
    existing_past_inner = _fp_inner_content(
        existing_metrics, fp_rec, fp_explanation, notes, first_principles_report, "past", gen_time,
    )
    existing_future_inner = _fp_inner_content(
        existing_metrics, fp_rec, fp_explanation, notes, first_principles_report, "future", gen_time,
    )

    def _present_panel(panel_id: str, metrics_obj: AllMetrics, inner: str) -> str:
        dq = _data_quality_banner(metrics_obj)
        return (
            f'<div id="{panel_id}" class="master-panel" style="display:none">'
            f'{dq}'
            f'<div style="max-width:1100px;margin:0 auto;padding:0 20px">'
            f'{inner}'
            f'</div>'
            f'</div>'
        )

    def _fp_panel(panel_id: str, inner: str) -> str:
        return (
            f'<div id="{panel_id}" class="master-panel" style="display:none">'
            f'<div style="max-width:1100px;margin:0 auto;padding:0 20px">'
            f'{inner}'
            f'</div>'
            f'</div>'
        )

    panels_html = "\n".join([
        _present_panel("panel-new-present", new_metrics, new_present_inner),
        _fp_panel("panel-new-past", new_past_inner),
        _fp_panel("panel-new-future", new_future_inner),
        _present_panel("panel-existing-present", existing_metrics, existing_present_inner),
        _fp_panel("panel-existing-past", existing_past_inner),
        _fp_panel("panel-existing-future", existing_future_inner),
    ])

    # ── Inline JavaScript ─────────────────────────────────────────────────────
    js = """
    <script>
    var _masterOwnership = 'existing';
    var _masterTimeline = 'present';

    function showMasterPanel(ownership, timeline) {
        document.querySelectorAll('.master-panel').forEach(function(el) {
            el.style.display = 'none';
        });
        var panel = document.getElementById('panel-' + ownership + '-' + timeline);
        if (panel) panel.style.display = 'block';

        document.querySelectorAll('#ownership-sel .sel-btn').forEach(function(btn) {
            var active = btn.getAttribute('data-val') === ownership;
            btn.style.background = active ? '#2980b9' : '#f9fafb';
            btn.style.color = active ? 'white' : '#2c3e50';
            btn.style.borderColor = active ? '#2980b9' : '#ddd';
        });
        document.querySelectorAll('#timeline-sel .sel-btn').forEach(function(btn) {
            var active = btn.getAttribute('data-val') === timeline;
            btn.style.background = active ? '#2980b9' : '#f9fafb';
            btn.style.color = active ? 'white' : '#2c3e50';
            btn.style.borderColor = active ? '#2980b9' : '#ddd';
        });
        _masterOwnership = ownership;
        _masterTimeline = timeline;
    }

    function setOwnership(ownership) { showMasterPanel(ownership, _masterTimeline); }
    function setTimeline(timeline) { showMasterPanel(_masterOwnership, timeline); }

    showMasterPanel('existing', 'present');
    </script>"""

    body = "\n".join([
        master_header,
        overview_section,
        controls,
        panels_html,
        js,
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Master CEO Report — {escape(m.ticker)} — {datetime.now().strftime('%Y-%m-%d')}</title>
  {plotly_cdn}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      background: {COLOR_BG};
      color: {COLOR_TEXT};
      margin: 0;
      padding: 0;
      line-height: 1.5;
    }}
    table {{ border-collapse: collapse; }}
    @media (max-width: 700px) {{
      div[style*="grid-template-columns:1fr 1fr"] {{
        display: block !important;
      }}
      #master-controls > div > div {{
        flex-wrap: wrap;
      }}
    }}
    .question-section {{
      background: {COLOR_CARD};
      border-radius: 8px;
      padding: 20px 24px;
      margin: 16px 0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }}
    .question-header {{
      font-size: 16px;
      font-weight: 700;
      color: {COLOR_TEXT};
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 2px solid #ecf0f1;
    }}
    .question-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {COLOR_NEUTRAL};
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>"""