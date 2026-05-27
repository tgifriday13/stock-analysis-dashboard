"""
visualizations.py — Plotly chart factory.

Intelligently selects the best visual for each metric type.
Every chart includes a timestamp watermark and a plain-English label.
Charts are returned as Plotly Figure objects — call .show() to display inline,
or .to_html() for embedding in the CEO report.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .metrics_calculator import AllMetrics
from .threshold_manager import ThresholdManager
from .utils import fmt_currency, fmt_pct, fmt_multiple, fmt_number

logger = logging.getLogger("dashboard.visualizations")

# ── Brand palette ─────────────────────────────────────────────────────────────
COLOR_STRONG  = "#27ae60"   # green
COLOR_OK      = "#f39c12"   # amber
COLOR_WEAK    = "#e74c3c"   # red
COLOR_NEUTRAL = "#7f8c8d"   # grey
COLOR_BG      = "#ffffff"
COLOR_GRID    = "#ecf0f1"
COLOR_TEXT    = "#2c3e50"
COLOR_ACCENT  = "#2980b9"   # blue

FONT_FAMILY   = "Inter, Arial, sans-serif"


def _watermark(label: str) -> dict:
    """Return a layout annotation used as a watermark / caption."""
    return dict(
        text=label,
        xref="paper", yref="paper",
        x=0.5, y=-0.14,
        showarrow=False,
        font=dict(size=10, color=COLOR_NEUTRAL, family=FONT_FAMILY),
        xanchor="center",
    )


def _base_layout(title: str, watermark_text: str = "", height: int = 320) -> dict:
    annots = [_watermark(watermark_text)] if watermark_text else []
    return dict(
        title=dict(text=title, font=dict(size=14, color=COLOR_TEXT, family=FONT_FAMILY)),
        height=height,
        margin=dict(l=40, r=20, t=50, b=60),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family=FONT_FAMILY, color=COLOR_TEXT),
        annotations=annots,
    )


def _metric_color(value: Optional[float], strong: float, acceptable: float,
                  higher_is_better: bool = True) -> str:
    if value is None:
        return COLOR_NEUTRAL
    if higher_is_better:
        if value >= strong:
            return COLOR_STRONG
        elif value >= acceptable:
            return COLOR_OK
        return COLOR_WEAK
    else:
        if value <= strong:
            return COLOR_STRONG
        elif value <= acceptable:
            return COLOR_OK
        return COLOR_WEAK


# ── Gauge chart ───────────────────────────────────────────────────────────────

def gauge_chart(
    value: Optional[float],
    label: str,
    min_val: float,
    max_val: float,
    strong_threshold: float,
    acceptable_threshold: float,
    format_fn=None,
    higher_is_better: bool = True,
    watermark: str = "",
    title: str = "",
) -> go.Figure:
    """
    Gauge indicator chart. Used for single KPI metrics like P/E, FCF yield, margins.
    """
    if format_fn is None:
        format_fn = lambda v: f"{v:.1f}"

    display_val = value if value is not None else 0
    display_str = format_fn(value) if value is not None else "N/A"

    # Color bands
    if higher_is_better:
        steps = [
            {"range": [min_val, acceptable_threshold], "color": "#fdecea"},
            {"range": [acceptable_threshold, strong_threshold], "color": "#fef9e7"},
            {"range": [strong_threshold, max_val], "color": "#eafaf1"},
        ]
    else:
        steps = [
            {"range": [min_val, strong_threshold], "color": "#eafaf1"},
            {"range": [strong_threshold, acceptable_threshold], "color": "#fef9e7"},
            {"range": [acceptable_threshold, max_val], "color": "#fdecea"},
        ]

    bar_color = _metric_color(value, strong_threshold, acceptable_threshold, higher_is_better)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        number={"valueformat": ".2f", "font": {"size": 20, "color": bar_color}},
        title={"text": label, "font": {"size": 13, "color": COLOR_TEXT}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickfont": {"size": 10}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": steps,
            "threshold": {
                "line": {"color": COLOR_STRONG, "width": 2},
                "thickness": 0.75,
                "value": strong_threshold,
            },
        },
    ))
    fig.update_layout(**_base_layout(title or label, watermark, height=260))
    return fig


# ── Bar chart (single or grouped) ────────────────────────────────────────────

def bar_chart(
    labels: List[str],
    values: List[Optional[float]],
    title: str,
    y_label: str = "",
    format_fn=None,
    colors: Optional[List[str]] = None,
    watermark: str = "",
    height: int = 320,
    compare_values: Optional[List[Optional[float]]] = None,
    compare_label: str = "At Purchase",
    current_label: str = "Current",
) -> go.Figure:
    """
    Horizontal or vertical bar chart. Optionally shows side-by-side 'now vs. at purchase'.
    """
    if format_fn is None:
        format_fn = lambda v: f"{v:.2f}" if v is not None else "N/A"

    clean_values = [v if v is not None else 0 for v in values]
    bar_colors = colors or [COLOR_ACCENT] * len(labels)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=current_label,
        x=labels,
        y=clean_values,
        marker_color=bar_colors,
        text=[format_fn(v) for v in values],
        textposition="outside",
    ))

    if compare_values:
        clean_compare = [v if v is not None else 0 for v in compare_values]
        fig.add_trace(go.Bar(
            name=compare_label,
            x=labels,
            y=clean_compare,
            marker_color=COLOR_NEUTRAL,
            opacity=0.6,
            text=[format_fn(v) for v in compare_values],
            textposition="outside",
        ))
        fig.update_layout(barmode="group")

    fig.update_layout(
        **_base_layout(title, watermark, height=height),
        yaxis_title=y_label,
        showlegend=bool(compare_values),
    )
    fig.update_yaxes(gridcolor=COLOR_GRID)
    return fig


# ── Line chart ────────────────────────────────────────────────────────────────

def line_chart(
    x: List,
    y_series: Dict[str, List],
    title: str,
    y_label: str = "",
    format_fn=None,
    watermark: str = "",
    height: int = 320,
    vline_x=None,
    vline_label: str = "Purchase Date",
    color_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Multi-series line chart. Used for price history, revenue trends, margin trends.
    vline_x: draw a vertical reference line at this x-value (e.g., purchase date).
    """
    fig = go.Figure()
    default_colors = [COLOR_ACCENT, COLOR_STRONG, COLOR_OK, COLOR_WEAK, COLOR_NEUTRAL]
    for idx, (name, vals) in enumerate(y_series.items()):
        color = (color_map or {}).get(name, default_colors[idx % len(default_colors)])
        fig.add_trace(go.Scatter(
            x=x,
            y=vals,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        ))

    if vline_x is not None:
        fig.add_vline(
            x=vline_x,
            line_dash="dot",
            line_color=COLOR_OK,
            annotation_text=vline_label,
            annotation_position="top right",
            annotation_font_size=10,
        )

    fig.update_layout(
        **_base_layout(title, watermark, height=height),
        yaxis_title=y_label,
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor=COLOR_GRID)
    fig.update_xaxes(gridcolor=COLOR_GRID)
    return fig


# ── Sparkline ─────────────────────────────────────────────────────────────────

def sparkline(
    values: List[Optional[float]],
    title: str,
    color: str = COLOR_ACCENT,
    watermark: str = "",
) -> go.Figure:
    """
    Minimal sparkline — small, label-only trend chart.
    """
    clean = [v if v is not None else float("nan") for v in values]
    fig = go.Figure(go.Scatter(
        y=clean,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba{tuple(list(bytes.fromhex(color.lstrip('#'))) + [30])}",
    ))
    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        title=dict(text=title, font=dict(size=10, color=COLOR_NEUTRAL)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ── Radar / spider chart ──────────────────────────────────────────────────────

def radar_chart(
    scores: Dict[str, float],
    title: str = "Fundamental Scorecard",
    watermark: str = "",
) -> go.Figure:
    """
    Radar chart showing scores across the 6 fundamental questions.
    """
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba(41,128,185,0.2)",
        line=dict(color=COLOR_ACCENT, width=2),
        name="Score",
    ))
    # Add a reference line at 5 (baseline)
    fig.add_trace(go.Scatterpolar(
        r=[5] * (len(labels) + 1),
        theta=labels_closed,
        mode="lines",
        line=dict(color=COLOR_NEUTRAL, dash="dot", width=1),
        name="Baseline (5)",
        showlegend=False,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=COLOR_GRID),
            angularaxis=dict(gridcolor=COLOR_GRID),
        ),
        **_base_layout(title, watermark, height=360),
        showlegend=False,
    )
    return fig


# ── Waterfall chart ───────────────────────────────────────────────────────────

def waterfall_chart(
    labels: List[str],
    values: List[float],
    title: str,
    watermark: str = "",
) -> go.Figure:
    """
    Waterfall / bridge chart. Used for showing FCF build-up or return attribution.
    """
    measures = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line": {"color": COLOR_NEUTRAL}},
        increasing={"marker": {"color": COLOR_STRONG}},
        decreasing={"marker": {"color": COLOR_WEAK}},
        totals={"marker": {"color": COLOR_ACCENT}},
    ))
    fig.update_layout(**_base_layout(title, watermark))
    return fig


# ── Dividend history bar chart ────────────────────────────────────────────────

def dividend_history_chart(
    dividends: pd.Series,
    title: str = "Annual Dividends Per Share",
    purchase_date=None,
    watermark: str = "",
) -> go.Figure:
    """
    Annual dividend bar chart with purchase-date vline.
    """
    if dividends is None or dividends.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title, watermark))
        fig.add_annotation(
            text="No dividend history available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=COLOR_NEUTRAL),
        )
        return fig

    annual = dividends.resample("YE").sum()
    annual = annual[annual > 0]
    if annual.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title, watermark))
        return fig

    vals = list(annual.values)
    colors = [COLOR_STRONG] * len(vals)

    fig = go.Figure(go.Bar(
        x=annual.index, y=vals,
        marker_color=colors,
        text=[f"${v:.2f}" for v in vals],
        textposition="outside",
    ))
    if purchase_date:
        purchase_ts = pd.Timestamp(purchase_date).normalize()
        y_top = float(max(vals)) * 1.08 if vals else 1.0
        fig.add_trace(go.Scatter(
            x=[purchase_ts, purchase_ts],
            y=[0, y_top],
            mode="lines",
            line=dict(color=COLOR_OK, dash="dot", width=1.2),
            name="Purchase Date",
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_annotation(
            x=purchase_ts,
            y=y_top,
            text="Bought",
            showarrow=False,
            font=dict(size=10, color=COLOR_OK),
            yshift=4,
        )
    fig.update_layout(**_base_layout(title, watermark))
    fig.update_yaxes(title_text="$ / Share", gridcolor=COLOR_GRID)
    fig.update_xaxes(tickformat="%Y")
    return fig


# ── Price history chart ───────────────────────────────────────────────────────

def price_history_chart(
    history: pd.DataFrame,
    ticker: str,
    purchase_date=None,
    cost_basis: Optional[float] = None,
    watermark: str = "",
) -> go.Figure:
    """
    5-year price chart with optional purchase date and cost basis lines.
    """
    if history is None or history.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(f"{ticker} — Price History", watermark))
        return fig

    hist = history.copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color=COLOR_ACCENT, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(41,128,185,0.06)",
    ))

    if purchase_date:
        fig.add_vline(
            x=str(pd.Timestamp(purchase_date).date()),
            line_dash="dot", line_color=COLOR_OK,
            annotation_text="Bought", annotation_position="top right",
            annotation_font_size=11,
        )
    if cost_basis:
        fig.add_hline(
            y=cost_basis,
            line_dash="dash", line_color=COLOR_OK,
            annotation_text=f"Cost Basis ${cost_basis:.2f}",
            annotation_position="right",
            annotation_font_size=10,
        )

    fig.update_layout(
        **_base_layout(f"{ticker} — 5-Year Price History", watermark, height=340),
        yaxis_title="Price (USD)",
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor=COLOR_GRID)
    return fig


# ── Risk / score slider ───────────────────────────────────────────────────────

def score_bar_chart(
    scores: Dict[str, float],
    title: str = "Question Scores (0–10)",
    watermark: str = "",
) -> go.Figure:
    """
    Horizontal bar chart of the 6 question scores, color-coded green/amber/red.
    """
    labels = list(scores.keys())
    values = list(scores.values())
    colors = [
        COLOR_STRONG if v >= 7 else COLOR_OK if v >= 4 else COLOR_WEAK
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(title, watermark, height=280),
        xaxis=dict(range=[0, 10], gridcolor=COLOR_GRID),
    )
    # Add reference line at 5
    fig.add_vline(x=5, line_dash="dot", line_color=COLOR_NEUTRAL, opacity=0.6)
    return fig


# ── Multi-metric comparison (peers) ──────────────────────────────────────────

def peer_comparison_chart(
    tickers: List[str],
    metric_values: Dict[str, List[Optional[float]]],
    metric_label: str,
    title: str = "",
    format_fn=None,
    watermark: str = "",
) -> go.Figure:
    """
    Grouped bar chart comparing a metric across multiple tickers.
    """
    if format_fn is None:
        format_fn = lambda v: f"{v:.2f}" if v is not None else "N/A"

    fig = go.Figure()
    colors = [COLOR_ACCENT, COLOR_STRONG, COLOR_OK, COLOR_WEAK, COLOR_NEUTRAL]
    for idx, (metric, vals) in enumerate(metric_values.items()):
        clean = [v if v is not None else 0 for v in vals]
        fig.add_trace(go.Bar(
            name=metric,
            x=tickers,
            y=clean,
            marker_color=colors[idx % len(colors)],
            text=[format_fn(v) for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        **_base_layout(title or f"Peer Comparison — {metric_label}", watermark),
        barmode="group",
        yaxis_title=metric_label,
    )
    return fig


# ── Convenience function: build all charts for a metric set ──────────────────

def build_all_charts(
    metrics: AllMetrics,
    tm: ThresholdManager,
) -> Dict[str, go.Figure]:
    """
    Build the full set of charts for a given AllMetrics result.
    Returns a dict keyed by chart name.
    """
    wm = metrics.staleness_label
    mode = metrics.mode
    purchase_date = metrics.portfolio.purchase_date if mode == "existing" else None
    cost_basis = metrics.portfolio.cost_basis if mode == "existing" else None
    charts = {}

    # ── Q1 ─────────────────────────────────────────────────────────────────────
    q1 = metrics.q1
    # Revenue bar (now vs. at purchase)
    rev_labels = ["TTM Revenue"]
    rev_vals = [q1.ttm_revenue]
    rev_compare = [q1.ttm_revenue_prior] if mode == "existing" else None
    charts["q1_revenue"] = bar_chart(
        rev_labels, rev_vals,
        title="Q1 — Revenue Scale",
        format_fn=fmt_currency,
        colors=[COLOR_ACCENT],
        compare_values=rev_compare,
        watermark=wm,
    )

    # ── Q2 ─────────────────────────────────────────────────────────────────────
    q2 = metrics.q2
    cagr_labels = ["1yr CAGR", "3yr CAGR", "5yr CAGR"]
    cagr_vals = [q2.revenue_1yr_cagr, q2.revenue_3yr_cagr, q2.revenue_5yr_cagr]
    cagr_colors = [
        _metric_color(v, tm.get("growth.revenue_cagr_1yr_strong"),
                      tm.get("growth.revenue_cagr_1yr_acceptable"))
        for v in cagr_vals
    ]
    charts["q2_revenue_cagr"] = bar_chart(
        cagr_labels, cagr_vals,
        title="Q2 — Revenue CAGR by Time Period",
        y_label="CAGR",
        format_fn=fmt_pct,
        colors=cagr_colors,
        watermark=wm,
    )

    # ── Q3 ─────────────────────────────────────────────────────────────────────
    q3 = metrics.q3
    margin_labels = ["Gross Margin", "Operating Margin", "Net Margin", "EBITDA Margin"]
    margin_vals = [q3.gross_margin, q3.operating_margin, q3.net_margin, q3.ebitda_margin]
    margin_prior = [q3.gross_margin_prior, q3.operating_margin_prior, q3.net_margin_prior, None]
    charts["q3_margins"] = bar_chart(
        margin_labels, margin_vals,
        title="Q3 — Profit Margins",
        y_label="Margin",
        format_fn=fmt_pct,
        compare_values=margin_prior if mode == "existing" else None,
        colors=[COLOR_STRONG if v and v > 0 else COLOR_WEAK for v in margin_vals],
        watermark=wm,
    )
    charts["q3_roic_gauge"] = gauge_chart(
        value=q3.roic,
        label="ROIC",
        min_val=0, max_val=0.40,
        strong_threshold=tm.get("profitability.roic_strong"),
        acceptable_threshold=tm.get("profitability.roic_acceptable"),
        format_fn=fmt_pct,
        watermark=wm,
        title="Q3 — Return on Invested Capital",
    )

    # ── Q4 ─────────────────────────────────────────────────────────────────────
    q4 = metrics.q4
    charts["q4_fcf_yield_gauge"] = gauge_chart(
        value=q4.fcf_yield,
        label="FCF Yield",
        min_val=0, max_val=0.15,
        strong_threshold=tm.get("cashflow.fcf_yield_strong"),
        acceptable_threshold=tm.get("cashflow.fcf_yield_acceptable"),
        format_fn=fmt_pct,
        watermark=wm,
        title="Q4 — Free Cash Flow Yield",
    )
    cf_labels = ["Operating CF", "CapEx", "Free CF"]
    cf_vals = [q4.operating_cash_flow, q4.capital_expenditure, q4.free_cash_flow]
    charts["q4_cashflow_bars"] = bar_chart(
        cf_labels, cf_vals,
        title="Q4 — Cash Flow Breakdown",
        format_fn=fmt_currency,
        colors=[COLOR_STRONG, COLOR_WEAK, COLOR_ACCENT],
        watermark=wm,
    )
    # Dividends
    if q4.is_dividend_payer:
        charts["q4_dividend_history"] = dividend_history_chart(
            metrics.q4.__dict__.get("_divs_series"),
            title=f"{metrics.ticker} — Annual Dividend History",
            purchase_date=purchase_date,
            watermark=wm,
        )

    # ── Q5 ─────────────────────────────────────────────────────────────────────
    q5 = metrics.q5
    charts["q5_debt_ebitda_gauge"] = gauge_chart(
        value=q5.debt_to_ebitda,
        label="Debt / EBITDA",
        min_val=0, max_val=8,
        strong_threshold=tm.get("balance_sheet.debt_to_ebitda_strong"),
        acceptable_threshold=tm.get("balance_sheet.debt_to_ebitda_acceptable"),
        format_fn=lambda v: fmt_multiple(v),
        higher_is_better=False,
        watermark=wm,
        title="Q5 — Leverage (Debt/EBITDA)",
    )
    bs_labels = ["Total Debt", "Cash", "Net Debt"]
    bs_vals = [q5.total_debt, q5.cash_and_equivalents, q5.net_debt]
    charts["q5_balance_sheet_bars"] = bar_chart(
        bs_labels, bs_vals,
        title="Q5 — Balance Sheet Snapshot",
        format_fn=fmt_currency,
        colors=[COLOR_WEAK, COLOR_STRONG, COLOR_ACCENT],
        watermark=wm,
    )

    # ── Q6 ─────────────────────────────────────────────────────────────────────
    q6 = metrics.q6
    charts["q6_pe_gauge"] = gauge_chart(
        value=q6.pe_ratio,
        label="Trailing P/E",
        min_val=0, max_val=60,
        strong_threshold=tm.get("valuation.pe_cheap"),
        acceptable_threshold=tm.get("valuation.pe_fair"),
        format_fn=fmt_multiple,
        higher_is_better=False,
        watermark=wm,
        title="Q6 — Valuation (Trailing P/E)",
    )
    charts["q6_ev_ebitda_gauge"] = gauge_chart(
        value=q6.ev_ebitda,
        label="EV/EBITDA",
        min_val=0, max_val=40,
        strong_threshold=tm.get("valuation.ev_ebitda_cheap"),
        acceptable_threshold=tm.get("valuation.ev_ebitda_fair"),
        format_fn=fmt_multiple,
        higher_is_better=False,
        watermark=wm,
        title="Q6 — Valuation (EV/EBITDA)",
    )
    val_labels = ["P/E", "Fwd P/E", "EV/EBITDA", "P/S", "P/B", "PEG"]
    val_vals = [q6.pe_ratio, q6.forward_pe, q6.ev_ebitda, q6.price_to_sales, q6.price_to_book, q6.peg_ratio]
    charts["q6_valuation_bars"] = bar_chart(
        val_labels, val_vals,
        title="Q6 — Valuation Multiples",
        format_fn=fmt_multiple,
        colors=[COLOR_NEUTRAL] * len(val_labels),
        watermark=wm,
    )

    # ── Scorecard ──────────────────────────────────────────────────────────────
    charts["scorecard_radar"] = radar_chart(
        metrics.scores,
        title=f"{metrics.ticker} — Fundamental Scorecard",
        watermark=wm,
    )
    charts["scorecard_bars"] = score_bar_chart(
        metrics.scores,
        watermark=wm,
    )

    # ── Price history ──────────────────────────────────────────────────────────
    from .data_fetcher import StockData  # avoid circular at module level
    charts["price_history"] = price_history_chart(
        metrics.__dict__.get("_history_df"),
        ticker=metrics.ticker,
        purchase_date=purchase_date,
        cost_basis=cost_basis,
        watermark=wm,
    )

    return charts
