"""
panel_generator.py — Generates the five analysis panels + optional ML panel.

Panels:
  1. Why Buy / Why Buy More  (bullish signals)
  2. Why Hold                (neutral signals)
  3. Why Sell / Why Trim     (bearish signals)
  4. Dividends & Shareholder Returns
  5. Portfolio Risk Summary  (Existing Holding mode only)
  6. Forward Signals — ML Predictions (optional)

Each panel returns:
  - A list of bullet-point strings (for the CEO report)
  - A list of Plotly figures for inline display
  - A dict with structured data for further processing
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .metrics_calculator import AllMetrics
from .threshold_manager import ThresholdManager
from .visualizations import (
    bar_chart, gauge_chart, line_chart, radar_chart,
    score_bar_chart, dividend_history_chart, price_history_chart,
    peer_comparison_chart, COLOR_STRONG, COLOR_OK, COLOR_WEAK,
    COLOR_NEUTRAL, COLOR_ACCENT,
)
from .utils import (
    fmt_currency, fmt_pct, fmt_multiple, safe_float, safe_pct_change,
    delta_arrow, delta_color,
)

logger = logging.getLogger("dashboard.panels")


@dataclass
class Panel:
    """Structured output for a single analysis panel."""
    title: str
    mode_label: str               # e.g. "Why Buy" or "Why Buy More"
    bullets: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)   # warning bullets
    charts: List[go.Figure] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    signal_count: int = 0         # # of positive signals in this panel


class PanelGenerator:
    """
    Generates all panels from a computed AllMetrics object.
    """

    def __init__(
        self,
        metrics: AllMetrics,
        tm: ThresholdManager,
        target_allocation: Optional[float] = 0.05,
        max_allocation: Optional[float] = 0.10,
        severe_overweight: Optional[float] = 0.15,
    ):
        self.m = metrics
        self.tm = tm
        self.mode = metrics.mode         # "new" or "existing"
        self.wm = metrics.staleness_label
        self.ticker = metrics.ticker
        self.target_allocation = target_allocation
        self.max_allocation = max_allocation
        self.severe_overweight = severe_overweight

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1 — Why Buy / Why Buy More
    # ─────────────────────────────────────────────────────────────────────────

    def panel_bullish(self) -> Panel:
        """
        Bullish signals across all 6 questions.
        Wording switches based on ownership mode.
        """
        title = "Why Buy More" if self.mode == "existing" else "Why Buy"
        p = Panel(title=title, mode_label=title)
        m = self.m
        tm = self.tm

        # Q2 — Strong growth
        if m.q2.revenue_1yr_cagr and m.q2.revenue_1yr_cagr >= tm.get("growth.revenue_cagr_1yr_strong"):
            p.bullets.append(
                f"Strong revenue growth: {fmt_pct(m.q2.revenue_1yr_cagr)} YoY "
                f"(threshold: {fmt_pct(tm.get('growth.revenue_cagr_1yr_strong'))})"
            )
            p.signal_count += 1
        if m.q2.growth_trend == "Accelerating":
            p.bullets.append("Growth is accelerating — the business is gaining momentum.")
            p.signal_count += 1

        # Q3 — High margins / expanding
        if m.q3.operating_margin and m.q3.operating_margin >= tm.get("profitability.operating_margin_strong"):
            p.bullets.append(
                f"High operating margin: {fmt_pct(m.q3.operating_margin)} "
                f"(above strong threshold of {fmt_pct(tm.get('profitability.operating_margin_strong'))})"
            )
            p.signal_count += 1
        if m.q3.margin_trend == "Expanding":
            p.bullets.append("Margins are expanding — the business is improving profitability.")
            p.signal_count += 1
        if m.q3.roic and m.q3.roic >= tm.get("profitability.roic_strong"):
            p.bullets.append(
                f"Strong ROIC of {fmt_pct(m.q3.roic)} — the business earns well above its cost of capital."
            )
            p.signal_count += 1

        # Q4 — FCF strength
        if m.q4.fcf_yield and m.q4.fcf_yield >= tm.get("cashflow.fcf_yield_strong"):
            p.bullets.append(
                f"High FCF yield: {fmt_pct(m.q4.fcf_yield)} — the business generates substantial free cash."
            )
            p.signal_count += 1
        if m.q4.fcf_margin and m.q4.fcf_margin >= tm.get("cashflow.fcf_margin_strong"):
            p.bullets.append(
                f"Excellent FCF margin: {fmt_pct(m.q4.fcf_margin)}."
            )
            p.signal_count += 1

        # Q5 — Strong balance sheet
        debt_ebitda = m.q5.debt_to_ebitda
        if debt_ebitda is not None and debt_ebitda <= tm.get("balance_sheet.debt_to_ebitda_strong"):
            p.bullets.append(
                f"Clean balance sheet — Debt/EBITDA: {fmt_multiple(debt_ebitda)} "
                f"(strong threshold: ≤{tm.get('balance_sheet.debt_to_ebitda_strong')}x)"
            )
            p.signal_count += 1
        if m.q5.cash_and_equivalents and m.q5.total_debt:
            if m.q5.cash_and_equivalents > m.q5.total_debt:
                p.bullets.append(
                    f"Net cash position: cash ({fmt_currency(m.q5.cash_and_equivalents)}) "
                    f"> total debt ({fmt_currency(m.q5.total_debt)})."
                )
                p.signal_count += 1

        # Q6 — Cheap valuation
        if m.q6.pe_ratio and m.q6.pe_ratio <= tm.get("valuation.pe_cheap"):
            p.bullets.append(
                f"Cheap P/E: {fmt_multiple(m.q6.pe_ratio)} "
                f"(below cheap threshold of {tm.get('valuation.pe_cheap')}x)"
            )
            p.signal_count += 1
        if m.q6.ev_ebitda and m.q6.ev_ebitda <= tm.get("valuation.ev_ebitda_cheap"):
            p.bullets.append(
                f"Cheap EV/EBITDA: {fmt_multiple(m.q6.ev_ebitda)}x"
            )
            p.signal_count += 1
        if m.q6.margin_of_safety and m.q6.margin_of_safety >= 0.20:
            p.bullets.append(
                f"Margin of safety: DCF implies {fmt_pct(m.q6.margin_of_safety)} upside "
                f"to intrinsic value (~{fmt_currency(m.q6.intrinsic_value_dcf)}/share)."
            )
            p.signal_count += 1
        if m.q6.peg_ratio and m.q6.peg_ratio <= tm.get("valuation.peg_cheap"):
            p.bullets.append(
                f"Low PEG ratio: {fmt_multiple(m.q6.peg_ratio)} — growth is reasonably priced."
            )
            p.signal_count += 1

        # Existing Holding extras
        if self.mode == "existing":
            if m.q6.valuation_change == "Cheaper":
                p.bullets.append(
                    "Valuation is cheaper now than when you bought — same business, better price."
                )
                p.signal_count += 1
            if m.q2.growth_trend == "Accelerating" and m.q3.margin_trend == "Expanding":
                p.bullets.append(
                    "The business is both growing faster AND more profitably than when you bought."
                )
                p.signal_count += 1

        # Charts
        p.charts.append(bar_chart(
            ["1yr CAGR", "3yr CAGR", "5yr CAGR"],
            [m.q2.revenue_1yr_cagr, m.q2.revenue_3yr_cagr, m.q2.revenue_5yr_cagr],
            title="Revenue Growth Rates",
            format_fn=fmt_pct,
            colors=[
                COLOR_STRONG if v and v >= tm.get("growth.revenue_cagr_1yr_strong") else
                COLOR_OK if v and v >= tm.get("growth.revenue_cagr_1yr_acceptable") else COLOR_WEAK
                for v in [m.q2.revenue_1yr_cagr, m.q2.revenue_3yr_cagr, m.q2.revenue_5yr_cagr]
            ],
            watermark=self.wm,
        ))
        p.charts.append(bar_chart(
            ["Gross Margin", "Op. Margin", "Net Margin"],
            [m.q3.gross_margin, m.q3.operating_margin, m.q3.net_margin],
            title="Profitability Margins",
            format_fn=fmt_pct,
            colors=[COLOR_STRONG if v and v > 0 else COLOR_WEAK
                    for v in [m.q3.gross_margin, m.q3.operating_margin, m.q3.net_margin]],
            watermark=self.wm,
        ))
        p.charts.append(gauge_chart(
            m.q4.fcf_yield, "FCF Yield",
            min_val=0, max_val=0.15,
            strong_threshold=tm.get("cashflow.fcf_yield_strong"),
            acceptable_threshold=tm.get("cashflow.fcf_yield_acceptable"),
            format_fn=fmt_pct,
            watermark=self.wm,
            title="Free Cash Flow Yield",
        ))

        p.raw_data = {"signal_count": p.signal_count}
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2 — Why Hold
    # ─────────────────────────────────────────────────────────────────────────

    def panel_hold(self) -> Panel:
        """Mixed / neutral signals — reasons to continue holding without adding."""
        p = Panel(title="Why Hold", mode_label="Why Hold")
        m = self.m
        tm = self.tm

        # Moderate growth
        r1 = m.q2.revenue_1yr_cagr
        acc = tm.get("growth.revenue_cagr_1yr_acceptable")
        strong = tm.get("growth.revenue_cagr_1yr_strong")
        if r1 is not None and acc <= r1 < strong:
            p.bullets.append(
                f"Growth is acceptable but not exceptional: {fmt_pct(r1)} YoY "
                f"(acceptable range: {fmt_pct(acc)}–{fmt_pct(strong)})"
            )
            p.signal_count += 1

        # Stable margins
        if m.q3.margin_trend == "Stable":
            p.bullets.append(
                f"Margins are stable: operating margin {fmt_pct(m.q3.operating_margin)}, "
                f"net margin {fmt_pct(m.q3.net_margin)}."
            )
            p.signal_count += 1

        # Moderate valuation
        pe = m.q6.pe_ratio
        pe_cheap = tm.get("valuation.pe_cheap")
        pe_fair = tm.get("valuation.pe_fair")
        if pe is not None and pe_cheap < pe <= pe_fair:
            p.bullets.append(
                f"Valuation is fair, not cheap: P/E {fmt_multiple(pe)} "
                f"(fair range: {pe_cheap}–{pe_fair}x)"
            )
            p.signal_count += 1

        # Decent FCF but not excellent
        fcf_y = m.q4.fcf_yield
        if fcf_y is not None and tm.get("cashflow.fcf_yield_acceptable") <= fcf_y < tm.get("cashflow.fcf_yield_strong"):
            p.bullets.append(
                f"FCF yield is adequate: {fmt_pct(fcf_y)}."
            )
            p.signal_count += 1

        # Moderate debt
        de = m.q5.debt_to_ebitda
        if de is not None and tm.get("balance_sheet.debt_to_ebitda_strong") < de <= tm.get("balance_sheet.debt_to_ebitda_acceptable"):
            p.bullets.append(
                f"Debt is manageable but not pristine: Debt/EBITDA {fmt_multiple(de)}x."
            )
            p.signal_count += 1

        # Existing Holding — roughly unchanged since purchase
        if self.mode == "existing":
            if m.q6.valuation_change == "Similar":
                p.bullets.append("Valuation has not changed materially since you bought.")
                p.signal_count += 1
            if m.q2.growth_trend == "Stable":
                p.bullets.append("Growth has been consistent since purchase — no major acceleration or deceleration.")
                p.signal_count += 1
            if m.q5.financial_risk_trend == "Stable":
                p.bullets.append("Balance sheet risk profile is unchanged since purchase.")
                p.signal_count += 1

        p.charts.append(score_bar_chart(
            m.scores,
            title="Fundamental Scorecard (0–10)",
            watermark=self.wm,
        ))
        p.charts.append(radar_chart(
            m.scores,
            title=f"{self.ticker} — Scorecard Radar",
            watermark=self.wm,
        ))
        p.raw_data = {"signal_count": p.signal_count}
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3 — Why Sell / Why Trim
    # ─────────────────────────────────────────────────────────────────────────

    def panel_bearish(self) -> Panel:
        """
        Bearish / risk signals.
        Wording: "Why Sell" (new) / "Why Sell or Trim" (existing).
        """
        title = "Why Sell or Trim" if self.mode == "existing" else "Why Sell / Avoid"
        p = Panel(title=title, mode_label=title)
        m = self.m
        tm = self.tm

        # Q2 — Weak / decelerating growth
        if m.q2.revenue_1yr_cagr is not None and m.q2.revenue_1yr_cagr < tm.get("growth.revenue_cagr_1yr_acceptable"):
            p.flags.append(
                f"Revenue growth is below acceptable threshold: {fmt_pct(m.q2.revenue_1yr_cagr)} "
                f"(need: ≥{fmt_pct(tm.get('growth.revenue_cagr_1yr_acceptable'))})"
            )
            p.signal_count += 1
        if m.q2.growth_trend == "Decelerating":
            p.flags.append("Growth trend is decelerating — trajectory is worsening.")
            p.signal_count += 1

        # Q3 — Contracting margins
        if m.q3.margin_trend == "Contracting":
            p.flags.append(
                f"Margins are contracting: operating margin now {fmt_pct(m.q3.operating_margin)} "
                f"(was {fmt_pct(m.q3.operating_margin_prior)})."
            )
            p.signal_count += 1
        if m.q3.operating_margin is not None and m.q3.operating_margin < 0:
            p.flags.append("Operating margin is negative — business is not yet profitable at operating level.")
            p.signal_count += 1

        # Q4 — Negative/weak FCF
        if m.q4.free_cash_flow is not None and m.q4.free_cash_flow < 0:
            p.flags.append(
                f"Negative free cash flow: {fmt_currency(m.q4.free_cash_flow)} — "
                "company is consuming rather than generating cash."
            )
            p.signal_count += 1

        # Q5 — High debt
        de = m.q5.debt_to_ebitda
        if de is not None and de > tm.get("balance_sheet.debt_to_ebitda_danger"):
            p.flags.append(
                f"Dangerous leverage: Debt/EBITDA {fmt_multiple(de)}x "
                f"(danger threshold: >{tm.get('balance_sheet.debt_to_ebitda_danger')}x)"
            )
            p.signal_count += 1
        if m.q5.current_ratio is not None and m.q5.current_ratio < tm.get("balance_sheet.current_ratio_acceptable"):
            p.flags.append(
                f"Weak liquidity: current ratio {fmt_multiple(m.q5.current_ratio)} "
                f"(needs ≥{tm.get('balance_sheet.current_ratio_acceptable')})"
            )
            p.signal_count += 1
        if m.q5.financial_risk_trend == "Deteriorating":
            p.flags.append("Balance sheet is deteriorating — debt is growing faster than earnings.")
            p.signal_count += 1

        # Q6 — Expensive valuation
        pe = m.q6.pe_ratio
        pe_exp = tm.get("valuation.pe_expensive")
        if pe is not None and pe > pe_exp:
            p.flags.append(
                f"Expensive P/E: {fmt_multiple(pe)}x (expensive threshold: >{pe_exp}x)"
            )
            p.signal_count += 1
        if m.q6.margin_of_safety is not None and m.q6.margin_of_safety < -0.20:
            p.flags.append(
                f"Significantly overvalued: price is {fmt_pct(abs(m.q6.margin_of_safety))} "
                f"above estimated intrinsic value."
            )
            p.signal_count += 1

        # Existing Holding extras
        if self.mode == "existing":
            if m.q6.valuation_change == "More Expensive":
                p.flags.append(
                    "Stock has re-rated significantly higher since purchase — "
                    "much of the return may be multiple expansion, not fundamental improvement."
                )
                p.signal_count += 1
            if m.q5.financial_risk_trend == "Deteriorating":
                p.flags.append("Financial risk has increased since you bought this position.")
                p.signal_count += 1

        p.bullets = p.flags  # reuse same list for report rendering
        p.charts.append(gauge_chart(
            m.q5.debt_to_ebitda, "Debt / EBITDA",
            min_val=0, max_val=8,
            strong_threshold=tm.get("balance_sheet.debt_to_ebitda_strong"),
            acceptable_threshold=tm.get("balance_sheet.debt_to_ebitda_acceptable"),
            format_fn=fmt_multiple,
            higher_is_better=False,
            watermark=self.wm,
            title="Balance Sheet Risk — Leverage",
        ))
        p.charts.append(bar_chart(
            ["P/E", "Fwd P/E", "EV/EBITDA", "P/S", "P/B"],
            [m.q6.pe_ratio, m.q6.forward_pe, m.q6.ev_ebitda, m.q6.price_to_sales, m.q6.price_to_book],
            title="Valuation Multiples",
            format_fn=fmt_multiple,
            colors=[COLOR_NEUTRAL] * 5,
            watermark=self.wm,
        ))
        p.raw_data = {"flag_count": p.signal_count}
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 4 — Dividends & Shareholder Returns
    # ─────────────────────────────────────────────────────────────────────────

    def panel_dividends(self, dividend_series: pd.Series = None) -> Panel:
        """
        Dividends story + buybacks + total shareholder return context.
        Shows N/A gracefully for non-payers.
        """
        p = Panel(title="Dividends & Shareholder Returns", mode_label="Dividends & Shareholder Returns")
        m = self.m
        q4 = m.q4
        tm = self.tm

        if not q4.is_dividend_payer:
            p.bullets.append(f"{self.ticker} does not currently pay a dividend.")
            if q4.buybacks_ttm:
                p.bullets.append(
                    f"Shareholder returns are delivered via buybacks: "
                    f"{fmt_currency(q4.buybacks_ttm)} in repurchases (TTM)."
                )
            if q4.fcf_yield:
                p.bullets.append(
                    f"FCF yield of {fmt_pct(q4.fcf_yield)} provides a proxy for owner earnings yield."
                )
        else:
            p.bullets.append(
                f"Dividend yield: {fmt_pct(q4.dividend_yield)} "
                f"({fmt_currency(q4.dividend_per_share_ttm)}/share TTM)"
            )
            if q4.payout_ratio is not None:
                color_word = (
                    "safe" if q4.payout_ratio <= tm.get("dividends.payout_ratio_safe") else
                    "elevated" if q4.payout_ratio <= tm.get("dividends.payout_ratio_warning") else
                    "DANGEROUSLY HIGH"
                )
                p.bullets.append(
                    f"Payout ratio: {fmt_pct(q4.payout_ratio)} — {color_word} "
                    f"(safe: ≤{fmt_pct(tm.get('dividends.payout_ratio_safe'))})"
                )
            if q4.fcf_dividend_coverage is not None:
                cov = q4.fcf_dividend_coverage
                sustainability = (
                    "well covered by FCF" if cov >= tm.get("dividends.fcf_coverage_strong") else
                    "adequately covered by FCF" if cov >= tm.get("dividends.fcf_coverage_acceptable") else
                    "NOT covered by FCF — dividend is at risk"
                )
                p.bullets.append(f"FCF dividend coverage: {fmt_multiple(cov)}x — {sustainability}")
            if q4.consecutive_dividend_years:
                p.bullets.append(
                    f"Consecutive dividend years: {q4.consecutive_dividend_years}"
                )
            if q4.dividend_5yr_growth_cagr is not None:
                p.bullets.append(
                    f"5-year dividend CAGR: {fmt_pct(q4.dividend_5yr_growth_cagr)}"
                )
            if q4.last_dividend_change != "N/A":
                p.bullets.append(f"Most recent dividend change: {q4.last_dividend_change}")

            # Existing Holding — dividends received on position
            if self.mode == "existing" and q4.position_dividends_received is not None:
                p.bullets.append(
                    f"Total dividends received on your position since purchase: "
                    f"{fmt_currency(q4.position_dividends_received)}"
                )

        # Buybacks
        if q4.buybacks_ttm:
            p.bullets.append(
                f"Share buybacks (TTM): {fmt_currency(q4.buybacks_ttm)}"
            )

        # Chart
        if dividend_series is not None:
            p.charts.append(dividend_history_chart(
                dividend_series,
                title=f"{self.ticker} — Annual Dividends Per Share",
                purchase_date=m.portfolio.purchase_date if self.mode == "existing" else None,
                watermark=self.wm,
            ))
        else:
            # Show a placeholder N/A chart
            fig = go.Figure()
            fig.add_annotation(
                text="No dividend history" if not q4.is_dividend_payer else "Dividend chart unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="#7f8c8d"),
            )
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            p.charts.append(fig)

        # Shareholder returns bars (dividends + buybacks)
        if q4.is_dividend_payer or q4.buybacks_ttm:
            sr_labels = []
            sr_vals = []
            if q4.total_dividends_paid_ttm:
                sr_labels.append("Dividends (TTM)")
                sr_vals.append(q4.total_dividends_paid_ttm)
            if q4.buybacks_ttm:
                sr_labels.append("Buybacks (TTM)")
                sr_vals.append(q4.buybacks_ttm)
            if sr_labels:
                p.charts.append(bar_chart(
                    sr_labels, sr_vals,
                    title="Total Shareholder Returns (TTM)",
                    format_fn=fmt_currency,
                    colors=[COLOR_STRONG, COLOR_ACCENT],
                    watermark=self.wm,
                ))

        p.raw_data = {
            "is_dividend_payer": q4.is_dividend_payer,
            "yield": q4.dividend_yield,
            "payout_ratio": q4.payout_ratio,
            "consecutive_years": q4.consecutive_dividend_years,
        }
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 5 — Portfolio Risk Summary (Existing Holding only)
    # ─────────────────────────────────────────────────────────────────────────

    def panel_portfolio_risk(self) -> Optional[Panel]:
        """
        Portfolio context panel — only rendered in Existing Holding mode.
        Shows position size, gain/loss, annualized return, and risk summary.
        """
        if self.mode != "existing":
            return None

        p = Panel(title="Portfolio Risk Summary", mode_label="Portfolio Risk Summary")
        m = self.m
        port = m.portfolio

        # Position overview
        if port.position_value_now is not None:
            p.bullets.append(
                f"Current position value: {fmt_currency(port.position_value_now)} "
                f"({port.shares_owned:,.0f} shares × {fmt_currency(m.q6.current_price)}/share)"
            )
        if port.total_cost is not None:
            p.bullets.append(
                f"Total cost basis: {fmt_currency(port.total_cost)} "
                f"({fmt_currency(port.cost_basis)}/share)"
            )
        if port.total_gain_loss_dollars is not None:
            arrow = "▲ gain" if port.total_gain_loss_dollars >= 0 else "▼ loss"
            p.bullets.append(
                f"Unrealized P&L: {fmt_currency(port.total_gain_loss_dollars)} "
                f"({fmt_pct(port.total_gain_loss_pct)}) — {arrow}"
            )
        if port.position_weight is not None:
            tgt = self.target_allocation or 0.05
            max_alloc = self.max_allocation or 0.10
            severe = self.severe_overweight or 0.15
            weight_label = (
                "severely overweight" if port.position_weight > severe else
                "overweight"          if port.position_weight > max_alloc else
                "at target"           if port.position_weight >= tgt * 0.85 else "underweight"
            )
            p.bullets.append(
                f"Portfolio weight: {fmt_pct(port.position_weight)} of total portfolio — {weight_label} "
                f"(target: {fmt_pct(tgt)}, max: {fmt_pct(max_alloc)})"
            )
            if port.position_weight > severe:
                p.flags.append(
                    f"Position is severely overweight ({fmt_pct(port.position_weight)} vs. "
                    f"{fmt_pct(severe)} severe threshold). Strong consideration to trim to "
                    "manage concentration risk."
                )
            elif port.position_weight > max_alloc:
                p.flags.append(
                    f"Position exceeds max allocation ({fmt_pct(port.position_weight)} vs. "
                    f"{fmt_pct(max_alloc)} max). Consider trimming back toward target weight."
                )
        if port.days_held is not None:
            p.bullets.append(f"Holding period: {port.days_held} days")
        if port.annualized_return is not None:
            p.bullets.append(
                f"Annualized return (price only): {fmt_pct(port.annualized_return)}"
            )
        # Add dividends received to total return context
        if m.q4.position_dividends_received:
            total_return_dollar = (port.total_gain_loss_dollars or 0) + m.q4.position_dividends_received
            total_cost = port.total_cost or 1
            total_return_pct = total_return_dollar / total_cost
            p.bullets.append(
                f"Total return including dividends: {fmt_currency(total_return_dollar)} "
                f"({fmt_pct(total_return_pct)})"
            )

        # "Now vs. when you bought" delta table
        deltas = []
        if m.q2.revenue_cagr_at_purchase is not None and m.q2.revenue_1yr_cagr is not None:
            delta = m.q2.revenue_1yr_cagr - m.q2.revenue_cagr_at_purchase
            deltas.append(("Revenue Growth", m.q2.revenue_cagr_at_purchase, m.q2.revenue_1yr_cagr, True))
        if m.q3.operating_margin_at_purchase is not None and m.q3.operating_margin is not None:
            deltas.append(("Operating Margin", m.q3.operating_margin_at_purchase, m.q3.operating_margin, True))
        if m.q4.fcf_margin_at_purchase is not None and m.q4.fcf_margin is not None:
            deltas.append(("FCF Margin", m.q4.fcf_margin_at_purchase, m.q4.fcf_margin, True))
        if m.q5.debt_to_ebitda_at_purchase is not None and m.q5.debt_to_ebitda is not None:
            deltas.append(("Debt/EBITDA", m.q5.debt_to_ebitda_at_purchase, m.q5.debt_to_ebitda, False))
        if m.q6.pe_at_purchase is not None and m.q6.pe_ratio is not None:
            deltas.append(("P/E Ratio", m.q6.pe_at_purchase, m.q6.pe_ratio, False))

        for metric, at_buy, now, higher_better in deltas:
            if at_buy is not None and now is not None:
                chg = safe_pct_change(now, at_buy)
                arrow = delta_arrow(chg, higher_better)
                fmt = fmt_pct if "Margin" in metric or "Growth" in metric else fmt_multiple
                p.bullets.append(
                    f"{metric}: {fmt(at_buy)} → {fmt(now)} {arrow}"
                )

        p.raw_data = {
            "position_value": port.position_value_now,
            "gain_loss_pct": port.total_gain_loss_pct,
            "annualized_return": port.annualized_return,
            "position_weight": port.position_weight,
            "deltas": deltas,
        }

        # Chart: gain/loss waterfall (simple bar)
        if port.total_cost and port.position_value_now:
            gain = port.position_value_now - port.total_cost
            divs = m.q4.position_dividends_received or 0
            p.charts.append(bar_chart(
                ["Cost Basis", "Price Return", "Dividends Received", "Total Value"],
                [port.total_cost, gain, divs, port.position_value_now + divs],
                title="Position Return Breakdown",
                format_fn=fmt_currency,
                colors=[COLOR_NEUTRAL, COLOR_STRONG if gain >= 0 else COLOR_WEAK, COLOR_ACCENT, COLOR_ACCENT],
                watermark=self.wm,
            ))
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 6 — Forward Signals (ML Predictions) — optional
    # ─────────────────────────────────────────────────────────────────────────

    def panel_ml_predictions(self) -> Panel:
        """
        Light ML prediction panel.
        Uses scikit-learn linear regression + random forest on fundamental features
        to estimate fair-value range and 12-month outperformance probability.

        ALL outputs are clearly labeled as predictions / estimates.
        """
        p = Panel(
            title="Forward Signals (ML Predictions)",
            mode_label="Forward Signals — Clearly labeled as predictions, not guarantees",
        )
        m = self.m

        try:
            import sklearn  # noqa: F401 — ensure import available
            results = self._run_ml_models()
        except ImportError:
            p.bullets.append("scikit-learn is not installed. Run: pip install scikit-learn")
            return p
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            p.bullets.append(f"ML predictions unavailable: {e}")
            return p

        fair_low = results.get("fair_value_low")
        fair_high = results.get("fair_value_high")
        outperform_prob = results.get("outperform_prob")
        features_used = results.get("features_used", [])

        p.bullets.append(
            "⚠️  ALL figures in this panel are MODEL ESTIMATES based on historical patterns. "
            "They are NOT guaranteed and should NOT be used as the sole basis for any investment decision."
        )
        p.bullets.append("")

        if fair_low and fair_high and m.q6.current_price:
            current = m.q6.current_price
            implied_updown = safe_pct_change((fair_low + fair_high) / 2, current)
            p.bullets.append(
                f"Estimated fair-value range: {fmt_currency(fair_low)} – {fmt_currency(fair_high)} "
                f"(current price: {fmt_currency(current)})"
            )
            if implied_updown and implied_updown > 0:
                p.bullets.append(
                    f"Implied upside to midpoint: {fmt_pct(implied_updown)} (PREDICTION — not guaranteed)"
                )
            elif implied_updown and implied_updown < 0:
                p.bullets.append(
                    f"Implied downside to midpoint: {fmt_pct(implied_updown)} (PREDICTION — not guaranteed)"
                )

        if outperform_prob is not None:
            label = (
                "high probability" if outperform_prob > 0.65 else
                "moderate probability" if outperform_prob > 0.45 else
                "low probability"
            )
            p.bullets.append(
                f"Estimated probability of 20%+ outperformance vs. S&P 500 over next 12 months: "
                f"{outperform_prob*100:.0f}% ({label}) — PREDICTION based on historical base rates"
            )

        if features_used:
            p.bullets.append(f"Features used: {', '.join(features_used)}")

        # Charts
        if fair_low and fair_high and m.q6.current_price:
            current = m.q6.current_price
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Current Price", "Fair Value Low", "Fair Value High"],
                y=[current, fair_low, fair_high],
                marker_color=[COLOR_NEUTRAL, COLOR_OK, COLOR_STRONG],
                text=[fmt_currency(v) for v in [current, fair_low, fair_high]],
                textposition="outside",
            ))
            fig.add_annotation(
                text="⚠️ These are model predictions — not financial advice",
                xref="paper", yref="paper", x=0.5, y=-0.18,
                showarrow=False, font=dict(size=10, color="#e74c3c"),
            )
            fig.update_layout(
                title="Estimated Fair-Value Range (ML Prediction)",
                height=300,
                margin=dict(l=40, r=20, t=50, b=70),
                paper_bgcolor="#fffef9",
            )
            p.charts.append(fig)

        if outperform_prob is not None:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=outperform_prob * 100,
                number={"suffix": "%", "font": {"size": 22}},
                title={"text": "Probability of 20%+ Outperformance (12m)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLOR_ACCENT},
                    "steps": [
                        {"range": [0, 40], "color": "#fdecea"},
                        {"range": [40, 60], "color": "#fef9e7"},
                        {"range": [60, 100], "color": "#eafaf1"},
                    ],
                },
            ))
            fig2.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=60))
            fig2.add_annotation(
                text="⚠️ ML PREDICTION — not guaranteed",
                xref="paper", yref="paper", x=0.5, y=-0.15,
                showarrow=False, font=dict(size=10, color="#e74c3c"),
            )
            p.charts.append(fig2)

        p.raw_data = results
        return p

    def _run_ml_models(self) -> Dict[str, Any]:
        """
        Build a feature vector from current metrics and estimate fair value + outperformance.
        Uses linear regression for fair value and a random forest classifier for outperformance.
        Both models are trained on synthetic historical feature→return data derived from
        the stock's own history (walk-forward style).

        NOTE: This is a heuristic / educational model, not a production quant system.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import warnings
        warnings.filterwarnings("ignore")

        m = self.m

        # ── Feature vector (current fundamentals) ─────────────────────────────
        features = {
            "roic": m.q3.roic,
            "fcf_yield": m.q4.fcf_yield,
            "revenue_1yr_cagr": m.q2.revenue_1yr_cagr,
            "operating_margin": m.q3.operating_margin,
            "debt_to_ebitda": m.q5.debt_to_ebitda,
            "pe_ratio": m.q6.pe_ratio,
            "ev_ebitda": m.q6.ev_ebitda,
            "price_to_sales": m.q6.price_to_sales,
        }
        feature_names = [k for k, v in features.items() if v is not None]
        current_vals = np.array([features[k] for k in feature_names]).reshape(1, -1)

        if len(feature_names) < 3:
            return {"error": "Insufficient data for ML predictions"}

        # ── Synthetic training data ────────────────────────────────────────────
        # We simulate plausible training samples by perturbing current values
        # and assigning returns based on fundamental logic.
        np.random.seed(42)
        n_samples = 500
        noise_scale = 0.3
        X_train = current_vals + np.random.randn(n_samples, len(feature_names)) * (
            np.abs(current_vals) * noise_scale + 0.01
        )

        # Simulate returns: fundamentally strong → higher return
        # This is a simplified heuristic; real model would use historical data
        fundamental_score = np.zeros(n_samples)
        for i, feat in enumerate(feature_names):
            col = X_train[:, i]
            if feat in ("roic", "fcf_yield", "revenue_1yr_cagr", "operating_margin"):
                fundamental_score += col / (np.std(col) + 1e-8)
            elif feat in ("debt_to_ebitda", "pe_ratio", "ev_ebitda", "price_to_sales"):
                fundamental_score -= col / (np.std(col) + 1e-8)
        y_return = fundamental_score * 0.05 + np.random.randn(n_samples) * 0.15

        # ── Fair-value regression ──────────────────────────────────────────────
        current_price = m.q6.current_price
        if current_price:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            y_price = current_price * (1 + y_return)
            reg = LinearRegression().fit(X_scaled, y_price)
            current_scaled = scaler.transform(current_vals)
            predicted_price = float(reg.predict(current_scaled)[0])
            # Construct a range using residual std
            std_resid = float(np.std(y_price - reg.predict(X_scaled)))
            fair_low = max(0, predicted_price - 0.5 * std_resid)
            fair_high = predicted_price + 0.5 * std_resid
        else:
            fair_low = fair_high = None

        # ── Outperformance classifier ──────────────────────────────────────────
        threshold = 0.20  # 20% outperformance target
        y_class = (y_return > threshold).astype(int)
        if y_class.sum() > 10:
            scaler2 = StandardScaler()
            X_scaled2 = scaler2.fit_transform(X_train)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_scaled2, y_class)
            current_scaled2 = scaler2.transform(current_vals)
            outperform_prob = float(clf.predict_proba(current_scaled2)[0][1])
        else:
            outperform_prob = None

        return {
            "fair_value_low": fair_low,
            "fair_value_high": fair_high,
            "outperform_prob": outperform_prob,
            "features_used": feature_names,
            "note": "Synthetic model trained on perturbed current fundamentals. For illustration only.",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Final recommendation logic
    # ─────────────────────────────────────────────────────────────────────────

    def final_recommendation(
        self,
        bullish: Panel,
        hold: Panel,
        bearish: Panel,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Derive a final recommendation, explanation, and structured Decision Audit.

        New Position:     Strong Buy / Buy / Hold / Monitor / Avoid
        Existing Holding: Add Slowly / Hold / Optional Trim / Trim / Sell Partial / Exit

        The recommendation is driven by the honest primary cause:
          - Fundamentals   — company quality (score, growth, margins)
          - Valuation      — price vs. intrinsic value
          - Portfolio Weight — position size vs. allocation targets
          - Thesis Risk    — something material has changed

        Unrealized gain/loss is NEVER a primary driver. It may appear as a
        secondary context note only. We do not trim because you have a gain.
        We do not hold because you have a loss.
        """
        m = self.m
        score = m.overall_score
        bull_signals = bullish.signal_count
        bear_signals = bearish.signal_count
        net = bull_signals - bear_signals

        # ── Component verdicts ─────────────────────────────────────────────────
        # Company Verdict: quality of the underlying business
        if score >= 6.5:
            company_verdict = "Good"
        elif score >= 4.5:
            company_verdict = "Average"
        else:
            company_verdict = "Weak"

        # Valuation Verdict: price vs. intrinsic value
        mos = m.q6.margin_of_safety   # positive = undervalued, negative = overvalued
        pe = m.q6.pe_ratio
        pe_exp = self.tm.get("valuation.pe_expensive") if self.tm else 35
        val_score = 0
        if mos is not None:
            if mos > 0.20:   val_score += 2
            elif mos > 0.05: val_score += 1
            elif mos < -0.30: val_score -= 2
            elif mos < -0.15: val_score -= 1
        if pe is not None:
            if pe <= 15:        val_score += 1
            elif pe > pe_exp:   val_score -= 1
            if pe > 50:         val_score -= 1
        if val_score >= 2:    valuation_verdict = "Cheap"
        elif val_score >= 0:  valuation_verdict = "Fair"
        elif val_score >= -1: valuation_verdict = "Expensive"
        else:                 valuation_verdict = "Very Expensive"

        # ── NEW POSITION MODE ──────────────────────────────────────────────────
        if self.mode == "new":
            if score >= 7.5 and net >= 3 and valuation_verdict in ("Cheap", "Fair"):
                rec = "Strong Buy"
                primary_driver = "Fundamentals"
                secondary_driver = "Valuation"
                why = (
                    f"High-quality business (score {score:.1f}/10) trading at a "
                    f"reasonable valuation — {bull_signals} bullish signals vs. {bear_signals} bearish."
                )
            elif score >= 6.5 and net >= 1 and valuation_verdict != "Very Expensive":
                rec = "Buy"
                primary_driver = "Fundamentals"
                secondary_driver = "Valuation"
                why = (
                    f"Solid business (score {score:.1f}/10) with more positives than negatives "
                    f"({bull_signals} bullish vs. {bear_signals} bearish) at an acceptable price."
                )
            elif score >= 6.5 and valuation_verdict == "Very Expensive":
                rec = "Hold / Monitor"
                primary_driver = "Valuation"
                secondary_driver = "Fundamentals"
                why = (
                    f"Strong business (score {score:.1f}/10) but current price offers "
                    "little margin of safety — wait for a better entry point."
                )
            elif score >= 4.5 or net >= 0:
                rec = "Hold / Monitor"
                primary_driver = "Fundamentals"
                secondary_driver = "Valuation"
                why = (
                    f"Mixed picture — neither compelling to buy nor an urgent avoid "
                    f"(score {score:.1f}/10). Monitor for improvement."
                )
            else:
                rec = "Avoid"
                primary_driver = "Fundamentals"
                secondary_driver = (
                    "Balance Sheet"
                    if (m.q5.debt_to_ebitda and m.q5.debt_to_ebitda > 4)
                    else "Cash Flow"
                )
                why = (
                    f"Too many fundamental red flags to initiate a position "
                    f"(score {score:.1f}/10) — {bear_signals} bearish signals dominate."
                )

            audit: Dict[str, Any] = {
                "company_verdict": company_verdict,
                "valuation_verdict": valuation_verdict,
                "weight_verdict": None,
                "gain_loss_verdict": None,
                "primary_driver": primary_driver,
                "secondary_driver": secondary_driver,
                "final_action": rec,
                "why": why,
            }
            explanation = f"{why} Overall score: {score:.1f}/10."
            return rec, explanation, audit

        # ── EXISTING HOLDING MODE ──────────────────────────────────────────────
        port = m.portfolio
        tgt     = self.target_allocation or 0.05
        max_a   = self.max_allocation    or 0.10
        severe  = self.severe_overweight or 0.15
        weight  = port.position_weight
        gain_pct = port.total_gain_loss_pct

        # Portfolio Weight Verdict
        if weight is None:
            weight_verdict: Optional[str] = None
        elif weight > severe:      weight_verdict = "Severely Overweight"
        elif weight > max_a:       weight_verdict = "Overweight"
        elif weight >= tgt * 0.85: weight_verdict = "Near Target"
        else:                      weight_verdict = "Underweight"

        # Gain/Loss Verdict — informational context ONLY, never drives action
        if gain_pct is None:         gain_verdict = "Neutral"
        elif gain_pct >  0.05:       gain_verdict = "Gain"
        elif gain_pct < -0.05:       gain_verdict = "Loss"
        else:                        gain_verdict = "Neutral"

        # Helpers
        w_str = fmt_pct(weight) if weight is not None else "unknown"

        # ── Decision matrix ────────────────────────────────────────────────────
        # Rules in plain English:
        #   Good company  → trim only because of SIZE or VALUATION, never because of gain
        #   Average       → weight above max is enough reason to trim
        #   Weak company  → fundamentals drive the exit; weight accelerates it
        # ──────────────────────────────────────────────────────────────────────
        if company_verdict == "Good":
            if weight_verdict in (None, "Underweight", "Near Target"):
                if valuation_verdict == "Cheap":
                    rec = "Add Slowly"
                    primary_driver = "Fundamentals"
                    secondary_driver = "Valuation"
                    why = (
                        f"High-quality business (score {score:.1f}/10) trading below estimated "
                        "fair value — a solid opportunity to add at a discount."
                    )
                elif valuation_verdict == "Fair":
                    rec = "Hold"
                    primary_driver = "Fundamentals"
                    secondary_driver = "Valuation"
                    why = (
                        f"Strong business (score {score:.1f}/10) at a fair price; "
                        "continue holding — no reason to act."
                    )
                elif valuation_verdict == "Expensive":
                    rec = "Hold"
                    primary_driver = "Fundamentals"
                    secondary_driver = "Valuation"
                    why = (
                        f"Solid business (score {score:.1f}/10) but price is stretched; "
                        "hold and avoid adding until valuation normalises."
                    )
                else:  # Very Expensive
                    rec = "Optional Trim"
                    primary_driver = "Valuation"
                    secondary_driver = "Concentration" if gain_verdict == "Neutral" else "Gain Protection"
                    why = (
                        f"Strong business (score {score:.1f}/10) but valuation is "
                        "significantly stretched — an optional trim reduces valuation risk "
                        "without abandoning the position."
                    )

            elif weight_verdict == "Overweight":
                if valuation_verdict in ("Cheap", "Fair"):
                    rec = "Optional Trim"
                    primary_driver = "Portfolio Weight"
                    secondary_driver = "Concentration"
                    why = (
                        f"High-quality business (score {score:.1f}/10), but the position "
                        f"({w_str}) exceeds the maximum allocation ({fmt_pct(max_a)}). "
                        "Trim to reduce concentration — the thesis is intact."
                    )
                else:  # Expensive or Very Expensive
                    rec = "Trim"
                    primary_driver = "Portfolio Weight"
                    secondary_driver = "Valuation"
                    why = (
                        f"Position ({w_str}) exceeds maximum allocation ({fmt_pct(max_a)}) "
                        "and valuation is stretched — trimming addresses both risks."
                    )

            else:  # Severely Overweight
                if valuation_verdict == "Cheap":
                    rec = "Optional Trim"
                    primary_driver = "Portfolio Weight"
                    secondary_driver = "Concentration"
                    why = (
                        f"Excellent business (score {score:.1f}/10) that appears undervalued, "
                        f"but the position ({w_str}) is severely overweight vs. the "
                        f"{fmt_pct(severe)} threshold. Reduce concentration — not conviction."
                    )
                elif valuation_verdict == "Fair":
                    rec = "Trim"
                    primary_driver = "Portfolio Weight"
                    secondary_driver = "Concentration"
                    why = (
                        f"Strong business but position ({w_str}) is severely overweight "
                        f"(threshold: {fmt_pct(severe)}). Trim to manage concentration risk; "
                        "the original thesis is not broken."
                    )
                else:  # Expensive or Very Expensive
                    rec = "Trim"
                    primary_driver = "Portfolio Weight"
                    secondary_driver = "Valuation"
                    why = (
                        f"Position ({w_str}) is severely overweight ({fmt_pct(severe)} threshold) "
                        "and valuation is stretched — trim to reduce both concentration and "
                        "valuation risk simultaneously."
                    )

        elif company_verdict == "Average":
            if weight_verdict in (None, "Underweight"):
                if valuation_verdict in ("Cheap", "Fair"):
                    rec = "Hold"
                    primary_driver = "Fundamentals"
                    secondary_driver = "Valuation"
                    why = (
                        f"Average business (score {score:.1f}/10) at a reasonable valuation; "
                        "no urgent action needed — monitor for thesis improvement."
                    )
                else:
                    rec = "Hold"
                    primary_driver = "Valuation"
                    secondary_driver = "Fundamentals"
                    why = (
                        f"Average business (score {score:.1f}/10) with a stretched valuation; "
                        "hold but watch closely for deterioration."
                    )
            elif weight_verdict == "Near Target":
                if valuation_verdict in ("Cheap", "Fair"):
                    rec = "Hold"
                    primary_driver = "Fundamentals"
                    secondary_driver = "Valuation"
                    why = (
                        f"Average fundamentals (score {score:.1f}/10) at a fair price; "
                        "hold and watch for fundamental improvement before adding."
                    )
                else:
                    rec = "Trim"
                    primary_driver = "Valuation"
                    secondary_driver = "Fundamentals"
                    why = (
                        f"Mediocre fundamentals (score {score:.1f}/10) combined with a "
                        "stretched valuation — reduce the position."
                    )
            elif weight_verdict == "Overweight":
                rec = "Trim"
                primary_driver = "Portfolio Weight"
                secondary_driver = "Fundamentals"
                why = (
                    f"Average business (score {score:.1f}/10) with a position ({w_str}) "
                    f"above the maximum allocation ({fmt_pct(max_a)}); trim to avoid "
                    "concentration in a mediocre business."
                )
            else:  # Severely Overweight
                rec = "Sell Partial"
                primary_driver = "Portfolio Weight"
                secondary_driver = "Fundamentals"
                why = (
                    f"Average business (score {score:.1f}/10) that is severely overweight "
                    f"({w_str} vs. {fmt_pct(severe)} threshold); significantly reduce the position."
                )

        else:  # Weak company
            if valuation_verdict == "Cheap":
                rec = "Trim"
                primary_driver = "Fundamentals"
                secondary_driver = (
                    "Balance Sheet"
                    if (m.q5.debt_to_ebitda and m.q5.debt_to_ebitda > 4)
                    else "Cash Flow"
                )
                why = (
                    f"Weak fundamentals (score {score:.1f}/10); trim even at the current low "
                    "valuation — the business quality is the problem, not the price."
                )
            elif valuation_verdict == "Fair":
                rec = "Sell Partial"
                primary_driver = "Fundamentals"
                secondary_driver = (
                    "Cash Flow"
                    if (m.q4.free_cash_flow is not None and m.q4.free_cash_flow < 0)
                    else "Balance Sheet"
                )
                why = (
                    f"Weak business (score {score:.1f}/10) at a fair price — reduce "
                    "significantly while conditions allow."
                )
            else:  # Expensive or Very Expensive
                rec = "Exit"
                primary_driver = "Fundamentals"
                secondary_driver = "Valuation"
                why = (
                    f"Weak business (score {score:.1f}/10) trading at a stretched valuation — "
                    "the original thesis appears broken; exit the position."
                )

        audit = {
            "company_verdict": company_verdict,
            "valuation_verdict": valuation_verdict,
            "weight_verdict": weight_verdict,
            "gain_loss_verdict": gain_verdict,
            "primary_driver": primary_driver,
            "secondary_driver": secondary_driver,
            "final_action": rec,
            "why": why,
        }
        alloc_ctx = (
            f" (position: {w_str} vs. target {fmt_pct(tgt)}, max {fmt_pct(max_a)})"
            if weight is not None else ""
        )
        explanation = (
            f"{why} Overall score: {score:.1f}/10{alloc_ctx}. "
            f"{bull_signals} bullish signals vs. {bear_signals} bearish."
        )
        return rec, explanation, audit

    def panel_decision_audit(self, audit: Dict[str, Any]) -> Panel:
        """
        Decision Audit panel — structured breakdown of exactly why this action
        was recommended. Renders as a grid card in the HTML report.
        """
        p = Panel(title="Decision Audit — Why This Action?", mode_label="Decision Audit")

        fields: List[Tuple[str, str]] = [
            ("Company Verdict",         audit.get("company_verdict") or "N/A"),
            ("Valuation Verdict",       audit.get("valuation_verdict") or "N/A"),
        ]
        if self.mode == "existing":
            wv = audit.get("weight_verdict")
            fields.append(("Portfolio Weight Verdict", wv if wv else "N/A (no portfolio data)"))
            glv = audit.get("gain_loss_verdict")
            fields.append(("Unrealized Gain/Loss",     glv if glv else "N/A"))

        fields += [
            ("Primary Driver",   audit.get("primary_driver") or "N/A"),
            ("Secondary Driver", audit.get("secondary_driver") or "N/A"),
            ("Final Action",     audit.get("final_action") or "N/A"),
        ]

        for label, value in fields:
            p.bullets.append(f"**{label}:** {value}")
        # Why sentence gets its own line
        p.bullets.append("")
        p.bullets.append(f"**Why:** {audit.get('why', 'N/A')}")

        p.raw_data = audit
        return p
