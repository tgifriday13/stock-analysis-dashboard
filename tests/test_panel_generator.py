"""
tests/test_panel_generator.py — Unit tests for src/panel_generator.py.

PanelGenerator takes an AllMetrics object + ThresholdManager and produces
Panel objects with bullets/flags/charts.  All tests are self-contained:
AllMetrics is constructed directly (it's a dataclass) and ThresholdManager
reads from config/default_thresholds.json (local file, no network).
"""

from datetime import date
from typing import Optional

import pytest
import plotly.graph_objects as go

from src.metrics_calculator import (
    AllMetrics, Q1Scale, Q2Growth, Q3Profitability,
    Q4CashFlow, Q5BalanceSheet, Q6Valuation, PortfolioContext,
)
from src.threshold_manager import ThresholdManager
from src.panel_generator import Panel, PanelGenerator


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tm():
    """Generic ThresholdManager (no sector) shared by all tests."""
    return ThresholdManager()


def _make_metrics(mode: str = "new") -> AllMetrics:
    """Minimal AllMetrics with all-None sub-metrics (no signals fire)."""
    return AllMetrics(ticker="TEST", company_name="Test Corp", mode=mode)


def _make_bullish_metrics(mode: str = "new") -> AllMetrics:
    """AllMetrics configured so every bullish bullet fires."""
    m = _make_metrics(mode=mode)
    tm = ThresholdManager()

    # Q2 — strong revenue growth, accelerating
    m.q2.revenue_1yr_cagr = tm.get("growth.revenue_cagr_1yr_strong") + 0.05  # 0.20
    m.q2.growth_trend = "Accelerating"

    # Q3 — high margin + expanding + strong ROIC
    m.q3.operating_margin = tm.get("profitability.operating_margin_strong") + 0.10  # 0.30
    m.q3.margin_trend = "Expanding"
    m.q3.roic = tm.get("profitability.roic_strong") + 0.05  # 0.20

    # Q4 — high FCF yield + high FCF margin
    m.q4.fcf_yield = tm.get("cashflow.fcf_yield_strong") + 0.02  # 0.09
    m.q4.fcf_margin = tm.get("cashflow.fcf_margin_strong", 0.20) + 0.05

    # Q5 — low debt + net cash position
    m.q5.debt_to_ebitda = tm.get("balance_sheet.debt_to_ebitda_strong") - 0.5  # 1.0
    m.q5.cash_and_equivalents = 5_000_000_000.0
    m.q5.total_debt = 2_000_000_000.0

    # Q6 — cheap valuation on every metric
    m.q6.pe_ratio = tm.get("valuation.pe_cheap") - 3  # 12
    m.q6.ev_ebitda = tm.get("valuation.ev_ebitda_cheap") - 2  # 8
    m.q6.margin_of_safety = 0.30
    m.q6.intrinsic_value_dcf = 200.0
    m.q6.current_price = 140.0
    m.q6.peg_ratio = tm.get("valuation.peg_cheap") - 0.2 if tm.get("valuation.peg_cheap") else 0.8

    # Scores for charts
    m.scores = {"Q1": 7.0, "Q2": 8.5, "Q3": 7.5, "Q4": 8.0, "Q5": 9.0, "Q6": 6.5}

    # Existing-holding extras
    if mode == "existing":
        m.q6.valuation_change = "Cheaper"

    return m


def _make_bearish_metrics(mode: str = "new") -> AllMetrics:
    """AllMetrics configured so every bearish flag fires."""
    m = _make_metrics(mode=mode)
    tm = ThresholdManager()

    # Q2 — weak growth + decelerating
    m.q2.revenue_1yr_cagr = tm.get("growth.revenue_cagr_1yr_acceptable") - 0.02  # 0.03
    m.q2.growth_trend = "Decelerating"

    # Q3 — negative operating margin + contracting
    m.q3.operating_margin = -0.05
    m.q3.operating_margin_prior = 0.05
    m.q3.margin_trend = "Contracting"

    # Q4 — negative FCF
    m.q4.free_cash_flow = -500_000_000.0

    # Q5 — high debt + weak liquidity + deteriorating
    m.q5.debt_to_ebitda = tm.get("balance_sheet.debt_to_ebitda_danger") + 1.5  # 6.5
    m.q5.current_ratio = tm.get("balance_sheet.current_ratio_acceptable") - 0.4  # 0.6
    m.q5.financial_risk_trend = "Deteriorating"

    # Q6 — expensive valuation + significantly overvalued
    m.q6.pe_ratio = tm.get("valuation.pe_expensive") + 15  # 50
    m.q6.margin_of_safety = -0.35

    # Existing-holding extras
    if mode == "existing":
        m.q6.valuation_change = "More Expensive"

    m.scores = {"Q1": 2.0, "Q2": 1.5, "Q3": 2.5, "Q4": 1.0, "Q5": 3.0, "Q6": 2.0}
    return m


def _make_hold_metrics(mode: str = "new") -> AllMetrics:
    """AllMetrics configured so neutral/hold bullets fire."""
    m = _make_metrics(mode=mode)
    tm = ThresholdManager()

    # Q2 — moderate (acceptable but not strong) growth + stable trend
    strong = tm.get("growth.revenue_cagr_1yr_strong")
    acceptable = tm.get("growth.revenue_cagr_1yr_acceptable")
    m.q2.revenue_1yr_cagr = (acceptable + strong) / 2  # midpoint
    m.q2.growth_trend = "Stable"

    # Q3 — stable margins
    m.q3.margin_trend = "Stable"
    m.q3.operating_margin = tm.get("profitability.operating_margin_acceptable") + 0.02
    m.q3.net_margin = 0.07

    # Q4 — adequate FCF yield
    m.q4.fcf_yield = (
        tm.get("cashflow.fcf_yield_acceptable") +
        tm.get("cashflow.fcf_yield_strong")
    ) / 2

    # Q5 — moderate debt
    strong_d = tm.get("balance_sheet.debt_to_ebitda_strong")
    acc_d = tm.get("balance_sheet.debt_to_ebitda_acceptable")
    m.q5.debt_to_ebitda = (strong_d + acc_d) / 2

    # Q6 — fair (not cheap) P/E
    m.q6.pe_ratio = tm.get("valuation.pe_cheap") + 3

    # Existing-holding extras
    if mode == "existing":
        m.q6.valuation_change = "Similar"
        m.q5.financial_risk_trend = "Stable"

    m.scores = {"Q1": 5.0, "Q2": 5.5, "Q3": 5.0, "Q4": 5.5, "Q5": 6.0, "Q6": 5.0}
    return m


# ── Panel: bullish (Why Buy / Why Buy More) ───────────────────────────────────

class TestPanelBullish:
    def test_returns_panel(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        assert isinstance(p, Panel)

    def test_strong_revenue_growth_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "Strong revenue growth" in text

    def test_accelerating_growth_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "accelerating" in text.lower()

    def test_high_operating_margin_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "operating margin" in text.lower()

    def test_expanding_margins_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "Margins are expanding" in text

    def test_net_cash_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "Net cash" in text

    def test_cheap_pe_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "Cheap P/E" in text

    def test_signal_count_positive(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        assert p.signal_count > 0

    def test_no_signals_when_all_none(self, tm):
        pg = PanelGenerator(_make_metrics(), tm)
        p = pg.panel_bullish()
        assert p.bullets == []
        assert p.signal_count == 0

    def test_new_mode_title(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(mode="new"), tm)
        p = pg.panel_bullish()
        assert p.title == "Why Buy"

    def test_existing_mode_title(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(mode="existing"), tm)
        p = pg.panel_bullish()
        assert p.title == "Why Buy More"

    def test_existing_mode_cheaper_valuation_bullet(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(mode="existing"), tm)
        p = pg.panel_bullish()
        text = " ".join(p.bullets)
        assert "cheaper" in text.lower()

    def test_charts_are_figures(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        assert len(p.charts) > 0
        for c in p.charts:
            assert isinstance(c, go.Figure)

    def test_raw_data_has_signal_count(self, tm):
        pg = PanelGenerator(_make_bullish_metrics(), tm)
        p = pg.panel_bullish()
        assert "signal_count" in p.raw_data


# ── Panel: hold (Why Hold) ────────────────────────────────────────────────────

class TestPanelHold:
    def test_returns_panel(self, tm):
        pg = PanelGenerator(_make_hold_metrics(), tm)
        p = pg.panel_hold()
        assert isinstance(p, Panel)

    def test_moderate_growth_bullet(self, tm):
        pg = PanelGenerator(_make_hold_metrics(), tm)
        p = pg.panel_hold()
        text = " ".join(p.bullets)
        assert "acceptable" in text.lower()

    def test_stable_margin_bullet(self, tm):
        pg = PanelGenerator(_make_hold_metrics(), tm)
        p = pg.panel_hold()
        text = " ".join(p.bullets)
        assert "stable" in text.lower()

    def test_signal_count_positive(self, tm):
        pg = PanelGenerator(_make_hold_metrics(), tm)
        p = pg.panel_hold()
        assert p.signal_count > 0

    def test_no_signals_when_all_none(self, tm):
        # panel_hold always renders a radar chart; supply scores so it can render.
        m = _make_metrics()
        m.scores = {"Q1": 5.0, "Q2": 5.0, "Q3": 5.0, "Q4": 5.0, "Q5": 5.0, "Q6": 5.0}
        pg = PanelGenerator(m, tm)
        p = pg.panel_hold()
        assert p.signal_count == 0

    def test_existing_similar_valuation_bullet(self, tm):
        pg = PanelGenerator(_make_hold_metrics(mode="existing"), tm)
        p = pg.panel_hold()
        text = " ".join(p.bullets)
        assert "not changed materially" in text

    def test_existing_stable_growth_bullet(self, tm):
        pg = PanelGenerator(_make_hold_metrics(mode="existing"), tm)
        p = pg.panel_hold()
        text = " ".join(p.bullets)
        assert "consistent since purchase" in text

    def test_charts_are_figures(self, tm):
        pg = PanelGenerator(_make_hold_metrics(), tm)
        p = pg.panel_hold()
        assert len(p.charts) > 0
        for c in p.charts:
            assert isinstance(c, go.Figure)


# ── Panel: bearish (Why Sell) ─────────────────────────────────────────────────

class TestPanelBearish:
    def test_returns_panel(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        assert isinstance(p, Panel)

    def test_weak_revenue_growth_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "below acceptable" in text

    def test_decelerating_growth_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "Decelerating" in text or "decelerating" in text

    def test_negative_operating_margin_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "negative" in text.lower()

    def test_negative_fcf_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "Negative free cash flow" in text

    def test_high_debt_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "Dangerous leverage" in text or "Debt/EBITDA" in text

    def test_expensive_pe_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "Expensive P/E" in text

    def test_significantly_overvalued_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "overvalued" in text

    def test_signal_count_positive(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        assert p.signal_count > 0

    def test_no_signals_when_all_none(self, tm):
        pg = PanelGenerator(_make_metrics(), tm)
        p = pg.panel_bearish()
        assert p.signal_count == 0

    def test_new_mode_title(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(mode="new"), tm)
        p = pg.panel_bearish()
        assert p.title == "Why Sell / Avoid"

    def test_existing_mode_title(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(mode="existing"), tm)
        p = pg.panel_bearish()
        assert p.title == "Why Sell or Trim"

    def test_existing_mode_expensive_valuation_flag(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(mode="existing"), tm)
        p = pg.panel_bearish()
        text = " ".join(p.bullets)
        assert "re-rated" in text or "multiple expansion" in text

    def test_charts_are_figures(self, tm):
        pg = PanelGenerator(_make_bearish_metrics(), tm)
        p = pg.panel_bearish()
        for c in p.charts:
            assert isinstance(c, go.Figure)


# ── Panel: dividends ──────────────────────────────────────────────────────────

class TestPanelDividends:
    def test_non_payer_bullet(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = False
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "does not currently pay a dividend" in text

    def test_non_payer_with_buybacks(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = False
        m.q4.buybacks_ttm = 2_000_000_000.0
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "buybacks" in text.lower()

    def test_payer_shows_yield(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.025
        m.q4.dividend_per_share_ttm = 2.50
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "Dividend yield" in text

    def test_safe_payout_ratio(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.03
        m.q4.dividend_per_share_ttm = 3.0
        m.q4.payout_ratio = tm.get("dividends.payout_ratio_safe") - 0.05
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "safe" in text

    def test_dangerous_payout_ratio(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.05
        m.q4.dividend_per_share_ttm = 5.0
        m.q4.payout_ratio = 0.95  # well above warning threshold
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "DANGEROUSLY HIGH" in text

    def test_payer_consecutive_years(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.02
        m.q4.dividend_per_share_ttm = 2.0
        m.q4.consecutive_dividend_years = 10
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "10" in text

    def test_fcf_well_covered(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.02
        m.q4.dividend_per_share_ttm = 2.0
        m.q4.fcf_dividend_coverage = tm.get("dividends.fcf_coverage_strong") + 1.0
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "well covered by FCF" in text

    def test_fcf_not_covered(self, tm):
        m = _make_metrics()
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.02
        m.q4.dividend_per_share_ttm = 2.0
        m.q4.fcf_dividend_coverage = 0.5  # < 1, not covered
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "NOT covered by FCF" in text

    def test_existing_dividends_received(self, tm):
        m = _make_metrics(mode="existing")
        m.q4.is_dividend_payer = True
        m.q4.dividend_yield = 0.02
        m.q4.dividend_per_share_ttm = 2.0
        m.q4.position_dividends_received = 1200.0
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        text = " ".join(p.bullets)
        assert "1,200" in text or "dividends received" in text.lower()

    def test_returns_chart(self, tm):
        m = _make_metrics()
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        assert len(p.charts) > 0
        assert isinstance(p.charts[0], go.Figure)

    def test_raw_data_has_is_dividend_payer(self, tm):
        m = _make_metrics()
        pg = PanelGenerator(m, tm)
        p = pg.panel_dividends()
        assert "is_dividend_payer" in p.raw_data


# ── Panel: portfolio risk (Existing Holding only) ─────────────────────────────

class TestPanelPortfolioRisk:
    def test_new_mode_returns_none(self, tm):
        pg = PanelGenerator(_make_metrics(mode="new"), tm)
        assert pg.panel_portfolio_risk() is None

    def test_existing_mode_returns_panel(self, tm):
        pg = PanelGenerator(_make_metrics(mode="existing"), tm)
        p = pg.panel_portfolio_risk()
        assert isinstance(p, Panel)

    def test_position_value_bullet(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.position_value_now = 50_000.0
        m.portfolio.shares_owned = 100.0
        m.q6.current_price = 500.0
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "position value" in text.lower()

    def test_total_cost_bullet(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.total_cost = 40_000.0
        m.portfolio.cost_basis = 400.0
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "cost basis" in text.lower()

    def test_gain_bullet_positive(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.total_gain_loss_dollars = 10_000.0
        m.portfolio.total_gain_loss_pct = 0.25
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "gain" in text.lower() or "▲" in text

    def test_gain_bullet_negative_shows_loss(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.total_gain_loss_dollars = -5_000.0
        m.portfolio.total_gain_loss_pct = -0.10
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "loss" in text.lower() or "▼" in text

    def test_severely_overweight_flag(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.position_weight = 0.20  # above severe (0.15)
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        assert len(p.flags) > 0
        flag_text = " ".join(p.flags)
        assert "severely overweight" in flag_text

    def test_overweight_flag(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.position_weight = 0.12  # above max (0.10)
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        assert len(p.flags) > 0

    def test_at_target_no_flag(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.position_weight = 0.05  # exactly target
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "at target" in text
        assert len(p.flags) == 0

    def test_holding_period_bullet(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.days_held = 365
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "365 days" in text

    def test_annualized_return_bullet(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.annualized_return = 0.18
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "Annualized return" in text

    def test_delta_table_rendered(self, tm):
        m = _make_metrics(mode="existing")
        m.q2.revenue_cagr_at_purchase = 0.10
        m.q2.revenue_1yr_cagr = 0.20
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        text = " ".join(p.bullets)
        assert "Revenue Growth" in text

    def test_chart_rendered_when_cost_and_value_present(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.total_cost = 40_000.0
        m.portfolio.position_value_now = 50_000.0
        m.portfolio.shares_owned = 100.0          # required for position bullet format
        m.q6.current_price = 500.0
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        assert len(p.charts) > 0

    def test_raw_data_has_position_weight(self, tm):
        m = _make_metrics(mode="existing")
        m.portfolio.position_weight = 0.07
        pg = PanelGenerator(m, tm)
        p = pg.panel_portfolio_risk()
        assert "position_weight" in p.raw_data
        assert p.raw_data["position_weight"] == pytest.approx(0.07)


# ── Panel: ML predictions ─────────────────────────────────────────────────────

class TestPanelML:
    def test_returns_panel(self, tm):
        """panel_ml_predictions should always return a Panel (no crash)."""
        m = _make_bullish_metrics()
        pg = PanelGenerator(m, tm)
        p = pg.panel_ml_predictions()
        assert isinstance(p, Panel)

    def test_bullets_not_empty(self, tm):
        """At least one bullet is always produced (disclaimer or result)."""
        m = _make_bullish_metrics()
        pg = PanelGenerator(m, tm)
        p = pg.panel_ml_predictions()
        assert len(p.bullets) > 0


# ── Panel: all_panels convenience check ──────────────────────────────────────

class TestAllPanelsCombinations:
    """Smoke tests: every mode × every panel combination must not raise."""

    @pytest.mark.parametrize("mode", ["new", "existing"])
    def test_bullish_no_crash(self, tm, mode):
        pg = PanelGenerator(_make_bullish_metrics(mode), tm)
        assert isinstance(pg.panel_bullish(), Panel)

    @pytest.mark.parametrize("mode", ["new", "existing"])
    def test_hold_no_crash(self, tm, mode):
        pg = PanelGenerator(_make_hold_metrics(mode), tm)
        assert isinstance(pg.panel_hold(), Panel)

    @pytest.mark.parametrize("mode", ["new", "existing"])
    def test_bearish_no_crash(self, tm, mode):
        pg = PanelGenerator(_make_bearish_metrics(mode), tm)
        assert isinstance(pg.panel_bearish(), Panel)

    @pytest.mark.parametrize("mode", ["new", "existing"])
    def test_dividends_no_crash(self, tm, mode):
        pg = PanelGenerator(_make_metrics(mode), tm)
        assert isinstance(pg.panel_dividends(), Panel)

    def test_portfolio_risk_new_is_none(self, tm):
        pg = PanelGenerator(_make_metrics(mode="new"), tm)
        assert pg.panel_portfolio_risk() is None

    def test_portfolio_risk_existing_is_panel(self, tm):
        pg = PanelGenerator(_make_metrics(mode="existing"), tm)
        assert isinstance(pg.panel_portfolio_risk(), Panel)
