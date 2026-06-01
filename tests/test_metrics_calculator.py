"""
tests/test_metrics_calculator.py — Unit tests for src/metrics_calculator.py.

Tests cover:
  - AllMetrics and all sub-dataclasses (construction, defaults)
  - _compute_decision_confidence (the most independently testable method)
  - _q1_scale cap-category brackets
  - overall_score computation
  - PortfolioContext defaults

MetricsCalculator requires a StockData object that makes real API calls, so we
mock it with unittest.mock.MagicMock everywhere the calculator is instantiated
directly.  For dataclass-only tests we just construct the dataclasses directly.
"""

from datetime import date
from unittest.mock import MagicMock, patch
from typing import Any

import pandas as pd
import pytest

from src.metrics_calculator import (
    AllMetrics,
    Q1Scale, Q2Growth, Q3Profitability,
    Q4CashFlow, Q5BalanceSheet, Q6Valuation,
    PortfolioContext,
    MetricsCalculator,
)


# ── Mock StockData factory ─────────────────────────────────────────────────────

def _make_stock(
    ticker: str = "MOCK",
    info: dict = None,
    days_since_filing: int = 30,
    history_empty: bool = False,
) -> MagicMock:
    """
    Returns a MagicMock that behaves like a StockData object.
    All expensive attributes are replaced with safe test stubs.
    """
    stock = MagicMock()
    stock.ticker = ticker
    stock.info = info if info is not None else {
        "longName": "Mock Corp",
        "marketCap": 5_000_000_000,
        "totalRevenue": 1_000_000_000,
        "operatingMargins": 0.20,
        "profitMargins": 0.15,
        "grossMargins": 0.50,
        "returnOnEquity": 0.18,
        "returnOnAssets": 0.10,
        "trailingEps": 5.0,
        "forwardEps": 5.5,
        "earningsGrowth": 0.12,
        "revenueGrowth": 0.10,
        "trailingPE": 20.0,
        "forwardPE": 18.0,
        "enterpriseValue": 4_800_000_000,
        "bookValue": 15.0,
        "priceToBook": 2.0,
        "currentPrice": 100.0,
        "sharesOutstanding": 100_000_000,
        "operatingCashflow": 200_000_000,
        "freeCashflow": 150_000_000,
        "ebitda": 250_000_000,
        "trailingAnnualDividendYield": 0.02,
        "trailingAnnualDividendRate": 2.0,
        "payoutRatio": 0.35,
        "dividendYield": 0.02,
        "sector": "Technology",
        "industry": "Software",
        "country": "USA",
        "exchange": "NASDAQ",
        "fullTimeEmployees": 10_000,
    }
    stock.fetch_timestamp = "2025-01-01T00:00:00"
    stock.staleness_label.return_value = "Current"
    stock.days_since_last_filing.return_value = days_since_filing

    # Minimal DataFrames — empty but not None
    stock.income_stmt = pd.DataFrame()
    stock.balance_sheet = pd.DataFrame()
    stock.cash_flow = pd.DataFrame()
    stock.dividends = pd.Series(dtype=float)

    # Price history — empty by default
    if history_empty:
        stock.history = pd.DataFrame()
    else:
        dates = pd.date_range("2023-01-01", periods=250, freq="B")
        stock.history = pd.DataFrame(
            {"Close": [100.0 + i * 0.01 for i in range(250)]},
            index=dates,
        )

    return stock


# ── AllMetrics dataclass ───────────────────────────────────────────────────────

class TestAllMetricsDataclass:
    def test_construction_with_required_fields(self):
        m = AllMetrics(ticker="AAPL", company_name="Apple Inc.", mode="new")
        assert m.ticker == "AAPL"
        assert m.company_name == "Apple Inc."
        assert m.mode == "new"

    def test_sub_dataclass_defaults(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        assert isinstance(m.q1, Q1Scale)
        assert isinstance(m.q2, Q2Growth)
        assert isinstance(m.q3, Q3Profitability)
        assert isinstance(m.q4, Q4CashFlow)
        assert isinstance(m.q5, Q5BalanceSheet)
        assert isinstance(m.q6, Q6Valuation)
        assert isinstance(m.portfolio, PortfolioContext)

    def test_default_scores_are_empty_dict(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        assert m.scores == {}

    def test_default_overall_score_is_zero(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        assert m.overall_score == 0.0

    def test_default_confidence_is_100(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        assert m.decision_confidence == 100.0

    def test_default_confidence_flags_empty(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        assert m.decision_confidence_flags == []

    def test_scores_mutation(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        m.scores = {"Q1": 8.0, "Q2": 7.5, "Q3": 6.0, "Q4": 9.0, "Q5": 7.0, "Q6": 5.5}
        m.overall_score = sum(m.scores.values()) / 6
        assert m.overall_score == pytest.approx(7.166, rel=1e-2)

    def test_mode_new(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        assert m.mode == "new"

    def test_mode_existing(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="existing")
        assert m.mode == "existing"

    def test_independent_sub_dataclasses(self):
        """Two AllMetrics instances should not share sub-dataclass objects."""
        m1 = AllMetrics(ticker="A", company_name="A Corp", mode="new")
        m2 = AllMetrics(ticker="B", company_name="B Corp", mode="new")
        m1.q2.revenue_1yr_cagr = 0.25
        assert m2.q2.revenue_1yr_cagr is None  # no shared state


# ── Sub-dataclass defaults ─────────────────────────────────────────────────────

class TestSubDataclassDefaults:
    def test_q1_defaults(self):
        q = Q1Scale()
        assert q.market_cap is None
        assert q.cap_category == "Unknown"

    def test_q2_defaults(self):
        q = Q2Growth()
        assert q.revenue_1yr_cagr is None
        assert q.growth_trend == "Unknown"

    def test_q3_defaults(self):
        q = Q3Profitability()
        assert q.operating_margin is None
        assert q.margin_trend == "Unknown"

    def test_q4_defaults(self):
        q = Q4CashFlow()
        assert q.is_dividend_payer is False
        assert q.last_dividend_change == "N/A"

    def test_q5_defaults(self):
        q = Q5BalanceSheet()
        assert q.debt_to_ebitda is None
        assert q.financial_risk_trend == "Unknown"

    def test_q6_defaults(self):
        q = Q6Valuation()
        assert q.current_price is None
        assert q.valuation_change == "Unknown"

    def test_portfolio_defaults(self):
        p = PortfolioContext()
        assert p.purchase_date is None
        assert p.cost_basis is None
        assert p.shares_owned is None


# ── Decision confidence ────────────────────────────────────────────────────────

class TestDecisionConfidence:
    """
    Test _compute_decision_confidence by constructing a calculator with a
    mocked StockData and a pre-built AllMetrics.
    """

    def _make_calc(self, mode="new", **kwargs) -> MetricsCalculator:
        stock = _make_stock(**kwargs)
        return MetricsCalculator(stock, mode=mode)

    def _full_metrics(self, mode="new") -> AllMetrics:
        """AllMetrics with all valuation fields populated — clean baseline."""
        m = AllMetrics(ticker="T", company_name="Test", mode=mode)
        m.q6.pe_ratio = 20.0
        m.q6.forward_pe = 18.0
        m.q6.ev_ebitda = 12.0
        m.q6.price_to_book = 2.0
        m.q2.revenue_1yr_cagr = 0.10
        m.q2.revenue_3yr_cagr = 0.08
        m.q2.revenue_5yr_cagr = 0.09
        return m

    def test_perfect_data_near_100(self):
        """Fresh data, no gaps → full confidence (only days_since deductions)."""
        calc = self._make_calc(days_since_filing=30, history_empty=False)
        conf, flags = calc._compute_decision_confidence(self._full_metrics())
        assert conf == pytest.approx(100.0)
        assert flags == []

    def test_empty_history_deducts_15(self):
        calc = self._make_calc(days_since_filing=30, history_empty=True)
        conf, flags = calc._compute_decision_confidence(self._full_metrics())
        assert conf == pytest.approx(85.0)
        assert any("history" in f.lower() for f in flags)

    def test_stale_filing_121_days_deducts_10(self):
        """121 days: (121-120)//30 * 3 = 0 * 3 = 0 additional; deduction = min(25, 10+0) = 10."""
        calc = self._make_calc(days_since_filing=121)
        conf, flags = calc._compute_decision_confidence(self._full_metrics())
        assert conf == pytest.approx(90.0)
        assert any("days old" in f for f in flags)

    def test_stale_filing_150_days_deducts_13(self):
        """150 days: (150-120)//30 * 3 = 1*3 = 3; deduction = min(25, 10+3) = 13."""
        calc = self._make_calc(days_since_filing=150)
        conf, flags = calc._compute_decision_confidence(self._full_metrics())
        assert conf == pytest.approx(87.0)

    def test_very_stale_filing_capped_at_25_deduction(self):
        """365 days: (365-120)//30 * 3 = 8*3 = 24; deduction = min(25, 10+24) = 25."""
        calc = self._make_calc(days_since_filing=365)
        conf, flags = calc._compute_decision_confidence(self._full_metrics())
        assert conf == pytest.approx(75.0)

    def test_unknown_filing_date_deducts_10(self):
        calc = self._make_calc()
        calc.stock.days_since_last_filing.return_value = None  # unknown
        conf, flags = calc._compute_decision_confidence(self._full_metrics())
        assert conf == pytest.approx(90.0)
        assert any("unknown" in f.lower() for f in flags)

    def test_three_missing_valuation_multiples_deducts_12(self):
        m = self._full_metrics()
        m.q6.pe_ratio = None
        m.q6.forward_pe = None
        m.q6.ev_ebitda = None
        calc = self._make_calc(days_since_filing=30)
        conf, flags = calc._compute_decision_confidence(m)
        assert conf == pytest.approx(88.0)
        assert any("most valuation" in f.lower() for f in flags)

    def test_two_missing_valuation_multiples_deducts_6(self):
        m = self._full_metrics()
        m.q6.pe_ratio = None
        m.q6.forward_pe = None
        calc = self._make_calc(days_since_filing=30)
        conf, flags = calc._compute_decision_confidence(m)
        assert conf == pytest.approx(94.0)
        assert any("several" in f.lower() for f in flags)

    def test_one_missing_valuation_no_deduction(self):
        m = self._full_metrics()
        m.q6.pe_ratio = None  # only 1 missing — no deduction
        calc = self._make_calc(days_since_filing=30)
        conf, flags = calc._compute_decision_confidence(m)
        assert conf == pytest.approx(100.0)

    def test_no_revenue_trend_deducts_8(self):
        m = self._full_metrics()
        m.q2.revenue_3yr_cagr = None
        m.q2.revenue_5yr_cagr = None
        calc = self._make_calc(days_since_filing=30)
        conf, flags = calc._compute_decision_confidence(m)
        assert conf == pytest.approx(92.0)
        assert any("revenue" in f.lower() for f in flags)

    def test_existing_no_cost_basis_deducts_10(self):
        calc = MetricsCalculator(
            _make_stock(days_since_filing=30),
            mode="existing",
            cost_basis=None,
            purchase_date=date(2023, 1, 1),
            total_portfolio_value=100_000.0,
        )
        conf, flags = calc._compute_decision_confidence(self._full_metrics("existing"))
        assert conf == pytest.approx(90.0)
        assert any("cost basis" in f.lower() for f in flags)

    def test_existing_no_purchase_date_deducts_5(self):
        calc = MetricsCalculator(
            _make_stock(days_since_filing=30),
            mode="existing",
            cost_basis=50.0,
            purchase_date=None,
            total_portfolio_value=100_000.0,
        )
        conf, flags = calc._compute_decision_confidence(self._full_metrics("existing"))
        assert conf == pytest.approx(95.0)
        assert any("purchase date" in f.lower() for f in flags)

    def test_existing_no_portfolio_value_deducts_5(self):
        calc = MetricsCalculator(
            _make_stock(days_since_filing=30),
            mode="existing",
            cost_basis=50.0,
            purchase_date=date(2023, 1, 1),
            total_portfolio_value=None,
        )
        conf, flags = calc._compute_decision_confidence(self._full_metrics("existing"))
        assert conf == pytest.approx(95.0)
        assert any("portfolio value" in f.lower() for f in flags)

    def test_existing_all_fields_missing_compound_deduction(self):
        """No cost_basis + no purchase_date + no portfolio_value = -20."""
        calc = MetricsCalculator(
            _make_stock(days_since_filing=30),
            mode="existing",
            cost_basis=None,
            purchase_date=None,
            total_portfolio_value=None,
        )
        conf, flags = calc._compute_decision_confidence(self._full_metrics("existing"))
        assert conf == pytest.approx(80.0)  # 100 - 10 - 5 - 5

    def test_new_mode_no_portfolio_deductions(self):
        """In 'new' mode no portfolio deductions should be applied."""
        calc = MetricsCalculator(
            _make_stock(days_since_filing=30),
            mode="new",
        )
        conf, flags = calc._compute_decision_confidence(self._full_metrics("new"))
        assert conf == pytest.approx(100.0)
        assert not any("cost basis" in f.lower() for f in flags)

    def test_confidence_clamped_to_zero_minimum(self):
        """With every possible deduction applied, result should not go below 0."""
        calc = MetricsCalculator(
            _make_stock(days_since_filing=365, history_empty=True),
            mode="existing",
            cost_basis=None,
            purchase_date=None,
            total_portfolio_value=None,
        )
        m = AllMetrics(ticker="X", company_name="X", mode="existing")
        # All valuation multiples None → -12
        # revenue trend None → -8
        conf, flags = calc._compute_decision_confidence(m)
        assert conf >= 0.0

    def test_confidence_never_exceeds_100(self):
        calc = self._make_calc(days_since_filing=30)
        conf, _ = calc._compute_decision_confidence(self._full_metrics())
        assert conf <= 100.0

    def test_flags_are_strings(self):
        calc = self._make_calc(days_since_filing=365, history_empty=True)
        m = self._full_metrics()
        m.q6.pe_ratio = None
        m.q6.forward_pe = None
        m.q6.ev_ebitda = None
        _, flags = calc._compute_decision_confidence(m)
        assert all(isinstance(f, str) for f in flags)
        assert len(flags) > 0


# ── Cap category ───────────────────────────────────────────────────────────────

class TestCapCategory:
    """Test _q1_scale cap-category classification against known brackets."""

    @pytest.mark.parametrize("market_cap,expected", [
        (100_000_000,  "Micro Cap"),  # < 300M
        (299_999_999,  "Micro Cap"),
        (300_000_000,  "Small Cap"),  # 300M–2B
        (1_500_000_000, "Small Cap"),
        (2_000_000_000, "Mid Cap"),   # 2B–10B
        (9_999_999_999, "Mid Cap"),
        (10_000_000_000, "Large Cap"), # 10B–200B
        (100_000_000_000, "Large Cap"),
        (200_000_000_000, "Mega Cap"), # ≥200B
        (3_000_000_000_000, "Mega Cap"),
    ])
    def test_cap_category_bracket(self, market_cap, expected):
        stock = _make_stock()
        stock.info["marketCap"] = market_cap
        calc = MetricsCalculator(stock, mode="new")
        q = calc._q1_scale()
        assert q.cap_category == expected

    def test_zero_market_cap_is_micro(self):
        stock = _make_stock()
        stock.info["marketCap"] = 0
        calc = MetricsCalculator(stock, mode="new")
        q = calc._q1_scale()
        assert q.cap_category == "Micro Cap"

    def test_none_market_cap_treated_as_zero(self):
        stock = _make_stock()
        stock.info["marketCap"] = None
        calc = MetricsCalculator(stock, mode="new")
        q = calc._q1_scale()
        assert q.cap_category == "Micro Cap"

    def test_q1_sector_and_industry(self):
        stock = _make_stock()
        stock.info["sector"] = "Healthcare"
        stock.info["industry"] = "Biotechnology"
        calc = MetricsCalculator(stock, mode="new")
        q = calc._q1_scale()
        assert q.sector == "Healthcare"
        assert q.industry == "Biotechnology"

    def test_q1_employee_count(self):
        stock = _make_stock()
        stock.info["fullTimeEmployees"] = 75_000
        calc = MetricsCalculator(stock, mode="new")
        q = calc._q1_scale()
        assert q.employee_count == 75_000

    def test_q1_data_as_of_set(self):
        stock = _make_stock()
        calc = MetricsCalculator(stock, mode="new")
        q = calc._q1_scale()
        assert q.data_as_of == "Current"  # staleness_label returns "Current"


# ── Overall score computation ─────────────────────────────────────────────────

class TestOverallScore:
    def test_average_of_all_six(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        m.scores = {"Q1": 6.0, "Q2": 7.0, "Q3": 8.0, "Q4": 5.0, "Q5": 9.0, "Q6": 7.0}
        m.overall_score = sum(m.scores.values()) / len(m.scores)
        assert m.overall_score == pytest.approx(7.0)

    def test_perfect_scores(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        m.scores = {q: 10.0 for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]}
        m.overall_score = sum(m.scores.values()) / len(m.scores)
        assert m.overall_score == pytest.approx(10.0)

    def test_zero_scores(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        m.scores = {q: 0.0 for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]}
        m.overall_score = sum(m.scores.values()) / len(m.scores)
        assert m.overall_score == pytest.approx(0.0)

    def test_empty_scores_default_zero(self):
        m = AllMetrics(ticker="X", company_name="X Corp", mode="new")
        # Simulate the compute() logic with empty scores
        m.overall_score = sum(m.scores.values()) / max(len(m.scores), 1)
        assert m.overall_score == 0.0


# ── PortfolioContext ───────────────────────────────────────────────────────────

class TestPortfolioContext:
    def test_default_values(self):
        p = PortfolioContext()
        assert p.purchase_date is None
        assert p.cost_basis is None
        assert p.shares_owned is None
        assert p.total_portfolio_value is None
        assert p.position_value_now is None
        assert p.position_weight is None
        assert p.total_cost is None
        assert p.total_gain_loss_dollars is None
        assert p.total_gain_loss_pct is None
        assert p.days_held is None
        assert p.annualized_return is None

    def test_gain_set_manually(self):
        p = PortfolioContext(
            total_cost=40_000.0,
            position_value_now=50_000.0,
            total_gain_loss_dollars=10_000.0,
            total_gain_loss_pct=0.25,
            days_held=365,
            annualized_return=0.25,
        )
        assert p.total_gain_loss_pct == pytest.approx(0.25)
        assert p.annualized_return == pytest.approx(0.25)

    def test_purchase_date_field(self):
        p = PortfolioContext(purchase_date=date(2022, 6, 1))
        assert p.purchase_date == date(2022, 6, 1)

    def test_position_weight_above_one_allowed(self):
        """No validation at dataclass level — caller's responsibility."""
        p = PortfolioContext(position_weight=1.50)
        assert p.position_weight == pytest.approx(1.50)
