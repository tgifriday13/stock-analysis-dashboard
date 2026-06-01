"""
metrics_calculator.py — Computes all fundamental metrics that answer the 6 core questions.

For Existing Holding mode, every metric is computed both "now" and "at purchase date"
so the dashboard can show "now vs. when you bought" deltas.

The 6 Questions:
  Q1 — Scale & Business Reality
  Q2 — Growth Trajectory
  Q3 — Profitability & Margin Quality
  Q4 — Cash Flow Generation & Capital Efficiency
  Q5 — Balance Sheet Strength & Debt
  Q6 — Valuation & Margin of Safety
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_fetcher import StockData
from .utils import safe_float, safe_divide, safe_pct_change, fmt_currency, fmt_pct, fmt_multiple

logger = logging.getLogger("dashboard.metrics")


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Q1Scale:
    """Q1 – Scale & Business Reality"""
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    ttm_revenue: Optional[float] = None
    ttm_revenue_prior: Optional[float] = None      # at purchase date (Existing Holding)
    revenue_delta_pct: Optional[float] = None
    cap_category: str = "Unknown"                  # Micro / Small / Mid / Large / Mega
    employee_count: Optional[int] = None
    sector: str = "Unknown"
    industry: str = "Unknown"
    country: str = "Unknown"
    exchange: str = "Unknown"
    data_as_of: str = ""


@dataclass
class Q2Growth:
    """Q2 – Growth Trajectory"""
    revenue_1yr_cagr: Optional[float] = None
    revenue_3yr_cagr: Optional[float] = None
    revenue_5yr_cagr: Optional[float] = None
    earnings_1yr_growth: Optional[float] = None
    earnings_3yr_cagr: Optional[float] = None
    eps_ttm: Optional[float] = None
    eps_prior_year: Optional[float] = None
    forward_revenue_growth: Optional[float] = None   # analyst estimate
    forward_eps_growth: Optional[float] = None
    # Purchase-date deltas (Existing Holding)
    revenue_cagr_at_purchase: Optional[float] = None
    growth_trend: str = "Unknown"                    # Accelerating / Stable / Decelerating


@dataclass
class Q3Profitability:
    """Q3 – Profitability & Margin Quality"""
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    # Prior period (for trend analysis)
    gross_margin_prior: Optional[float] = None
    operating_margin_prior: Optional[float] = None
    net_margin_prior: Optional[float] = None
    # Purchase-date snapshot
    gross_margin_at_purchase: Optional[float] = None
    operating_margin_at_purchase: Optional[float] = None
    margin_trend: str = "Unknown"                    # Expanding / Stable / Contracting


@dataclass
class Q4CashFlow:
    """Q4 – Cash Flow Generation & Capital Efficiency"""
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_margin: Optional[float] = None
    fcf_per_share: Optional[float] = None
    fcf_yield: Optional[float] = None              # FCF / Market Cap
    roic: Optional[float] = None
    shares_outstanding: Optional[float] = None
    buybacks_ttm: Optional[float] = None
    # Dividends (always computed; N/A for non-payers)
    dividend_yield: Optional[float] = None
    dividend_per_share_ttm: Optional[float] = None
    total_dividends_paid_ttm: Optional[float] = None
    payout_ratio: Optional[float] = None
    fcf_dividend_coverage: Optional[float] = None  # FCF / Total Dividends
    consecutive_dividend_years: Optional[int] = None
    dividend_5yr_growth_cagr: Optional[float] = None
    last_dividend_change: str = "N/A"              # "Increase", "Cut", "Stable", "N/A"
    is_dividend_payer: bool = False
    # Existing Holding — dividends received
    position_dividends_received: Optional[float] = None
    # Purchase-date snapshot
    fcf_at_purchase: Optional[float] = None
    fcf_margin_at_purchase: Optional[float] = None


@dataclass
class Q5BalanceSheet:
    """Q5 – Balance Sheet Strength & Debt"""
    total_debt: Optional[float] = None
    net_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_equity: Optional[float] = None
    ebitda: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    net_debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None      # EBIT / Interest expense
    altman_z_score: Optional[float] = None         # Simple proxy
    # Purchase-date snapshot
    debt_to_ebitda_at_purchase: Optional[float] = None
    financial_risk_trend: str = "Unknown"          # Improving / Stable / Deteriorating


@dataclass
class Q6Valuation:
    """Q6 – Valuation & Margin of Safety"""
    current_price: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_fcf: Optional[float] = None
    fcf_yield: Optional[float] = None
    intrinsic_value_dcf: Optional[float] = None   # Simple DCF estimate
    margin_of_safety: Optional[float] = None       # (IV - Price) / IV
    # Purchase-date snapshot (Existing Holding)
    price_at_purchase: Optional[float] = None
    pe_at_purchase: Optional[float] = None
    ev_ebitda_at_purchase: Optional[float] = None
    valuation_change: str = "Unknown"              # Cheaper / Similar / More Expensive
    # Gain/loss (Existing Holding)
    unrealized_gain_loss: Optional[float] = None   # in dollars
    unrealized_gain_loss_pct: Optional[float] = None
    total_return_including_dividends: Optional[float] = None


@dataclass
class PortfolioContext:
    """Extra fields populated only in Existing Holding mode."""
    purchase_date: Optional[date] = None
    cost_basis: Optional[float] = None             # per share
    shares_owned: Optional[float] = None
    total_portfolio_value: Optional[float] = None
    position_value_now: Optional[float] = None
    position_weight: Optional[float] = None        # position / portfolio
    total_cost: Optional[float] = None             # cost_basis * shares
    total_gain_loss_dollars: Optional[float] = None
    total_gain_loss_pct: Optional[float] = None
    days_held: Optional[int] = None
    annualized_return: Optional[float] = None


@dataclass
class AllMetrics:
    """Top-level container passed to all downstream modules."""
    ticker: str
    company_name: str
    mode: str                                       # "new" | "existing"
    q1: Q1Scale = field(default_factory=Q1Scale)
    q2: Q2Growth = field(default_factory=Q2Growth)
    q3: Q3Profitability = field(default_factory=Q3Profitability)
    q4: Q4CashFlow = field(default_factory=Q4CashFlow)
    q5: Q5BalanceSheet = field(default_factory=Q5BalanceSheet)
    q6: Q6Valuation = field(default_factory=Q6Valuation)
    portfolio: PortfolioContext = field(default_factory=PortfolioContext)
    staleness_label: str = ""
    fetch_timestamp: str = ""
    notes: str = ""
    # Summary scores (0–10 per question, average → overall)
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    # Confidence in the decision given data completeness (0–100)
    decision_confidence: float = 100.0
    decision_confidence_flags: List[str] = field(default_factory=list)


# ── Main calculator ───────────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Compute all metrics from a StockData object.
    Handles both 'new' and 'existing' ownership modes.
    """

    def __init__(
        self,
        stock: StockData,
        mode: str = "new",
        purchase_date: Optional[date] = None,
        cost_basis: Optional[float] = None,
        shares_owned: Optional[float] = None,
        total_portfolio_value: Optional[float] = None,
        notes: str = "",
    ):
        self.stock = stock
        self.mode = mode.lower()  # "new" or "existing"
        self.purchase_date = purchase_date
        self.cost_basis = cost_basis
        self.shares_owned = shares_owned
        self.total_portfolio_value = total_portfolio_value
        self.notes = notes

        self.info = stock.info
        self.income = stock.income_stmt
        self.balance = stock.balance_sheet
        self.cashflow = stock.cash_flow
        self.hist = stock.history
        self.divs = stock.dividends

    def compute(self) -> AllMetrics:
        metrics = AllMetrics(
            ticker=self.stock.ticker,
            company_name=self.info.get("longName", self.stock.ticker),
            mode=self.mode,
            staleness_label=self.stock.staleness_label(),
            fetch_timestamp=self.stock.fetch_timestamp,
            notes=self.notes,
        )
        metrics.q1 = self._q1_scale()
        metrics.q2 = self._q2_growth()
        metrics.q3 = self._q3_profitability()
        metrics.q4 = self._q4_cashflow()
        metrics.q5 = self._q5_balance_sheet()
        metrics.q6 = self._q6_valuation()
        if self.mode == "existing":
            metrics.portfolio = self._portfolio_context(metrics)
        metrics.scores = self._compute_scores(metrics)
        metrics.overall_score = sum(metrics.scores.values()) / max(len(metrics.scores), 1)
        confidence, flags = self._compute_decision_confidence(metrics)
        metrics.decision_confidence = confidence
        metrics.decision_confidence_flags = flags
        return metrics

    # ── Decision confidence ───────────────────────────────────────────────────

    def _compute_decision_confidence(
        self, metrics: "AllMetrics"
    ) -> Tuple[float, List[str]]:
        """
        Return (confidence_pct, flags).
        Starts at 100 and deducts for each data-quality issue found.
        """
        score = 100.0
        flags: List[str] = []

        # Price history missing
        if self.hist is None or (hasattr(self.hist, "empty") and self.hist.empty):
            score -= 15
            flags.append("Price history unavailable — charts may be incomplete")

        # Stale filing data
        days_old = self.stock.days_since_last_filing()
        if days_old is not None and days_old > 120:
            deduction = min(25, 10 + (days_old - 120) // 30 * 3)
            score -= deduction
            flags.append(
                f"Filing data is {days_old} days old — verify figures before acting"
            )
        elif days_old is None:
            score -= 10
            flags.append("Filing date unknown — data freshness cannot be confirmed")

        # Valuation data incomplete
        q6 = metrics.q6
        val_missing = sum(
            1 for v in [q6.pe_ratio, q6.forward_pe, q6.ev_ebitda, q6.price_to_book]
            if v is None
        )
        if val_missing >= 3:
            score -= 12
            flags.append("Most valuation multiples are unavailable")
        elif val_missing >= 2:
            score -= 6
            flags.append("Several valuation multiples are missing")

        # Historical revenue trend unavailable (need at least 2 years)
        q2 = metrics.q2
        if q2.revenue_3yr_cagr is None and q2.revenue_5yr_cagr is None:
            score -= 8
            flags.append("Multi-year revenue trend data unavailable")

        # Existing mode — missing portfolio context
        if self.mode == "existing":
            if self.cost_basis is None:
                score -= 10
                flags.append("Cost basis not provided — gain/loss analysis unavailable")
            if self.purchase_date is None:
                score -= 5
                flags.append("Purchase date not provided — holding period unknown")
            if self.total_portfolio_value is None:
                score -= 5
                flags.append("Total portfolio value not provided — position sizing unavailable")

        confidence = max(0.0, min(100.0, score))
        return confidence, flags

    # ── Q1: Scale ─────────────────────────────────────────────────────────────

    def _q1_scale(self) -> Q1Scale:
        q = Q1Scale()
        i = self.info

        q.market_cap = safe_float(i.get("marketCap"))
        q.enterprise_value = safe_float(i.get("enterpriseValue"))
        q.ttm_revenue = safe_float(i.get("totalRevenue"))
        q.employee_count = i.get("fullTimeEmployees")
        q.sector = i.get("sector", "Unknown")
        q.industry = i.get("industry", "Unknown")
        q.country = i.get("country", "Unknown")
        q.exchange = i.get("exchange", "Unknown")

        # Cap category
        mc = q.market_cap or 0
        if mc < 3e8:
            q.cap_category = "Micro Cap"
        elif mc < 2e9:
            q.cap_category = "Small Cap"
        elif mc < 1e10:
            q.cap_category = "Mid Cap"
        elif mc < 2e11:
            q.cap_category = "Large Cap"
        else:
            q.cap_category = "Mega Cap"

        # Revenue delta for existing holders
        if self.mode == "existing" and self.purchase_date and not self.income.empty:
            q.ttm_revenue_prior = self._revenue_at_date(self.purchase_date)
            q.revenue_delta_pct = safe_pct_change(q.ttm_revenue, q.ttm_revenue_prior)

        q.data_as_of = self.stock.staleness_label()
        return q

    # ── Q2: Growth ────────────────────────────────────────────────────────────

    def _q2_growth(self) -> Q2Growth:
        q = Q2Growth()
        i = self.info

        revenues = self._revenue_series()

        if len(revenues) >= 2:
            q.revenue_1yr_cagr = safe_pct_change(revenues[0], revenues[1])
        if len(revenues) >= 4:
            q.revenue_3yr_cagr = self._cagr(revenues[3], revenues[0], 3)
        if len(revenues) >= 4:
            # Use all available years (yfinance annual data typically gives 4)
            n = len(revenues) - 1
            q.revenue_5yr_cagr = self._cagr(revenues[-1], revenues[0], n)

        # EPS growth
        q.eps_ttm = safe_float(i.get("trailingEps"))
        q.eps_prior_year = safe_float(i.get("forwardEps"))  # approximate
        q.earnings_1yr_growth = safe_float(i.get("earningsGrowth"))
        q.earnings_3yr_cagr = safe_float(i.get("earningsQuarterlyGrowth"))

        # Forward estimates
        q.forward_revenue_growth = safe_float(i.get("revenueGrowth"))
        q.forward_eps_growth = safe_float(i.get("earningsGrowth"))

        # Purchase-date snapshot
        if self.mode == "existing" and self.purchase_date and len(revenues) >= 2:
            q.revenue_cagr_at_purchase = self._revenue_cagr_at_date(self.purchase_date)

        # Trend assessment
        cagrs = [x for x in [q.revenue_1yr_cagr, q.revenue_3yr_cagr, q.revenue_5yr_cagr] if x is not None]
        if len(cagrs) >= 2:
            if cagrs[0] > cagrs[-1] + 0.02:
                q.growth_trend = "Accelerating"
            elif cagrs[0] < cagrs[-1] - 0.02:
                q.growth_trend = "Decelerating"
            else:
                q.growth_trend = "Stable"

        return q

    # ── Q3: Profitability ─────────────────────────────────────────────────────

    def _q3_profitability(self) -> Q3Profitability:
        q = Q3Profitability()
        i = self.info

        q.gross_margin = safe_float(i.get("grossMargins"))
        q.operating_margin = safe_float(i.get("operatingMargins"))
        q.net_margin = safe_float(i.get("profitMargins"))
        q.ebitda_margin = safe_divide(
            safe_float(i.get("ebitda")), safe_float(i.get("totalRevenue"))
        )
        q.roe = safe_float(i.get("returnOnEquity"))
        q.roa = safe_float(i.get("returnOnAssets"))

        # ROIC = NOPAT / Invested Capital (approximate)
        nopat = self._get_income_row("Net Income", "Net Income From Continuing Operation Net Minority Interest")
        invested_capital = self._compute_invested_capital()
        q.roic = safe_divide(nopat, invested_capital)

        # Prior period margins (1 year ago column)
        if not self.income.empty and len(self.income.columns) >= 2:
            rev_now = self._get_income_col(0, "Total Revenue")
            rev_prior = self._get_income_col(1, "Total Revenue")
            gp_now = self._get_income_col(0, "Gross Profit")
            gp_prior = self._get_income_col(1, "Gross Profit")
            op_now = self._get_income_col(0, "Operating Income")
            op_prior = self._get_income_col(1, "Operating Income")
            ni_now = self._get_income_col(0, "Net Income")
            ni_prior = self._get_income_col(1, "Net Income")

            q.gross_margin_prior = safe_divide(gp_prior, rev_prior)
            q.operating_margin_prior = safe_divide(op_prior, rev_prior)
            q.net_margin_prior = safe_divide(ni_prior, rev_prior)

        # Purchase-date snapshot
        if self.mode == "existing" and self.purchase_date:
            q.gross_margin_at_purchase = self._margin_at_date(self.purchase_date, "gross")
            q.operating_margin_at_purchase = self._margin_at_date(self.purchase_date, "operating")

        # Margin trend
        if q.gross_margin is not None and q.gross_margin_prior is not None:
            delta = q.gross_margin - q.gross_margin_prior
            if delta > 0.01:
                q.margin_trend = "Expanding"
            elif delta < -0.01:
                q.margin_trend = "Contracting"
            else:
                q.margin_trend = "Stable"

        return q

    # ── Q4: Cash Flow ─────────────────────────────────────────────────────────

    def _q4_cashflow(self) -> Q4CashFlow:
        q = Q4CashFlow()
        i = self.info

        # Operating CF — prefer info (fast), fall back to statement
        q.operating_cash_flow = safe_float(i.get("operatingCashflow")) or \
            self._get_cf_row("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")

        # CapEx — NOT in info dict; always pull from cashflow statement
        capex_raw = self._get_cf_row("Capital Expenditure", "Purchase Of PPE")
        q.capital_expenditure = abs(capex_raw) if capex_raw is not None else None

        # FCF — prefer direct row from cashflow statement (most accurate)
        fcf_direct = self._get_cf_row("Free Cash Flow")
        q.free_cash_flow = fcf_direct or safe_float(i.get("freeCashflow")) or (
            (q.operating_cash_flow - q.capital_expenditure)
            if q.operating_cash_flow is not None and q.capital_expenditure is not None
            else None
        )

        # FCF margin
        rev = safe_float(i.get("totalRevenue"))
        q.fcf_margin = safe_divide(q.free_cash_flow, rev)

        # FCF per share & yield
        q.shares_outstanding = safe_float(i.get("sharesOutstanding"))
        q.fcf_per_share = safe_divide(q.free_cash_flow, q.shares_outstanding)
        mc = safe_float(i.get("marketCap"))
        q.fcf_yield = safe_divide(q.free_cash_flow, mc)

        # ROIC (cross-ref with Q3)
        nopat = self._get_income_row("Net Income", "Net Income From Continuing Operation Net Minority Interest")
        ic = self._compute_invested_capital()
        q.roic = safe_divide(nopat, ic)

        # Buybacks — yfinance uses "Repurchase Of Capital Stock"
        q.buybacks_ttm = self._get_cf_row("Repurchase Of Capital Stock", "Common Stock Payments")
        if q.buybacks_ttm is not None:
            q.buybacks_ttm = abs(q.buybacks_ttm)

        # Dividends — use trailingAnnualDividendRate for the TTM per-share amount.
        # NOTE: yfinance 'dividendYield' returns a percentage value (e.g. 0.38 = 0.38%),
        # while 'trailingAnnualDividendYield' returns a proper decimal (0.00377).
        # Always prefer trailingAnnualDividendYield for consistent decimal form.
        raw_yield = safe_float(i.get("trailingAnnualDividendYield")) or safe_float(i.get("dividendYield"))
        # If the value is > 0.10 it was returned as a percentage — convert to decimal
        if raw_yield and raw_yield > 0.10:
            raw_yield = raw_yield / 100.0
        q.dividend_yield = raw_yield
        q.dividend_per_share_ttm = safe_float(i.get("trailingAnnualDividendRate")) or safe_float(i.get("dividendRate"))
        q.is_dividend_payer = (q.dividend_yield is not None and q.dividend_yield > 0)

        if q.is_dividend_payer and q.shares_outstanding:
            # Total dividends paid — prefer cashflow statement, fall back to per-share * shares
            cf_divs = self._get_cf_row("Common Stock Dividend Paid", "Cash Dividends Paid")
            if cf_divs is not None:
                q.total_dividends_paid_ttm = abs(cf_divs)
            elif q.dividend_per_share_ttm:
                q.total_dividends_paid_ttm = q.dividend_per_share_ttm * q.shares_outstanding
            eps = safe_float(i.get("trailingEps"))
            if eps and eps > 0 and q.dividend_per_share_ttm:
                q.payout_ratio = safe_divide(q.dividend_per_share_ttm, eps)
            q.fcf_dividend_coverage = safe_divide(q.free_cash_flow, q.total_dividends_paid_ttm)

        # Dividend history analysis
        if not self.divs.empty:
            q.consecutive_dividend_years = self._count_consecutive_dividend_years()
            q.dividend_5yr_growth_cagr = self._dividend_cagr(years=5)
            q.last_dividend_change = self._last_dividend_change()

        # Existing Holding — dividends received on position
        if self.mode == "existing" and self.purchase_date and self.shares_owned and not self.divs.empty:
            q.position_dividends_received = self._dividends_received_since(
                self.purchase_date, self.shares_owned
            )

        # Purchase-date snapshot
        if self.mode == "existing" and self.purchase_date:
            q.fcf_at_purchase = self._fcf_at_date(self.purchase_date)
            q.fcf_margin_at_purchase = safe_divide(q.fcf_at_purchase, self._revenue_at_date(self.purchase_date))

        return q

    # ── Q5: Balance Sheet ─────────────────────────────────────────────────────

    def _q5_balance_sheet(self) -> Q5BalanceSheet:
        q = Q5BalanceSheet()
        i = self.info

        q.total_debt = safe_float(i.get("totalDebt")) or \
            self._get_balance_row("Total Debt", "Long Term Debt And Capital Lease Obligation")
        q.cash_and_equivalents = safe_float(i.get("totalCash")) or \
            self._get_balance_row("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")

        # Net debt — use balance sheet row directly if available
        net_debt_bs = self._get_balance_row("Net Debt")
        if net_debt_bs is not None:
            q.net_debt = net_debt_bs
        elif q.total_debt is not None and q.cash_and_equivalents is not None:
            q.net_debt = q.total_debt - q.cash_and_equivalents

        # Total equity — balance sheet is authoritative (bookValue in info is per-share)
        q.total_equity = self._get_balance_row("Stockholders Equity", "Common Stock Equity") or \
            self._get_balance_row("Total Equity Gross Minority Interest")

        # EBITDA — use info first, then income statement row, then EBIT + D&A
        q.ebitda = safe_float(i.get("ebitda")) or self._get_income_row("EBITDA", "Normalized EBITDA")
        if q.ebitda is None:
            ebit = self._get_income_row("EBIT", "Operating Income")
            da = self._get_cf_row("Depreciation And Amortization", "Depreciation Amortization Depletion")
            if ebit is not None and da is not None:
                q.ebitda = ebit + abs(da)

        q.debt_to_ebitda = safe_divide(q.total_debt, q.ebitda)
        q.debt_to_equity = safe_divide(q.total_debt, q.total_equity)
        q.net_debt_to_equity = safe_divide(q.net_debt, q.total_equity)

        q.current_ratio = safe_float(i.get("currentRatio"))
        q.quick_ratio = safe_float(i.get("quickRatio"))

        # Interest coverage = EBIT / Interest expense
        ebit = self._get_income_row("EBIT", "Operating Income")
        interest = self._get_income_row("Interest Expense", "Interest Expense Non Operating")
        # Fallback: some companies (e.g. Apple) net interest income rather than separate expense
        if (interest is None or interest != interest) and ebit is not None:  # NaN check
            net_int = self._get_income_row("Net Interest Income")
            if net_int is not None and net_int < 0:
                interest = abs(net_int)
        if ebit is not None and interest is not None and interest != 0:
            q.interest_coverage = abs(ebit) / abs(interest)

        # Purchase-date snapshot
        if self.mode == "existing" and self.purchase_date:
            q.debt_to_ebitda_at_purchase = self._debt_ebitda_at_date(self.purchase_date)
            if q.debt_to_ebitda is not None and q.debt_to_ebitda_at_purchase is not None:
                delta = q.debt_to_ebitda - q.debt_to_ebitda_at_purchase
                if delta < -0.3:
                    q.financial_risk_trend = "Improving"
                elif delta > 0.3:
                    q.financial_risk_trend = "Deteriorating"
                else:
                    q.financial_risk_trend = "Stable"

        return q

    # ── Q6: Valuation ─────────────────────────────────────────────────────────

    def _q6_valuation(self) -> Q6Valuation:
        q = Q6Valuation()
        i = self.info

        q.current_price = safe_float(i.get("currentPrice")) or safe_float(i.get("regularMarketPrice"))
        q.pe_ratio = safe_float(i.get("trailingPE"))
        q.forward_pe = safe_float(i.get("forwardPE"))
        q.peg_ratio = safe_float(i.get("pegRatio"))
        q.ev_ebitda = safe_float(i.get("enterpriseToEbitda"))
        q.price_to_sales = safe_float(i.get("priceToSalesTrailing12Months"))
        q.price_to_book = safe_float(i.get("priceToBook"))

        # Price / FCF
        mc = safe_float(i.get("marketCap"))
        fcf = safe_float(i.get("freeCashflow")) or self._estimate_fcf()
        q.price_to_fcf = safe_divide(mc, fcf)
        q.fcf_yield = safe_divide(fcf, mc)

        # Simple DCF intrinsic value estimate
        q.intrinsic_value_dcf = self._simple_dcf()
        if q.intrinsic_value_dcf and q.current_price and q.intrinsic_value_dcf > 0:
            q.margin_of_safety = (q.intrinsic_value_dcf - q.current_price) / q.intrinsic_value_dcf

        # Existing Holding — purchase price & valuation change
        if self.mode == "existing" and self.purchase_date:
            q.price_at_purchase = self.stock.get_price_on_date(self.purchase_date)
            if self.cost_basis:
                q.price_at_purchase = self.cost_basis  # user-supplied is authoritative
            if q.price_at_purchase and q.current_price:
                q.unrealized_gain_loss_pct = safe_pct_change(q.current_price, q.price_at_purchase)
                if self.shares_owned:
                    q.unrealized_gain_loss = (q.current_price - q.price_at_purchase) * self.shares_owned

            # Rough valuation-at-purchase (use current ratios as proxy — no perfect solution without historical data)
            q.pe_at_purchase = self._pe_at_purchase_date()
            q.ev_ebitda_at_purchase = self._ev_ebitda_at_purchase_date()

            # Valuation change narrative
            if q.pe_ratio and q.pe_at_purchase:
                ratio = q.pe_ratio / q.pe_at_purchase
                if ratio < 0.85:
                    q.valuation_change = "Cheaper"
                elif ratio > 1.15:
                    q.valuation_change = "More Expensive"
                else:
                    q.valuation_change = "Similar"

        return q

    # ── Portfolio context ─────────────────────────────────────────────────────

    def _portfolio_context(self, metrics: AllMetrics) -> PortfolioContext:
        p = PortfolioContext()
        p.purchase_date = self.purchase_date
        p.cost_basis = self.cost_basis
        p.shares_owned = self.shares_owned
        p.total_portfolio_value = self.total_portfolio_value

        if self.cost_basis and self.shares_owned:
            p.total_cost = self.cost_basis * self.shares_owned

        price = metrics.q6.current_price
        if price and self.shares_owned:
            p.position_value_now = price * self.shares_owned

        if p.position_value_now and self.total_portfolio_value and self.total_portfolio_value > 0:
            p.position_weight = p.position_value_now / self.total_portfolio_value

        if p.position_value_now and p.total_cost:
            p.total_gain_loss_dollars = p.position_value_now - p.total_cost
            p.total_gain_loss_pct = safe_pct_change(p.position_value_now, p.total_cost)

        if self.purchase_date:
            p.days_held = (date.today() - self.purchase_date).days
            if p.total_gain_loss_pct is not None and p.days_held and p.days_held > 0:
                years_held = p.days_held / 365.25
                try:
                    p.annualized_return = (1 + p.total_gain_loss_pct) ** (1 / years_held) - 1
                except Exception:
                    pass

        return p

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_scores(self, m: AllMetrics) -> Dict[str, float]:
        """
        Score each of the 6 questions on a 0–10 scale.
        10 = outstanding, 5 = acceptable, 0 = very poor.
        """
        scores = {}

        # Q1 — Scale (0–10, based on revenue and cap size)
        q1_pts = 5.0  # neutral base — scale is descriptive, not directional
        if m.q1.ttm_revenue and m.q1.ttm_revenue > 1e8:
            q1_pts = min(10.0, 5.0 + math.log10(m.q1.ttm_revenue / 1e8))
        scores["Q1_Scale"] = round(q1_pts, 1)

        # Q2 — Growth (0–10)
        g = m.q2
        g_score = 5.0
        if g.revenue_1yr_cagr is not None:
            if g.revenue_1yr_cagr >= 0.20: g_score += 2
            elif g.revenue_1yr_cagr >= 0.10: g_score += 1
            elif g.revenue_1yr_cagr < 0: g_score -= 2
        if g.revenue_3yr_cagr is not None:
            if g.revenue_3yr_cagr >= 0.12: g_score += 1.5
            elif g.revenue_3yr_cagr < 0: g_score -= 1.5
        if g.growth_trend == "Accelerating": g_score += 1
        elif g.growth_trend == "Decelerating": g_score -= 1
        scores["Q2_Growth"] = round(max(0.0, min(10.0, g_score)), 1)

        # Q3 — Profitability (0–10)
        p = m.q3
        p_score = 5.0
        if p.operating_margin is not None:
            if p.operating_margin >= 0.25: p_score += 2
            elif p.operating_margin >= 0.10: p_score += 1
            elif p.operating_margin < 0: p_score -= 2
        if p.net_margin is not None:
            if p.net_margin >= 0.15: p_score += 1.5
            elif p.net_margin < 0: p_score -= 1.5
        if p.roic is not None:
            if p.roic >= 0.20: p_score += 1.5
            elif p.roic >= 0.10: p_score += 0.5
        if p.margin_trend == "Expanding": p_score += 0.5
        elif p.margin_trend == "Contracting": p_score -= 0.5
        scores["Q3_Profitability"] = round(max(0.0, min(10.0, p_score)), 1)

        # Q4 — Cash Flow (0–10)
        cf = m.q4
        cf_score = 5.0
        if cf.fcf_yield is not None:
            if cf.fcf_yield >= 0.05: cf_score += 2
            elif cf.fcf_yield >= 0.02: cf_score += 1
            elif cf.fcf_yield < 0: cf_score -= 2
        if cf.fcf_margin is not None:
            if cf.fcf_margin >= 0.15: cf_score += 1.5
            elif cf.fcf_margin < 0: cf_score -= 1.5
        if cf.roic is not None:
            if cf.roic >= 0.15: cf_score += 1.5
        scores["Q4_CashFlow"] = round(max(0.0, min(10.0, cf_score)), 1)

        # Q5 — Balance Sheet (0–10)
        bs = m.q5
        bs_score = 5.0
        if bs.debt_to_ebitda is not None:
            if bs.debt_to_ebitda <= 1.5: bs_score += 2
            elif bs.debt_to_ebitda <= 3.0: bs_score += 0.5
            elif bs.debt_to_ebitda > 5.0: bs_score -= 2
        if bs.current_ratio is not None:
            if bs.current_ratio >= 2.0: bs_score += 1.5
            elif bs.current_ratio < 1.0: bs_score -= 1.5
        if bs.interest_coverage is not None:
            if bs.interest_coverage >= 10: bs_score += 1.5
            elif bs.interest_coverage < 3: bs_score -= 1.5
        if bs.financial_risk_trend == "Improving": bs_score += 0.5
        elif bs.financial_risk_trend == "Deteriorating": bs_score -= 0.5
        scores["Q5_BalanceSheet"] = round(max(0.0, min(10.0, bs_score)), 1)

        # Q6 — Valuation (0–10)
        v = m.q6
        v_score = 5.0
        if v.pe_ratio is not None:
            if v.pe_ratio <= 15: v_score += 2
            elif v.pe_ratio <= 25: v_score += 1
            elif v.pe_ratio > 40: v_score -= 2
        if v.fcf_yield is not None:
            if v.fcf_yield >= 0.05: v_score += 2
            elif v.fcf_yield < 0.015: v_score -= 1
        if v.margin_of_safety is not None:
            if v.margin_of_safety >= 0.30: v_score += 1.5
            elif v.margin_of_safety < 0: v_score -= 1
        if v.peg_ratio is not None:
            if v.peg_ratio <= 1.0: v_score += 1
            elif v.peg_ratio > 2.5: v_score -= 1
        scores["Q6_Valuation"] = round(max(0.0, min(10.0, v_score)), 1)

        return scores

    # ── Helper methods ────────────────────────────────────────────────────────

    def _revenue_series(self) -> List[Optional[float]]:
        """Return list of annual revenues, most recent first."""
        if self.income.empty:
            ttm = safe_float(self.info.get("totalRevenue"))
            return [ttm] if ttm else []
        rows = [r for r in ["Total Revenue", "Revenue"] if r in self.income.index]
        if not rows:
            return []
        series = self.income.loc[rows[0]]
        values = [safe_float(v) for v in series.values]
        # Strip trailing None values (some yfinance columns have NaN for oldest year)
        while values and values[-1] is None:
            values.pop()
        return values

    def _cagr(self, start: Optional[float], end: Optional[float], years: int) -> Optional[float]:
        if start is None or end is None or start <= 0 or years <= 0:
            return None
        try:
            return (end / start) ** (1 / years) - 1
        except Exception:
            return None

    def _get_income_row(self, *row_names) -> Optional[float]:
        """Get the most recent non-None value for any of the given row names."""
        if self.income.empty:
            return None
        for name in row_names:
            if name in self.income.index:
                for val in self.income.loc[name]:
                    result = safe_float(val)
                    if result is not None:
                        return result
        return None

    def _get_income_col(self, col_idx: int, *row_names) -> Optional[float]:
        """Get value at a specific column index."""
        if self.income.empty or col_idx >= len(self.income.columns):
            return None
        for name in row_names:
            if name in self.income.index:
                return safe_float(self.income.loc[name].iloc[col_idx])
        return None

    def _get_cf_row(self, *row_names) -> Optional[float]:
        if self.cashflow.empty:
            return None
        for name in row_names:
            if name in self.cashflow.index:
                return safe_float(self.cashflow.loc[name].iloc[0])
        return None

    def _get_balance_row(self, *row_names) -> Optional[float]:
        if self.balance.empty:
            return None
        for name in row_names:
            if name in self.balance.index:
                return safe_float(self.balance.loc[name].iloc[0])
        return None

    def _compute_invested_capital(self) -> Optional[float]:
        """Invested capital = Total Equity + Total Debt - Cash.
        Use the balance sheet 'Invested Capital' row directly if available.
        """
        ic_direct = self._get_balance_row("Invested Capital")
        if ic_direct is not None:
            return ic_direct
        equity = self._get_balance_row("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
        debt = safe_float(self.info.get("totalDebt")) or self._get_balance_row("Total Debt", "Long Term Debt")
        cash = safe_float(self.info.get("totalCash")) or self._get_balance_row("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
        if equity is None:
            return None
        debt = debt or 0.0
        cash = cash or 0.0
        return equity + debt - cash

    def _estimate_fcf(self) -> Optional[float]:
        ocf = self._get_cf_row("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        capex = self._get_cf_row("Capital Expenditure", "Purchase Of PPE")
        if ocf is None:
            return None
        capex = abs(capex) if capex is not None else 0
        return ocf - capex

    def _simple_dcf(self) -> Optional[float]:
        """
        Simple Gordon Growth DCF: Price = FCF_per_share * (1+g) / (wacc - g)
        Uses conservative assumptions; clearly labeled as estimate.
        """
        fcf = safe_float(self.info.get("freeCashflow")) or self._estimate_fcf()
        shares = safe_float(self.info.get("sharesOutstanding"))
        if not fcf or not shares or shares == 0:
            return None
        fcf_ps = fcf / shares
        if fcf_ps <= 0:
            return None
        # Use 5yr revenue CAGR as proxy for perpetual growth, capped at 3%
        rev_series = self._revenue_series()
        if len(rev_series) >= 6:
            cagr = self._cagr(rev_series[5], rev_series[0], 5) or 0.03
        elif len(rev_series) >= 2:
            cagr = self._cagr(rev_series[1], rev_series[0], 1) or 0.03
        else:
            cagr = 0.03
        terminal_growth = max(0.01, min(0.03, cagr * 0.5))  # cap terminal growth at 3%
        wacc = 0.10  # conservative 10% discount rate
        if wacc <= terminal_growth:
            return None
        intrinsic_value = fcf_ps * (1 + terminal_growth) / (wacc - terminal_growth)
        return intrinsic_value

    # ── Purchase-date helpers (Existing Holding mode) ─────────────────────────

    def _revenue_at_date(self, target_date: date) -> Optional[float]:
        """Estimate revenue at purchase date from annual filings."""
        revenues = self._revenue_series()
        if not revenues or len(revenues) < 2:
            return revenues[0] if revenues else None
        # Simple interpolation: use the most recently available prior-year figure
        return revenues[1] if len(revenues) > 1 else revenues[0]

    def _revenue_cagr_at_date(self, target_date: date) -> Optional[float]:
        """Revenue CAGR as it would have appeared near purchase date."""
        revenues = self._revenue_series()
        if len(revenues) < 3:
            return None
        return self._cagr(revenues[2], revenues[1], 1)

    def _margin_at_date(self, target_date: date, margin_type: str) -> Optional[float]:
        if self.income.empty or len(self.income.columns) < 2:
            return None
        rev = self._get_income_col(1, "Total Revenue")
        if margin_type == "gross":
            num = self._get_income_col(1, "Gross Profit")
        else:
            num = self._get_income_col(1, "Operating Income")
        return safe_divide(num, rev)

    def _fcf_at_date(self, target_date: date) -> Optional[float]:
        if self.cashflow.empty or len(self.cashflow.columns) < 2:
            return None
        ocf_rows = [r for r in ["Free Cash Flow", "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"] if r in self.cashflow.index]
        if ocf_rows and ocf_rows[0] == "Free Cash Flow":
            return safe_float(self.cashflow.loc["Free Cash Flow"].iloc[min(1, len(self.cashflow.columns)-1)])
        cap_rows = [r for r in ["Capital Expenditure", "Purchase Of PPE"] if r in self.cashflow.index]
        ocf = safe_float(self.cashflow.loc[ocf_rows[0]].iloc[min(1, len(self.cashflow.columns)-1)]) if ocf_rows else None
        capex = safe_float(self.cashflow.loc[cap_rows[0]].iloc[min(1, len(self.cashflow.columns)-1)]) if cap_rows else None
        if ocf is None:
            return None
        capex = abs(capex) if capex is not None else 0
        return ocf - capex

    def _debt_ebitda_at_date(self, target_date: date) -> Optional[float]:
        if self.balance.empty or self.income.empty:
            return None
        debt_rows = [r for r in ["Total Debt", "Long Term Debt And Capital Lease Obligation"] if r in self.balance.index]
        debt = safe_float(self.balance.loc[debt_rows[0]].iloc[min(1, len(self.balance.columns)-1)]) if debt_rows else None
        ebitda_proxy = None
        # Try direct EBITDA row first
        ebitda_rows = [r for r in ["EBITDA", "Normalized EBITDA"] if r in self.income.index]
        if ebitda_rows:
            ebitda_proxy = safe_float(self.income.loc[ebitda_rows[0]].iloc[min(1, len(self.income.columns)-1)])
        if ebitda_proxy is None:
            op_rows = [r for r in ["EBIT", "Operating Income"] if r in self.income.index]
            cf_rows = [r for r in ["Depreciation And Amortization", "Depreciation Amortization Depletion"] if r in self.cashflow.index]
            if op_rows and not self.cashflow.empty and cf_rows:
                ebit = safe_float(self.income.loc[op_rows[0]].iloc[min(1, len(self.income.columns)-1)])
                da = safe_float(self.cashflow.loc[cf_rows[0]].iloc[min(1, len(self.cashflow.columns)-1)])
                if ebit is not None and da is not None:
                    ebitda_proxy = ebit + abs(da)
        return safe_divide(debt, ebitda_proxy)

    def _pe_at_purchase_date(self) -> Optional[float]:
        """Estimate trailing P/E at purchase date using historical price and EPS proxy."""
        if not self.purchase_date:
            return None
        price = self.stock.get_price_on_date(self.purchase_date)
        if self.cost_basis:
            price = self.cost_basis
        eps_prior = self._get_income_col(1, "Net Income")
        shares = safe_float(self.info.get("sharesOutstanding"))
        if price and eps_prior and shares and shares > 0:
            eps = eps_prior / shares
            if eps > 0:
                return price / eps
        return None

    def _ev_ebitda_at_purchase_date(self) -> Optional[float]:
        """Very rough EV/EBITDA at purchase: use current EBITDA, historical price."""
        ebitda = safe_float(self.info.get("ebitda"))
        if not ebitda or not self.purchase_date:
            return None
        price = self.stock.get_price_on_date(self.purchase_date)
        if self.cost_basis:
            price = self.cost_basis
        shares = safe_float(self.info.get("sharesOutstanding"))
        debt = safe_float(self.info.get("totalDebt")) or 0
        cash = safe_float(self.info.get("totalCash")) or 0
        if price and shares:
            ev = price * shares + debt - cash
            return safe_divide(ev, ebitda)
        return None

    # ── Dividend helpers ──────────────────────────────────────────────────────

    def _annual_dividends(self) -> "pd.Series":
        """Return annual dividend sums. Handles tz-aware DatetimeIndex."""
        import pandas as pd
        divs = self.divs.copy()
        # Normalise tz-aware index to tz-naive for resample compatibility
        if hasattr(divs.index, "tz") and divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        return divs.resample("YE").sum()

    def _count_consecutive_dividend_years(self) -> int:
        if self.divs.empty:
            return 0
        annual = self._annual_dividends()
        annual = annual[annual > 0]
        if annual.empty:
            return 0
        years = sorted(annual.index.year, reverse=True)
        count = 0
        expected = years[0]
        for y in years:
            if y == expected:
                count += 1
                expected -= 1
            else:
                break
        return count

    def _dividend_cagr(self, years: int = 5) -> Optional[float]:
        if self.divs.empty:
            return None
        annual = self._annual_dividends()
        annual = annual[annual > 0]
        if len(annual) < 2:
            return None
        recent_years = annual.tail(years + 1)
        if len(recent_years) < 2:
            return None
        start = safe_float(recent_years.iloc[0])
        end = safe_float(recent_years.iloc[-1])
        n = len(recent_years) - 1
        return self._cagr(start, end, n)

    def _last_dividend_change(self) -> str:
        if self.divs.empty:
            return "N/A"
        annual = self._annual_dividends()
        annual = annual[annual > 0]
        if len(annual) < 2:
            return "N/A"
        last = float(annual.iloc[-1])
        prior = float(annual.iloc[-2])
        if prior == 0:
            return "Initiated"
        change = (last - prior) / prior
        if change > 0.01:
            return f"Increase (+{change*100:.1f}%)"
        elif change < -0.01:
            return f"Cut ({change*100:.1f}%)"
        return "Stable"

    def _dividends_received_since(self, since_date: date, shares: float) -> Optional[float]:
        if self.divs.empty:
            return None
        divs = self.divs.copy()
        divs.index = pd.to_datetime(divs.index).tz_localize(None)
        mask = divs.index >= pd.Timestamp(since_date)
        return float((divs[mask] * shares).sum())
