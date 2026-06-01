"""
metric_interpreter.py — Plain-English metric interpretation system.

Classifies every metric as Excellent / Good / Neutral / Weak / Bad.
Provides color-coded badges, plain-English meaning, and one-sentence commentary.
Works for any ticker; sector context adjusts thresholds when available.
"""

from typing import Dict, Optional, Tuple

# ── Rating color palette ──────────────────────────────────────────────────────
RATING_TEXT_COLORS: Dict[str, str] = {
    "Excellent": "#145a32",
    "Good":      "#1e8449",
    "Neutral":   "#7d6608",
    "Weak":      "#a04000",
    "Bad":       "#922b21",
    "N/A":       "#566573",
}
RATING_BG_COLORS: Dict[str, str] = {
    "Excellent": "#d5f5e3",
    "Good":      "#eafaf1",
    "Neutral":   "#fef9e7",
    "Weak":      "#fdebd0",
    "Bad":       "#fdecea",
    "N/A":       "#f2f3f4",
}
RATING_BORDER_COLORS: Dict[str, str] = {
    "Excellent": "#27ae60",
    "Good":      "#52be80",
    "Neutral":   "#f39c12",
    "Weak":      "#e67e22",
    "Bad":       "#e74c3c",
    "N/A":       "#aab7b8",
}

# Sector-specific PE/PS multipliers (mirrors threshold_manager.py logic)
_SECTOR_ADJUSTMENTS: Dict[str, Dict] = {
    "Technology":        {"pe_mult": 1.6, "ps_mult": 2.5, "gross_margin_boost": 0.10},
    "Healthcare":        {"pe_mult": 1.4, "ps_mult": 1.5},
    "Financials":        {"pe_mult": 0.7, "debt_ebitda_mult": 2.5},
    "Real Estate":       {"pe_mult": 1.8, "debt_ebitda_mult": 3.0},
    "Utilities":         {"pe_mult": 1.2, "debt_ebitda_mult": 2.0},
    "Energy":            {"pe_mult": 0.8, "capex_relax": True},
    "Communication Services": {"pe_mult": 1.2, "ps_mult": 1.5},
    "Consumer Discretionary": {"pe_mult": 1.1},
    "Consumer Staples":  {"pe_mult": 0.9},
    "Industrials":       {"pe_mult": 1.0, "capex_relax": True},
    "Materials":         {"pe_mult": 0.9},
}


def _get_sector_adj(sector: Optional[str]) -> Dict:
    if not sector:
        return {}
    for key, adj in _SECTOR_ADJUSTMENTS.items():
        if key.lower() in sector.lower() or sector.lower() in key.lower():
            return adj
    return {}


# ── Metric rules dictionary ───────────────────────────────────────────────────
# Each entry:
#   display_name    — human-readable name
#   direction       — "higher_is_better" | "lower_is_better" | "context_dependent"
#   excellent/good/neutral/weak/bad — [low, high] inclusive range for the rating
#   plain_english   — one line: what the metric measures
#   warning         — one line: important caveat or context
#   benchmark_label — what the thresholds represent
#   format          — "pct" | "multiple" | "currency" | "number"

METRIC_RULES: Dict[str, Dict] = {

    # ── Q1: Scale ─────────────────────────────────────────────────────────────
    "market_cap": {
        "display_name": "Market Cap",
        "direction": "higher_is_better",
        "excellent": [2e11,  float("inf")],
        "good":      [1e10,  2e11],
        "neutral":   [2e9,   1e10],
        "weak":      [3e8,   2e9],
        "bad":       [-float("inf"), 3e8],
        "plain_english": "Total market value of all outstanding shares — a measure of company size and investor confidence.",
        "warning": "Larger companies offer more stability and liquidity; smaller ones carry more execution risk.",
        "benchmark_label": "Mega >$200B | Large $10–200B | Mid $2–10B | Small <$2B",
        "format": "currency",
    },
    "enterprise_value": {
        "display_name": "Enterprise Value (EV)",
        "direction": "context_dependent",
        "plain_english": "Total theoretical acquisition cost: market cap + debt − cash. More complete than market cap alone.",
        "warning": "EV is best used as the numerator in EV/EBITDA or EV/Sales — not as a standalone signal.",
        "benchmark_label": "N/A — use EV/EBITDA or EV/Sales for context",
        "format": "currency",
    },
    "ttm_revenue": {
        "display_name": "TTM Revenue",
        "direction": "higher_is_better",
        "excellent": [1e10,  float("inf")],
        "good":      [1e9,   1e10],
        "neutral":   [1e8,   1e9],
        "weak":      [1e7,   1e8],
        "bad":       [-float("inf"), 1e7],
        "plain_english": "Total revenue earned over the trailing twelve months — the top-line scale of the business.",
        "warning": "Revenue must be read alongside growth rate and margin to determine true business quality.",
        "benchmark_label": "Absolute size: >$10B large | $1–10B mid | <$100M small",
        "format": "currency",
    },

    # ── Q2: Growth ────────────────────────────────────────────────────────────
    "revenue_1yr_cagr": {
        "display_name": "Revenue Growth (1yr)",
        "direction": "higher_is_better",
        "excellent": [0.20,  float("inf")],
        "good":      [0.10,  0.20],
        "neutral":   [0.05,  0.10],
        "weak":      [0.00,  0.05],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Year-over-year revenue growth rate — how much faster the business is getting.",
        "warning": "Single-year figures can be distorted by acquisitions, divestitures, or macro events. Check the multi-year trend.",
        "benchmark_label": ">20% strong | 10–20% good | 5–10% moderate | <0% declining",
        "format": "pct",
    },
    "revenue_3yr_cagr": {
        "display_name": "Revenue Growth (3yr CAGR)",
        "direction": "higher_is_better",
        "excellent": [0.15,  float("inf")],
        "good":      [0.08,  0.15],
        "neutral":   [0.03,  0.08],
        "weak":      [0.00,  0.03],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Annualized 3-year revenue growth — smooths out single-year noise for a cleaner trend view.",
        "warning": "Consistent multi-year growth is a more reliable quality signal than any single year.",
        "benchmark_label": ">15% excellent | 8–15% good | <0% declining",
        "format": "pct",
    },
    "revenue_5yr_cagr": {
        "display_name": "Revenue Growth (5yr CAGR)",
        "direction": "higher_is_better",
        "excellent": [0.12,  float("inf")],
        "good":      [0.06,  0.12],
        "neutral":   [0.02,  0.06],
        "weak":      [0.00,  0.02],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Annualized 5-year revenue growth — the strongest structural trend signal available.",
        "warning": "A 5-year growth track record is one of the most reliable indicators of a durable business.",
        "benchmark_label": ">12% excellent | 6–12% good | <0% structural decline",
        "format": "pct",
    },
    "earnings_1yr_growth": {
        "display_name": "Earnings Growth (1yr)",
        "direction": "higher_is_better",
        "excellent": [0.20,  float("inf")],
        "good":      [0.10,  0.20],
        "neutral":   [0.00,  0.10],
        "weak":      [-0.10, 0.00],
        "bad":       [-float("inf"), -0.10],
        "plain_english": "Year-over-year change in net earnings — shows whether profits are expanding or contracting.",
        "warning": "Earnings can be manipulated through accounting choices. Always cross-check with free cash flow growth.",
        "benchmark_label": ">20% strong | 0–10% modest | <-10% contracting",
        "format": "pct",
    },
    "forward_revenue_growth": {
        "display_name": "Forward Revenue Growth (est.)",
        "direction": "higher_is_better",
        "excellent": [0.20,  float("inf")],
        "good":      [0.10,  0.20],
        "neutral":   [0.05,  0.10],
        "weak":      [0.00,  0.05],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Analyst consensus estimate for next year's revenue growth — a forward-looking signal.",
        "warning": "Analyst estimates are typically optimistic and often revised down. Treat as directional, not precise.",
        "benchmark_label": "Analyst consensus (forward-looking — treat as approximate)",
        "format": "pct",
    },

    # ── Q3: Profitability ─────────────────────────────────────────────────────
    "gross_margin": {
        "display_name": "Gross Margin",
        "direction": "higher_is_better",
        "excellent": [0.60,  float("inf")],
        "good":      [0.40,  0.60],
        "neutral":   [0.25,  0.40],
        "weak":      [0.10,  0.25],
        "bad":       [-float("inf"), 0.10],
        "plain_english": "Revenue remaining after direct cost of goods sold — a proxy for pricing power and product differentiation.",
        "warning": "High gross margin often signals a competitive moat. Always compare to sector peers (retail naturally runs lower).",
        "benchmark_label": "Rules of thumb — sector-adjust for retail/manufacturing",
        "format": "pct",
    },
    "operating_margin": {
        "display_name": "Operating Margin",
        "direction": "higher_is_better",
        "excellent": [0.25,  float("inf")],
        "good":      [0.15,  0.25],
        "neutral":   [0.05,  0.15],
        "weak":      [0.00,  0.05],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Revenue remaining after all operating costs — shows how efficiently the core business runs.",
        "warning": "Operating margin above 20% typically signals durable competitive advantage or pricing power.",
        "benchmark_label": ">25% excellent | 15–25% good | <0% unprofitable",
        "format": "pct",
    },
    "net_margin": {
        "display_name": "Net Profit Margin",
        "direction": "higher_is_better",
        "excellent": [0.15,  float("inf")],
        "good":      [0.08,  0.15],
        "neutral":   [0.03,  0.08],
        "weak":      [0.00,  0.03],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Percentage of revenue that becomes bottom-line profit after all expenses, interest, and taxes.",
        "warning": "Net margin can be distorted by interest, taxes, and one-off items. Cross-check with operating margin.",
        "benchmark_label": ">15% strong | 3–8% modest | <0% loss-making",
        "format": "pct",
    },
    "ebitda_margin": {
        "display_name": "EBITDA Margin",
        "direction": "higher_is_better",
        "excellent": [0.30,  float("inf")],
        "good":      [0.20,  0.30],
        "neutral":   [0.10,  0.20],
        "weak":      [0.05,  0.10],
        "bad":       [-float("inf"), 0.05],
        "plain_english": "Operating profitability before interest, taxes, depreciation, and amortization — useful for comparing companies.",
        "warning": "EBITDA excludes depreciation, which can be real economic cost for capital-intensive businesses.",
        "benchmark_label": ">30% excellent | 20–30% good | <5% thin",
        "format": "pct",
    },
    "roe": {
        "display_name": "Return on Equity (ROE)",
        "direction": "higher_is_better",
        "excellent": [0.25,  float("inf")],
        "good":      [0.15,  0.25],
        "neutral":   [0.08,  0.15],
        "weak":      [0.00,  0.08],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Net profit generated for each dollar of shareholder equity — measures how well management uses investor capital.",
        "warning": "Very high ROE combined with high debt may reflect leverage, not genuine operational excellence. Check ROIC.",
        "benchmark_label": ">25% exceptional | 15–25% strong | <0% destroying equity value",
        "format": "pct",
    },
    "roa": {
        "display_name": "Return on Assets (ROA)",
        "direction": "higher_is_better",
        "excellent": [0.10,  float("inf")],
        "good":      [0.05,  0.10],
        "neutral":   [0.02,  0.05],
        "weak":      [0.00,  0.02],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Net profit relative to total assets — measures overall asset efficiency regardless of financing.",
        "warning": "Asset-heavy industries (utilities, manufacturing) naturally run lower ROA than software or services.",
        "benchmark_label": ">10% excellent | 5–10% good | <0% losing money on assets",
        "format": "pct",
    },
    "roic": {
        "display_name": "Return on Invested Capital (ROIC)",
        "direction": "higher_is_better",
        "excellent": [0.20,  float("inf")],
        "good":      [0.12,  0.20],
        "neutral":   [0.07,  0.12],
        "weak":      [0.00,  0.07],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Return earned on every dollar of capital deployed in the business — the gold standard of capital efficiency.",
        "warning": "ROIC above the cost of capital (typically 8–10%) means the company is genuinely creating value for owners.",
        "benchmark_label": "WACC ~8–10% | >15% outstanding value creator | <WACC destroys value",
        "format": "pct",
    },

    # ── Q4: Cash Flow ─────────────────────────────────────────────────────────
    "fcf_margin": {
        "display_name": "FCF Margin",
        "direction": "higher_is_better",
        "excellent": [0.20,  float("inf")],
        "good":      [0.10,  0.20],
        "neutral":   [0.05,  0.10],
        "weak":      [0.00,  0.05],
        "bad":       [-float("inf"), 0.00],
        "plain_english": "Free cash flow as a percentage of revenue — the purest measure of how much cash the business actually generates.",
        "warning": "FCF margin is harder to manipulate than reported earnings and is a key indicator of business quality.",
        "benchmark_label": ">20% excellent | 10–20% good | <0% cash-burning",
        "format": "pct",
    },
    "fcf_yield": {
        "display_name": "FCF Yield",
        "direction": "higher_is_better",
        "excellent": [0.08,  float("inf")],
        "good":      [0.05,  0.08],
        "neutral":   [0.03,  0.05],
        "weak":      [0.015, 0.03],
        "bad":       [-float("inf"), 0.015],
        "plain_english": "Free cash flow as a percentage of market cap — like a cash dividend yield even if no dividend is paid.",
        "warning": "FCF yield above 5% often means the stock offers fair-to-attractive cash returns at current price.",
        "benchmark_label": ">8% excellent | >5% good | <2% expensive on cash basis",
        "format": "pct",
    },
    "capex_intensity": {
        "display_name": "CapEx Intensity (CapEx / Revenue)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 0.03],
        "good":      [0.03, 0.07],
        "neutral":   [0.07, 0.12],
        "weak":      [0.12, 0.20],
        "bad":       [0.20, float("inf")],
        "plain_english": "Capital expenditure as a share of revenue — lower means a more cash-light, scalable business.",
        "warning": "Software and services companies typically run <3%; manufacturers and utilities can be 15–25%.",
        "benchmark_label": "<3% cash-light | 7–12% moderate | >20% capital-intensive",
        "format": "pct",
    },

    # ── Q5: Balance Sheet ─────────────────────────────────────────────────────
    "debt_to_ebitda": {
        "display_name": "Debt / EBITDA",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 1.5],
        "good":      [1.5, 2.5],
        "neutral":   [2.5, 3.5],
        "weak":      [3.5, 5.0],
        "bad":       [5.0, float("inf")],
        "plain_english": "Years of operating cash flow needed to pay off all debt — the primary leverage health check.",
        "warning": "Above 4–5x is high; above 6x signals elevated financial risk, especially in downturns or rate cycles.",
        "benchmark_label": "<2x healthy | >4x elevated | >6x distressed (varies by sector)",
        "format": "multiple",
    },
    "debt_to_equity": {
        "display_name": "Debt / Equity",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 0.30],
        "good":      [0.30, 0.70],
        "neutral":   [0.70, 1.50],
        "weak":      [1.50, 3.00],
        "bad":       [3.00, float("inf")],
        "plain_english": "Total debt relative to shareholder equity — measures how much the business is financed by debt vs. owners.",
        "warning": "High D/E is acceptable for stable-cash-flow businesses (utilities, REITs) but dangerous for cyclicals.",
        "benchmark_label": "<0.5 conservative | >2.0 highly leveraged",
        "format": "multiple",
    },
    "current_ratio": {
        "display_name": "Current Ratio",
        "direction": "higher_is_better",
        "excellent": [2.00, float("inf")],
        "good":      [1.50, 2.00],
        "neutral":   [1.20, 1.50],
        "weak":      [1.00, 1.20],
        "bad":       [-float("inf"), 1.00],
        "plain_english": "Current assets divided by current liabilities — measures ability to cover short-term obligations.",
        "warning": "Below 1.0x means the company owes more in the next 12 months than it has in liquid assets. Red flag.",
        "benchmark_label": ">2.0 safe | 1.0–1.5 watch | <1.0 liquidity warning",
        "format": "multiple",
    },
    "quick_ratio": {
        "display_name": "Quick Ratio",
        "direction": "higher_is_better",
        "excellent": [2.00, float("inf")],
        "good":      [1.00, 2.00],
        "neutral":   [0.80, 1.00],
        "weak":      [0.50, 0.80],
        "bad":       [-float("inf"), 0.50],
        "plain_english": "Like current ratio but excludes inventory — a stricter test of immediate liquidity.",
        "warning": "Important for companies that carry inventory that may be slow to convert to cash.",
        "benchmark_label": ">1.0 healthy | <0.5 tight liquidity",
        "format": "multiple",
    },
    "interest_coverage": {
        "display_name": "Interest Coverage",
        "direction": "higher_is_better",
        "excellent": [15.0, float("inf")],
        "good":      [8.0,  15.0],
        "neutral":   [4.0,  8.0],
        "weak":      [2.0,  4.0],
        "bad":       [-float("inf"), 2.0],
        "plain_english": "Times operating earnings cover interest payments — measures how safely the company can service its debt.",
        "warning": "Below 2x means earnings could drop modestly and debt payments become difficult. Financial stress signal.",
        "benchmark_label": ">5x comfortable | 2–4x cautious | <2x financial stress",
        "format": "multiple",
    },

    # ── Q6: Valuation ─────────────────────────────────────────────────────────
    "pe_ratio": {
        "display_name": "P/E Ratio (Trailing)",
        "direction": "lower_is_better",
        "excellent": [0,  15],
        "good":      [15, 20],
        "neutral":   [20, 30],
        "weak":      [30, 50],
        "bad":       [50, float("inf")],
        "plain_english": "Price paid for each dollar of the company's trailing 12-month earnings.",
        "warning": "A high P/E is justified only if growth is strong and sustained. Always cross-check with PEG and FCF yield.",
        "benchmark_label": "Market avg ~20x | sector-adjusted for tech/growth",
        "format": "multiple",
    },
    "forward_pe": {
        "display_name": "Forward P/E",
        "direction": "lower_is_better",
        "excellent": [0,  15],
        "good":      [15, 20],
        "neutral":   [20, 28],
        "weak":      [28, 40],
        "bad":       [40, float("inf")],
        "plain_english": "Price paid for next year's estimated earnings — forward-looking and often more relevant than trailing P/E.",
        "warning": "Analyst estimates are frequently revised. Forward P/E is best used as a directional signal, not a precise measure.",
        "benchmark_label": "Analyst consensus estimates (approximate)",
        "format": "multiple",
    },
    "peg_ratio": {
        "display_name": "PEG Ratio",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 1.0],
        "good":      [1.0, 1.5],
        "neutral":   [1.5, 2.0],
        "weak":      [2.0, 3.0],
        "bad":       [3.0, float("inf")],
        "plain_english": "P/E ratio divided by earnings growth rate — adjusts for growth so high-growth and value stocks can be compared.",
        "warning": "PEG below 1.0 signals the market may be underpricing the growth opportunity (Peter Lynch rule).",
        "benchmark_label": "PEG <1 = undervalued | PEG >2 = expensive relative to growth",
        "format": "multiple",
    },
    "ev_ebitda": {
        "display_name": "EV / EBITDA",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 8],
        "good":      [8,  12],
        "neutral":   [12, 18],
        "weak":      [18, 25],
        "bad":       [25, float("inf")],
        "plain_english": "Enterprise value relative to operating earnings — a capital-structure-neutral valuation multiple.",
        "warning": "EV/EBITDA is preferred over P/E when comparing companies with different debt loads or tax situations.",
        "benchmark_label": "Market avg ~12–15x | sector-adjusted | <8x cheap | >25x expensive",
        "format": "multiple",
    },
    "price_to_sales": {
        "display_name": "Price / Sales (P/S)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 2],
        "good":      [2,  4],
        "neutral":   [4,  6],
        "weak":      [6,  10],
        "bad":       [10, float("inf")],
        "plain_english": "Market cap relative to annual revenue — useful when earnings are low, negative, or not meaningful.",
        "warning": "A high P/S is only justified if margins are expanding and growth is durable. Software often runs >10x.",
        "benchmark_label": "Sector-dependent: retail <1x normal | software >10x common",
        "format": "multiple",
    },
    "price_to_book": {
        "display_name": "Price / Book (P/B)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 2],
        "good":      [2,  4],
        "neutral":   [4,  6],
        "weak":      [6,  10],
        "bad":       [10, float("inf")],
        "plain_english": "Market cap relative to the accounting book value of equity — shows premium paid over net assets.",
        "warning": "P/B matters most for asset-heavy industries (banks, industrials). Asset-light companies naturally have high P/B.",
        "benchmark_label": "Sector-dependent: banks ~1x normal | software >10x common",
        "format": "multiple",
    },
    "price_to_fcf": {
        "display_name": "Price / Free Cash Flow",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 15],
        "good":      [15, 25],
        "neutral":   [25, 35],
        "weak":      [35, 50],
        "bad":       [50, float("inf")],
        "plain_english": "Market cap relative to annual free cash flow — a purer valuation measure than P/E.",
        "warning": "P/FCF below 20x generally means cash generation is attractively valued at the current price.",
        "benchmark_label": "Market avg ~25–30x | <15x cheap | >50x very expensive",
        "format": "multiple",
    },
    "margin_of_safety": {
        "display_name": "Margin of Safety (DCF est.)",
        "direction": "higher_is_better",
        "excellent": [0.30,  float("inf")],
        "good":      [0.15,  0.30],
        "neutral":   [0.00,  0.15],
        "weak":      [-0.15, 0.00],
        "bad":       [-float("inf"), -0.15],
        "plain_english": "How far below the estimated DCF intrinsic value the stock trades — a cushion for uncertainty and error.",
        "warning": "This is a simplified DCF estimate (10% WACC). Use as directional context, not a precise fair-value target.",
        "benchmark_label": "Conservative DCF (10% WACC, capped terminal growth)",
        "format": "pct",
    },

    # ── First-Principles (Past/Future) ──────────────────────────────────────
    "real_owner_earnings_cagr_5y": {
        "display_name": "Real Owner Earnings CAGR (5Y)",
        "direction": "higher_is_better",
        "excellent": [0.10, float("inf")],
        "good": [0.06, 0.10],
        "neutral": [0.02, 0.06],
        "weak": [0.00, 0.02],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Inflation-adjusted annual growth in owner earnings per share over five years.",
        "warning": "Sustained positive real per-share compounding is a core business-quality signal.",
        "benchmark_label": ">=6% strong | 2-6% watch | <=0% fail risk",
        "format": "pct",
    },
    "real_fcf_per_share_cagr_5y": {
        "display_name": "Real FCF/Share CAGR (5Y)",
        "direction": "higher_is_better",
        "excellent": [0.10, float("inf")],
        "good": [0.06, 0.10],
        "neutral": [0.02, 0.06],
        "weak": [0.00, 0.02],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Inflation-adjusted free-cash-flow-per-share annual growth over five years.",
        "warning": "Cash-flow compounding should corroborate accounting earnings compounding.",
        "benchmark_label": ">=6% strong | 2-6% watch | <=0% weak",
        "format": "pct",
    },
    "real_revenue_per_share_cagr_5y": {
        "display_name": "Real Revenue/Share CAGR (5Y)",
        "direction": "higher_is_better",
        "excellent": [0.08, float("inf")],
        "good": [0.05, 0.08],
        "neutral": [0.02, 0.05],
        "weak": [0.00, 0.02],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Inflation-adjusted top-line growth on a per-share basis.",
        "warning": "Per-share framing removes dilution noise from raw revenue growth.",
        "benchmark_label": ">=5% good | 2-5% moderate | <=0% weak",
        "format": "pct",
    },
    "cumulative_real_owner_earnings_change": {
        "display_name": "Cumulative Real Owner Earnings Change",
        "direction": "higher_is_better",
        "excellent": [0.25, float("inf")],
        "good": [0.10, 0.25],
        "neutral": [0.00, 0.10],
        "weak": [-0.20, 0.00],
        "bad": [-float("inf"), -0.20],
        "plain_english": "Total real change in owner earnings per share across the observed cycle.",
        "warning": "Large cumulative declines often indicate structural deterioration.",
        "benchmark_label": "Decline worse than -20% is a falsification flag",
        "format": "pct",
    },
    "median_incremental_roic_5y": {
        "display_name": "Median Incremental ROIC (5Y)",
        "direction": "higher_is_better",
        "excellent": [0.15, float("inf")],
        "good": [0.12, 0.15],
        "neutral": [0.08, 0.12],
        "weak": [0.00, 0.08],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Median return earned on newly invested capital over five fiscal years.",
        "warning": "Incremental returns below hurdle suggest low-quality reinvestment.",
        "benchmark_label": ">=12% strong | 8-12% watch | <8% weak",
        "format": "pct",
    },
    "incremental_roic_years_below_watch": {
        "display_name": "Years Incremental ROIC Below Hurdle",
        "direction": "lower_is_better",
        "excellent": [0.0, 0.0],
        "good": [0.0, 1.0],
        "neutral": [1.0, 2.0],
        "weak": [2.0, 3.0],
        "bad": [3.0, float("inf")],
        "plain_english": "Count of years where incremental returns were below the watch threshold.",
        "warning": "Repeated sub-hurdle years can indicate deteriorating project quality.",
        "benchmark_label": "2+ years is an elevated risk signal",
        "format": "number",
    },
    "accrual_ratio_latest": {
        "display_name": "Accrual Ratio (Latest)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 0.03],
        "good": [0.03, 0.05],
        "neutral": [0.05, 0.08],
        "weak": [0.08, 0.12],
        "bad": [0.12, float("inf")],
        "plain_english": "Difference between earnings and cash flow relative to assets.",
        "warning": "Higher accruals imply lower earnings quality and weaker persistence.",
        "benchmark_label": "<=3% strong | >8% weak",
        "format": "pct",
    },
    "cfo_to_ni_latest": {
        "display_name": "CFO / Net Income (Latest)",
        "direction": "higher_is_better",
        "excellent": [1.2, float("inf")],
        "good": [1.0, 1.2],
        "neutral": [0.8, 1.0],
        "weak": [0.6, 0.8],
        "bad": [-float("inf"), 0.6],
        "plain_english": "Cash conversion of accounting earnings.",
        "warning": "Persistent CFO/NI below 1.0 can indicate weak earnings backing.",
        "benchmark_label": ">=1.0 preferred | <0.8 caution",
        "format": "multiple",
    },
    "revenue_cagr_5y": {
        "display_name": "Revenue CAGR (5Y)",
        "direction": "higher_is_better",
        "excellent": [0.12, float("inf")],
        "good": [0.08, 0.12],
        "neutral": [0.03, 0.08],
        "weak": [0.00, 0.03],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Annualized five-year revenue growth rate.",
        "warning": "Use with share growth to test whether owners benefit on a per-share basis.",
        "benchmark_label": ">=8% strong | 3-8% moderate | <=0% weak",
        "format": "pct",
    },
    "share_cagr_5y": {
        "display_name": "Share Count CAGR (5Y)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 0.005],
        "good": [0.005, 0.01],
        "neutral": [0.01, 0.02],
        "weak": [0.02, 0.04],
        "bad": [0.04, float("inf")],
        "plain_english": "Annualized growth in diluted shares outstanding (dilution rate).",
        "warning": "High share issuance can offset operating growth for existing owners.",
        "benchmark_label": "<=0.5% strong | >2% dilution risk",
        "format": "pct",
    },
    "dilution_adjusted_growth": {
        "display_name": "Dilution-Adjusted Growth",
        "direction": "higher_is_better",
        "excellent": [0.08, float("inf")],
        "good": [0.04, 0.08],
        "neutral": [0.01, 0.04],
        "weak": [0.00, 0.01],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Revenue growth net of share-count growth.",
        "warning": "Positive spread implies owners capture growth rather than get diluted out.",
        "benchmark_label": "Higher is better; <=0 suggests dilution drag",
        "format": "pct",
    },
    "net_debt_trend": {
        "display_name": "Net Debt Trend",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), -0.10],
        "good": [-0.10, 0.00],
        "neutral": [0.00, 0.15],
        "weak": [0.15, 0.25],
        "bad": [0.25, float("inf")],
        "plain_english": "Relative change in net debt over the analysis window.",
        "warning": "Rising leverage with weak cash generation is a key fragility pattern.",
        "benchmark_label": ">25% increase is deterioration risk",
        "format": "pct",
    },
    "fcf_payout_coverage_latest": {
        "display_name": "FCF Payout Coverage (Latest)",
        "direction": "higher_is_better",
        "excellent": [2.0, float("inf")],
        "good": [1.5, 2.0],
        "neutral": [1.0, 1.5],
        "weak": [0.8, 1.0],
        "bad": [-float("inf"), 0.8],
        "plain_english": "How many times free cash flow covers shareholder payouts.",
        "warning": "Coverage below 1.0 implies payouts rely on external funding or balance sheet draw.",
        "benchmark_label": ">=1.5 strong | 1.0-1.5 watch | <1.0 weak",
        "format": "multiple",
    },
    "payout_to_fcf_5y": {
        "display_name": "Payout / FCF (5Y)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 0.60],
        "good": [0.60, 1.00],
        "neutral": [1.00, 1.20],
        "weak": [1.20, 1.50],
        "bad": [1.50, float("inf")],
        "plain_english": "Five-year cumulative payout intensity versus free cash generation.",
        "warning": "Sustained payout ratios above 1.0 reduce reinvestment flexibility.",
        "benchmark_label": "<=1.0 preferred | >1.2 caution",
        "format": "multiple",
    },
    "min_fcf_margin": {
        "display_name": "Minimum FCF Margin",
        "direction": "higher_is_better",
        "excellent": [0.12, float("inf")],
        "good": [0.07, 0.12],
        "neutral": [0.03, 0.07],
        "weak": [0.00, 0.03],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Worst observed free-cash-flow margin through the sampled period.",
        "warning": "Negative trough margins indicate fragility during stress periods.",
        "benchmark_label": "Higher trough margins imply stronger resilience",
        "format": "pct",
    },
    "interest_coverage_trough": {
        "display_name": "Interest Coverage Trough",
        "direction": "higher_is_better",
        "excellent": [4.0, float("inf")],
        "good": [3.0, 4.0],
        "neutral": [2.0, 3.0],
        "weak": [1.0, 2.0],
        "bad": [-float("inf"), 1.0],
        "plain_english": "Lowest observed ability to cover interest expense from operating earnings.",
        "warning": "Values below 2x indicate elevated refinance and solvency pressure.",
        "benchmark_label": ">3x strong | 2-3x watch | <2x weak",
        "format": "multiple",
    },
    "recovery_speed_years": {
        "display_name": "Recovery Speed (Years)",
        "direction": "lower_is_better",
        "excellent": [0.0, 1.0],
        "good": [1.0, 2.0],
        "neutral": [2.0, 3.0],
        "weak": [3.0, 4.0],
        "bad": [4.0, float("inf")],
        "plain_english": "Approximate years spent below median earnings during down cycles.",
        "warning": "Longer recovery windows imply weaker business elasticity under stress.",
        "benchmark_label": "Faster recovery is better",
        "format": "number",
    },
    "forward_incremental_roic_spread": {
        "display_name": "Forward Incremental ROIC Spread",
        "direction": "higher_is_better",
        "excellent": [0.06, float("inf")],
        "good": [0.04, 0.06],
        "neutral": [0.01, 0.04],
        "weak": [0.00, 0.01],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Projected spread between incremental ROIC and capital hurdle.",
        "warning": "Spread compression toward zero implies fading reinvestment quality.",
        "benchmark_label": ">=4pp strong | 1-4pp watch | <1pp weak",
        "format": "pct",
    },
    "base_incremental_roic": {
        "display_name": "Base Incremental ROIC",
        "direction": "higher_is_better",
        "excellent": [0.16, float("inf")],
        "good": [0.12, 0.16],
        "neutral": [0.08, 0.12],
        "weak": [0.00, 0.08],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Base-case incremental ROIC assumption anchored to historical experience.",
        "warning": "Use with spread metric to judge runway attractiveness.",
        "benchmark_label": ">=12% favorable base rate",
        "format": "pct",
    },
    "scenario_bear_growth": {
        "display_name": "Scenario Growth (Bear)",
        "direction": "higher_is_better",
        "excellent": [0.04, float("inf")],
        "good": [0.01, 0.04],
        "neutral": [0.00, 0.01],
        "weak": [-0.03, 0.00],
        "bad": [-float("inf"), -0.03],
        "plain_english": "Macro-conditioned downside growth assumption used in scenario analysis.",
        "warning": "Deeply negative bear growth increases financing and durability risk.",
        "benchmark_label": "Closer to zero or positive is more resilient",
        "format": "pct",
    },
    "scenario_base_growth": {
        "display_name": "Scenario Growth (Base)",
        "direction": "higher_is_better",
        "excellent": [0.08, float("inf")],
        "good": [0.04, 0.08],
        "neutral": [0.01, 0.04],
        "weak": [0.00, 0.01],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Central growth assumption generated from history and macro state.",
        "warning": "Base growth should be feasible without depending on aggressive margin expansion.",
        "benchmark_label": "Higher sustainable base growth supports compounding",
        "format": "pct",
    },
    "scenario_bull_growth": {
        "display_name": "Scenario Growth (Bull)",
        "direction": "higher_is_better",
        "excellent": [0.12, float("inf")],
        "good": [0.08, 0.12],
        "neutral": [0.04, 0.08],
        "weak": [0.01, 0.04],
        "bad": [-float("inf"), 0.01],
        "plain_english": "Upside growth assumption from the upper historical distribution band.",
        "warning": "Bull assumptions should remain within historically feasible ranges.",
        "benchmark_label": "Reference upside scenario, not base expectation",
        "format": "pct",
    },
    "bear_drawdown": {
        "display_name": "Bear-Case Drawdown",
        "direction": "higher_is_better",
        "excellent": [-0.10, float("inf")],
        "good": [-0.25, -0.10],
        "neutral": [-0.40, -0.25],
        "weak": [-0.55, -0.40],
        "bad": [-float("inf"), -0.55],
        "plain_english": "Maximum downside change in owner earnings per share under the bear scenario.",
        "warning": "Drawdowns beyond -40% imply fragile downside economics.",
        "benchmark_label": ">=-40% acceptable | <-40% weak",
        "format": "pct",
    },
    "operating_margin_cv": {
        "display_name": "Operating Margin Variability (CV)",
        "direction": "lower_is_better",
        "excellent": [0.0, 0.20],
        "good": [0.20, 0.25],
        "neutral": [0.25, 0.35],
        "weak": [0.35, 0.50],
        "bad": [0.50, float("inf")],
        "plain_english": "Coefficient of variation of operating margin across the observed history.",
        "warning": "Higher variability suggests weaker pricing power or operating instability.",
        "benchmark_label": "<=0.25 strong | >0.25 watch/weak",
        "format": "multiple",
    },
    "operating_margin_latest": {
        "display_name": "Operating Margin (Latest)",
        "direction": "higher_is_better",
        "excellent": [0.25, float("inf")],
        "good": [0.15, 0.25],
        "neutral": [0.08, 0.15],
        "weak": [0.00, 0.08],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Most recent operating margin level used in forward durability checks.",
        "warning": "Interpret alongside historical variability and feasible-band constraints.",
        "benchmark_label": "Higher stable margins indicate stronger pricing power",
        "format": "pct",
    },
    "historical_margin_p90": {
        "display_name": "Historical Margin P90",
        "direction": "higher_is_better",
        "excellent": [0.30, float("inf")],
        "good": [0.22, 0.30],
        "neutral": [0.15, 0.22],
        "weak": [0.08, 0.15],
        "bad": [-float("inf"), 0.08],
        "plain_english": "Upper feasible band for historical operating margin performance.",
        "warning": "Base-case requirements above this band often indicate unrealistic assumptions.",
        "benchmark_label": "Used as feasibility ceiling reference",
        "format": "pct",
    },
    "required_margin_base_case": {
        "display_name": "Required Margin (Base Case)",
        "direction": "higher_is_better",
        "excellent": [0.25, float("inf")],
        "good": [0.18, 0.25],
        "neutral": [0.12, 0.18],
        "weak": [0.06, 0.12],
        "bad": [-float("inf"), 0.06],
        "plain_english": "Operating margin needed in base case to satisfy forward assumptions.",
        "warning": "Must be checked against historical feasible bands to avoid over-optimism.",
        "benchmark_label": "Compare versus historical P90 for realism",
        "format": "pct",
    },
    "stress_interest_coverage": {
        "display_name": "Stress Interest Coverage",
        "direction": "higher_is_better",
        "excellent": [4.0, float("inf")],
        "good": [3.0, 4.0],
        "neutral": [2.0, 3.0],
        "weak": [1.0, 2.0],
        "bad": [-float("inf"), 1.0],
        "plain_english": "Coverage of interest expense under stressed earnings assumptions.",
        "warning": "Coverage below 2x under stress is a key liquidity/solvency warning.",
        "benchmark_label": ">3x strong | 2-3x watch | <2x weak",
        "format": "multiple",
    },
    "stress_net_debt_ebitda_proxy": {
        "display_name": "Stress Net Debt / EBITDA",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 1.5],
        "good": [1.5, 2.5],
        "neutral": [2.5, 3.5],
        "weak": [3.5, 5.0],
        "bad": [5.0, float("inf")],
        "plain_english": "Leverage proxy under a stressed EBITDA/earnings assumption.",
        "warning": "Higher stressed leverage reduces refinancing flexibility in downturns.",
        "benchmark_label": "<2.5x strong | >3.5x riskier",
        "format": "multiple",
    },
    "liquidity_runway_current_ratio": {
        "display_name": "Liquidity Runway (Current Ratio)",
        "direction": "higher_is_better",
        "excellent": [2.0, float("inf")],
        "good": [1.5, 2.0],
        "neutral": [1.2, 1.5],
        "weak": [1.0, 1.2],
        "bad": [-float("inf"), 1.0],
        "plain_english": "Current assets relative to current liabilities as a short-term liquidity proxy.",
        "warning": "Ratios below 1.0 can indicate near-term funding pressure.",
        "benchmark_label": ">=1.5 comfortable | <1.0 weak",
        "format": "multiple",
    },
    "implied_growth_reverse_dcf": {
        "display_name": "Implied Growth (Reverse DCF)",
        "direction": "lower_is_better",
        "excellent": [-float("inf"), 0.06],
        "good": [0.06, 0.10],
        "neutral": [0.10, 0.15],
        "weak": [0.15, 0.20],
        "bad": [0.20, float("inf")],
        "plain_english": "Growth rate embedded in the current price under reverse-DCF assumptions.",
        "warning": "Higher implied growth increases execution risk and narrows margin of safety.",
        "benchmark_label": "Lower implied growth is easier to underwrite",
        "format": "pct",
    },
    "historical_growth_p60": {
        "display_name": "Historical Growth P60",
        "direction": "higher_is_better",
        "excellent": [0.12, float("inf")],
        "good": [0.08, 0.12],
        "neutral": [0.04, 0.08],
        "weak": [0.00, 0.04],
        "bad": [-float("inf"), 0.00],
        "plain_english": "60th percentile of historical growth outcomes used as feasibility reference.",
        "warning": "Acts as a moderate historical feasibility anchor.",
        "benchmark_label": "Reference band, not a direct target",
        "format": "pct",
    },
    "historical_growth_p90": {
        "display_name": "Historical Growth P90",
        "direction": "higher_is_better",
        "excellent": [0.18, float("inf")],
        "good": [0.12, 0.18],
        "neutral": [0.08, 0.12],
        "weak": [0.03, 0.08],
        "bad": [-float("inf"), 0.03],
        "plain_english": "90th percentile of historical growth outcomes, representing a high-feasibility bound.",
        "warning": "Implied assumptions above this range often indicate stretched expectations.",
        "benchmark_label": "Upper feasibility reference band",
        "format": "pct",
    },
    "expected_return_no_multiple_expansion": {
        "display_name": "Expected Return (No Multiple Expansion)",
        "direction": "higher_is_better",
        "excellent": [0.15, float("inf")],
        "good": [0.12, 0.15],
        "neutral": [0.08, 0.12],
        "weak": [0.04, 0.08],
        "bad": [-float("inf"), 0.04],
        "plain_english": "Base expected shareholder return from cash yield, real growth, and dilution/friction only.",
        "warning": "If this is weak, the thesis may rely excessively on valuation multiple expansion.",
        "benchmark_label": ">=12% strong | 8-12% watch | <8% weak",
        "format": "pct",
    },
    "cash_yield": {
        "display_name": "Cash Yield",
        "direction": "higher_is_better",
        "excellent": [0.06, float("inf")],
        "good": [0.04, 0.06],
        "neutral": [0.02, 0.04],
        "weak": [0.01, 0.02],
        "bad": [-float("inf"), 0.01],
        "plain_english": "Shareholder cash return yield from dividends and buybacks relative to market value.",
        "warning": "Low cash yield increases reliance on future growth for total return.",
        "benchmark_label": "Higher cash yield improves return floor",
        "format": "pct",
    },
    "real_growth": {
        "display_name": "Real Growth",
        "direction": "higher_is_better",
        "excellent": [0.08, float("inf")],
        "good": [0.05, 0.08],
        "neutral": [0.02, 0.05],
        "weak": [0.00, 0.02],
        "bad": [-float("inf"), 0.00],
        "plain_english": "Inflation-adjusted owner-earnings growth contribution to expected return.",
        "warning": "Real growth is the primary long-run compounding driver beyond cash yield.",
        "benchmark_label": ">=5% healthy real compounding",
        "format": "pct",
    },
    "dilution_friction": {
        "display_name": "Dilution / Friction",
        "direction": "higher_is_better",
        "excellent": [0.02, float("inf")],
        "good": [0.00, 0.02],
        "neutral": [-0.01, 0.00],
        "weak": [-0.03, -0.01],
        "bad": [-float("inf"), -0.03],
        "plain_english": "Per-share ownership effect from share count changes (positive means anti-dilutive).",
        "warning": "Persistent dilution lowers owner-level compounding even when business growth looks solid.",
        "benchmark_label": "Positive is best | materially negative is dilution drag",
        "format": "pct",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_rating_colors(rating: str) -> Dict[str, str]:
    return {
        "text":   RATING_TEXT_COLORS.get(rating, RATING_TEXT_COLORS["N/A"]),
        "bg":     RATING_BG_COLORS.get(rating, RATING_BG_COLORS["N/A"]),
        "border": RATING_BORDER_COLORS.get(rating, RATING_BORDER_COLORS["N/A"]),
    }


def classify_metric(value: float, rule: Dict, sector: Optional[str] = None) -> str:
    """
    Return Excellent / Good / Neutral / Weak / Bad.
    Applies light sector adjustments to PE and EV/EBITDA.
    """
    direction = rule.get("direction", "higher_is_better")
    if direction == "context_dependent":
        return "N/A"

    adj = _get_sector_adj(sector)
    # Adjust PE thresholds
    pe_mult = adj.get("pe_mult", 1.0)
    ev_mult = adj.get("debt_ebitda_mult", 1.0) if "debt_ebitda" not in rule.get("display_name", "").lower() else adj.get("debt_ebitda_mult", 1.0)

    def _adjust(bounds: Optional[list], mult: float) -> Optional[list]:
        if bounds is None:
            return None
        lo, hi = bounds
        if lo != -float("inf"):
            lo = lo * mult
        if hi != float("inf"):
            hi = hi * mult
        return [lo, hi]

    # Determine if sector PE mult applies
    display = rule.get("display_name", "")
    is_pe_metric = any(kw in display for kw in ["P/E", "Forward P/E"])
    is_ev_metric = "EV / EBITDA" in display
    is_ps_metric = "P/S" in display
    ps_mult = adj.get("ps_mult", 1.0)

    for rating in ["excellent", "good", "neutral", "weak", "bad"]:
        bounds = rule.get(rating)
        if bounds is None:
            continue
        if is_pe_metric and pe_mult != 1.0:
            bounds = _adjust(bounds, pe_mult)
        elif is_ev_metric and pe_mult != 1.0:
            bounds = _adjust(bounds, pe_mult)
        elif is_ps_metric and ps_mult != 1.0:
            bounds = _adjust(bounds, ps_mult)

        lo, hi = bounds
        if lo <= value <= hi:
            return rating.capitalize()

    # Fallback
    if direction == "higher_is_better":
        bad_bounds = rule.get("bad", [-float("inf"), 0])
        return "Bad" if value < bad_bounds[1] else "Excellent"
    else:
        bad_bounds = rule.get("bad", [0, float("inf")])
        return "Bad" if value > bad_bounds[0] else "Excellent"


def interpret_metric(
    metric_key: str,
    value: Optional[float],
    sector: Optional[str] = None,
) -> Dict[str, str]:
    """
    Full interpretation for one metric.
    Returns dict: display_name, rating, color_text/bg/border,
                  plain_english, direction, benchmark_label, comment.
    """
    rule = METRIC_RULES.get(metric_key)
    if rule is None or value is None:
        return {
            "display_name": metric_key.replace("_", " ").title(),
            "rating": "N/A",
            "color_text": RATING_TEXT_COLORS["N/A"],
            "color_bg": RATING_BG_COLORS["N/A"],
            "color_border": RATING_BORDER_COLORS["N/A"],
            "plain_english": "Definition not available.",
            "direction": "N/A",
            "benchmark_label": "N/A",
            "comment": "Not enough data to interpret this metric.",
        }

    rating = classify_metric(value, rule, sector)
    colors = get_rating_colors(rating)
    comment = _build_comment(rule, rating)

    return {
        "display_name": rule["display_name"],
        "rating": rating,
        "color_text": colors["text"],
        "color_bg": colors["bg"],
        "color_border": colors["border"],
        "plain_english": rule.get("plain_english", ""),
        "direction": rule.get("direction", "N/A"),
        "benchmark_label": rule.get("benchmark_label", "General rules of thumb"),
        "comment": comment,
    }


def _build_comment(rule: Dict, rating: str) -> str:
    display = rule.get("display_name", "this metric")
    warning = rule.get("warning", "")
    prefixes = {
        "Excellent": f"{display} is in excellent territory.",
        "Good":      f"{display} is healthy.",
        "Neutral":   f"{display} is acceptable but warrants monitoring.",
        "Weak":      f"{display} is below average — a caution flag.",
        "Bad":       f"{display} is in concerning territory — a red flag.",
        "N/A":       "",
    }
    prefix = prefixes.get(rating, "")
    if not prefix:
        return warning or "Insufficient data."
    return f"{prefix} {warning}".strip() if warning else prefix


def build_badge_html(rating: str) -> str:
    """Colored pill badge for a rating."""
    colors = get_rating_colors(rating)
    icons = {
        "Excellent": "★",
        "Good":      "✓",
        "Neutral":   "~",
        "Weak":      "▲",
        "Bad":       "✗",
        "N/A":       "?",
    }
    icon = icons.get(rating, "")
    return (
        f'<span style="display:inline-block;background:{colors["bg"]};'
        f'color:{colors["text"]};border:1px solid {colors["border"]};'
        f'padding:2px 9px;border-radius:12px;font-size:11px;font-weight:700;'
        f'white-space:nowrap">{icon} {rating}</span>'
    )


def build_interpretation_table_html(
    metric_entries: list,
    sector: Optional[str] = None,
) -> str:
    """
    Build a full interpretation guide table.

    metric_entries: list of (metric_key, formatted_value_str, raw_value)
    Returns an HTML string.
    """
    if not metric_entries:
        return ""

    rows_html = ""
    for metric_key, fmt_value, raw_value in metric_entries:
        interp = interpret_metric(metric_key, raw_value, sector)
        badge = build_badge_html(interp["rating"])
        direction_icons = {
            "higher_is_better": "↑ Higher better",
            "lower_is_better":  "↓ Lower better",
            "context_dependent": "⇔ Context",
        }
        dir_label = direction_icons.get(interp["direction"], interp["direction"])

        rows_html += f"""
        <tr style="border-bottom:1px solid #ecf0f1;vertical-align:top">
          <td style="padding:7px 10px;font-weight:600;color:#2c3e50;white-space:nowrap">
            {interp['display_name']}
            <div style="font-size:10px;color:#aab7b8;font-weight:400;margin-top:2px">{dir_label}</div>
          </td>
          <td style="padding:7px 10px;font-weight:700;color:#2c3e50;text-align:right;white-space:nowrap">
            {fmt_value}
          </td>
          <td style="padding:7px 10px;text-align:center">
            {badge}
          </td>
          <td style="padding:7px 10px;color:#566573;font-size:12px;max-width:180px">
            {interp['plain_english']}
          </td>
          <td style="padding:7px 10px;color:#7f8c8d;font-size:11px;font-style:italic;word-wrap:break-word;overflow-wrap:break-word;min-width:120px;max-width:180px">
            {interp['benchmark_label']}
          </td>
          <td style="padding:7px 10px;color:#566573;font-size:12px;word-wrap:break-word;overflow-wrap:break-word;min-width:160px;max-width:240px">
            {interp['comment']}
          </td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto;margin-top:14px">
      <table style="width:100%;border-collapse:collapse;font-size:13px;min-width:700px;table-layout:fixed">
        <colgroup>
          <col style="width:18%">
          <col style="width:9%">
          <col style="width:10%">
          <col style="width:25%">
          <col style="width:18%">
          <col style="width:20%">
        </colgroup>
        <thead>
          <tr style="background:#ecf0f1;border-bottom:2px solid #d5d8dc">
            <th style="padding:8px 10px;text-align:left;color:#2c3e50;font-size:12px">Metric</th>
            <th style="padding:8px 10px;text-align:right;color:#2c3e50;font-size:12px">Value</th>
            <th style="padding:8px 10px;text-align:center;color:#2c3e50;font-size:12px">Rating</th>
            <th style="padding:8px 10px;text-align:left;color:#2c3e50;font-size:12px">What it means</th>
            <th style="padding:8px 10px;text-align:left;color:#2c3e50;font-size:12px">Benchmark</th>
            <th style="padding:8px 10px;text-align:left;color:#2c3e50;font-size:12px">Comment</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def color_legend_html() -> str:
    """Compact legend bar explaining the 5-tier scoring color system."""
    items = [
        ("Excellent", "8.0–10.0 — outstanding"),
        ("Good",      "6.5–7.9 — healthy"),
        ("Neutral",   "5.0–6.4 — acceptable, watch"),
        ("Weak",      "3.5–4.9 — below average, caution"),
        ("Bad",       "0.0–3.4 — concerning, investigate"),
        ("N/A",       "missing / insufficient data"),
    ]
    badges = " &nbsp; ".join(build_badge_html(r) + f'&thinsp;<span style="font-size:11px;color:#566573">{desc}</span>' for r, desc in items)
    return f"""
    <div style="background:#fafafa;border:1px solid #ecf0f1;border-radius:8px;
                padding:10px 16px;margin:12px 0;display:flex;flex-wrap:wrap;
                gap:8px;align-items:center">
      <span style="font-size:12px;font-weight:700;color:#2c3e50;margin-right:8px">Scoring:</span>
      {badges}
    </div>"""
