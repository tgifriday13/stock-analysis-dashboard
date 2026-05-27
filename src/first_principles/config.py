"""Configuration, mappings, and question specs for first-principles engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"
STOOQ_PRICE_URL = "https://stooq.com/q/d/l/"

DEFAULT_ACCEPTANCE_BUFFER_DAYS = 1
DEFAULT_MIN_COVERAGE = 0.70
DEFAULT_PERCENTILE_WINSOR_LIMIT = 0.05

FRED_SERIES = {
    "risk_free_10y": "DGS10",
    "baa_spread": "BAA10Y",
    "cpi": "CPIAUCSL",
    "recession": "USREC",
}

FLOW_FIELDS = {
    "revenue",
    "net_income",
    "operating_cash_flow",
    "capex",
    "operating_income",
    "interest_expense",
    "income_tax_expense",
    "dividends_paid",
    "repurchases",
}

# Canonical field -> ordered tag precedence (taxonomy, tag)
CANONICAL_TAG_PRIORITY: Dict[str, List[Dict[str, str]]] = {
    "revenue": [
        {"taxonomy": "us-gaap", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax"},
        {"taxonomy": "us-gaap", "tag": "SalesRevenueNet"},
        {"taxonomy": "us-gaap", "tag": "Revenues"},
    ],
    "net_income": [
        {"taxonomy": "us-gaap", "tag": "NetIncomeLoss"},
        {"taxonomy": "us-gaap", "tag": "ProfitLoss"},
    ],
    "operating_income": [
        {"taxonomy": "us-gaap", "tag": "OperatingIncomeLoss"},
    ],
    "operating_cash_flow": [
        {"taxonomy": "us-gaap", "tag": "NetCashProvidedByUsedInOperatingActivities"},
        {"taxonomy": "us-gaap", "tag": "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"},
    ],
    "capex": [
        {"taxonomy": "us-gaap", "tag": "PaymentsToAcquirePropertyPlantAndEquipment"},
        {"taxonomy": "us-gaap", "tag": "PropertyPlantAndEquipmentAdditions"},
    ],
    "total_assets": [
        {"taxonomy": "us-gaap", "tag": "Assets"},
    ],
    "total_liabilities": [
        {"taxonomy": "us-gaap", "tag": "Liabilities"},
    ],
    "total_equity": [
        {"taxonomy": "us-gaap", "tag": "StockholdersEquity"},
        {"taxonomy": "us-gaap", "tag": "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"},
    ],
    "cash_and_equivalents": [
        {"taxonomy": "us-gaap", "tag": "CashAndCashEquivalentsAtCarryingValue"},
        {"taxonomy": "us-gaap", "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"},
    ],
    "total_debt": [
        {"taxonomy": "us-gaap", "tag": "DebtAndFinanceLeaseLiabilities"},
        {"taxonomy": "us-gaap", "tag": "LongTermDebtAndFinanceLeaseLiabilities"},
        {"taxonomy": "us-gaap", "tag": "LongTermDebt"},
    ],
    "current_assets": [
        {"taxonomy": "us-gaap", "tag": "AssetsCurrent"},
    ],
    "current_liabilities": [
        {"taxonomy": "us-gaap", "tag": "LiabilitiesCurrent"},
    ],
    "interest_expense": [
        {"taxonomy": "us-gaap", "tag": "InterestExpense"},
    ],
    "income_tax_expense": [
        {"taxonomy": "us-gaap", "tag": "IncomeTaxExpenseBenefit"},
    ],
    "diluted_shares": [
        {"taxonomy": "us-gaap", "tag": "WeightedAverageNumberOfDilutedSharesOutstanding"},
        {"taxonomy": "us-gaap", "tag": "WeightedAverageNumberOfSharesOutstandingDiluted"},
    ],
    "dividends_paid": [
        {"taxonomy": "us-gaap", "tag": "PaymentsOfDividendsCommonStock"},
        {"taxonomy": "us-gaap", "tag": "PaymentsOfDividends"},
    ],
    "repurchases": [
        {"taxonomy": "us-gaap", "tag": "PaymentsForRepurchaseOfCommonStock"},
    ],
}


@dataclass
class QuestionSpec:
    """Question-level config used by the scoring engine."""

    question_id: str
    question: str
    metric_family: List[str]
    required_fields: List[str]
    min_coverage: float
    thresholds: Dict[str, Any]
    falsification: str


BASE_QUESTION_SPECS: List[QuestionSpec] = [
    QuestionSpec(
        question_id="P1",
        question="Did per-share owner earnings compound in real terms?",
        metric_family=["real_owner_earnings_cagr", "real_fcf_per_share_cagr", "real_revenue_per_share_cagr"],
        required_fields=["net_income", "operating_cash_flow", "capex", "diluted_shares", "revenue", "cpi"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong": 0.06, "watch": 0.02, "decline_breach": -0.20},
        falsification="Real 5Y owner-earnings CAGR <= 0% or cumulative per-share decline > 20%",
    ),
    QuestionSpec(
        question_id="P2",
        question="Did incremental capital earn attractive returns?",
        metric_family=["incremental_roic", "reinvestment_efficiency"],
        required_fields=["operating_income", "income_tax_expense", "total_equity", "total_debt", "cash_and_equivalents"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong": 0.12, "watch": 0.08, "years_below_hurdle": 2},
        falsification="Median 5Y incremental ROIC below hurdle for 2+ years",
    ),
    QuestionSpec(
        question_id="P3",
        question="Were earnings cash-backed (low accrual distortion)?",
        metric_family=["accrual_ratio", "cfo_to_ni", "working_cap_accrual_proxy"],
        required_fields=["net_income", "operating_cash_flow", "total_assets", "current_assets", "current_liabilities"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"accrual_strong": 0.03, "accrual_fail": 0.08, "cfo_to_ni_fail": 0.8, "cfo_to_ni_strong": 1.0},
        falsification="Accrual ratio > 8% for 2 years or CFO/NI < 0.8",
    ),
    QuestionSpec(
        question_id="P4",
        question="Was growth funded internally (not by dilution/leverage creep)?",
        metric_family=["dilution_adjusted_growth", "share_cagr", "net_debt_trend"],
        required_fields=["revenue", "diluted_shares", "total_debt", "cash_and_equivalents", "operating_cash_flow", "capex"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong_share_cagr": 0.005, "watch_share_cagr": 0.02},
        falsification="Share CAGR > 2% while FCF weak, or leverage trend deteriorates materially",
    ),
    QuestionSpec(
        question_id="P5",
        question="Did capital allocation create value?",
        metric_family=["fcf_payout_coverage", "buyback_quality", "reinvestment_vs_payout_mix"],
        required_fields=["operating_cash_flow", "capex", "dividends_paid", "repurchases", "total_debt", "cash_and_equivalents"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"coverage_strong": 1.5, "coverage_watch": 1.0},
        falsification="Payout financed by balance-sheet stress or persistent value-destructive pattern",
    ),
    QuestionSpec(
        question_id="P6",
        question="Was the business resilient in stress regimes?",
        metric_family=["min_fcf_margin", "interest_coverage_trough", "recovery_speed"],
        required_fields=["operating_cash_flow", "capex", "revenue", "operating_income", "interest_expense", "recession"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong": 3.0, "watch": 2.0},
        falsification="Breach of solvency-like thresholds during stress windows",
    ),
    QuestionSpec(
        question_id="F1",
        question="Is reinvestment runway still attractive?",
        metric_family=["forward_incremental_roic_spread", "reinvestment_capacity"],
        required_fields=["operating_income", "income_tax_expense", "total_equity", "total_debt", "cash_and_equivalents"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong_spread": 0.04, "watch_spread": 0.01},
        falsification="Forward spread to hurdle collapses in base case",
    ),
    QuestionSpec(
        question_id="F2",
        question="What is the 5Y owner-earnings distribution?",
        metric_family=["owner_earnings_distribution", "fcf_per_share_distribution"],
        required_fields=["operating_cash_flow", "capex", "diluted_shares", "cpi", "recession", "baa_spread"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"max_bear_drawdown": -0.40},
        falsification="Bear case implies persistent cash deficit or forced financing",
    ),
    QuestionSpec(
        question_id="F3",
        question="How durable are margins/pricing power?",
        metric_family=["margin_persistence", "margin_variability", "pass_through_proxy"],
        required_fields=["revenue", "operating_income", "cpi"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"max_cv": 0.25},
        falsification="Required margin in base case exceeds historical feasible band",
    ),
    QuestionSpec(
        question_id="F4",
        question="Is funding/liquidity resilient under slowdown?",
        metric_family=["stress_net_debt_ebitda", "stress_interest_coverage", "liquidity_runway"],
        required_fields=["total_debt", "cash_and_equivalents", "current_assets", "current_liabilities", "operating_income", "interest_expense"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong": 3.0, "watch": 2.0},
        falsification="Stress coverage <2x or liquidity runway breach",
    ),
    QuestionSpec(
        question_id="F5",
        question="What growth is implied by today’s price?",
        metric_family=["implied_growth", "implied_margin", "feasibility_band"],
        required_fields=["price", "diluted_shares", "operating_cash_flow", "capex", "risk_free_10y"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong_percentile": 0.60, "weak_percentile": 0.90},
        falsification="Implied assumptions exceed historical and sector-feasible bounds",
    ),
    QuestionSpec(
        question_id="F6",
        question="What is expected shareholder return without multiple expansion?",
        metric_family=["return_decomposition", "cash_yield", "real_growth", "dilution_drag"],
        required_fields=["price", "diluted_shares", "dividends_paid", "repurchases", "operating_cash_flow", "capex", "cpi"],
        min_coverage=DEFAULT_MIN_COVERAGE,
        thresholds={"strong": 0.12, "watch": 0.08},
        falsification="Return target met only via multiple expansion",
    ),
]


FINANCIAL_TEMPLATE_ADJUSTMENTS = {
    "P6": {"thresholds": {"strong": 2.5, "watch": 1.5}},
    "F4": {"thresholds": {"strong": 2.5, "watch": 1.5}},
}


def is_financial_sector(sector: str) -> bool:
    """Heuristic sector classifier for template selection."""
    s = (sector or "").lower()
    keywords = ("financial", "bank", "insurance", "capital market", "asset management")
    return any(k in s for k in keywords)


def get_question_specs(template: str = "non_financial") -> List[QuestionSpec]:
    """Return question specs with template-specific threshold adjustments."""
    specs = []
    for base in BASE_QUESTION_SPECS:
        spec = QuestionSpec(
            question_id=base.question_id,
            question=base.question,
            metric_family=list(base.metric_family),
            required_fields=list(base.required_fields),
            min_coverage=base.min_coverage,
            thresholds=dict(base.thresholds),
            falsification=base.falsification,
        )
        specs.append(spec)

    if template == "financial":
        by_id = {s.question_id: s for s in specs}
        for qid, patch in FINANCIAL_TEMPLATE_ADJUSTMENTS.items():
            if qid in by_id and "thresholds" in patch:
                by_id[qid].thresholds.update(patch["thresholds"])

    return specs
