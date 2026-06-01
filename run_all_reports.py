"""
Runner: generates all 6 report variants for one ticker.

Outputs written to outputs/:
- new: present, past, future
- existing: present, past, future
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Tuple

# Ensure local imports resolve when running from project root.
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from src.data_fetcher import StockData
from src.first_principles import FirstPrinciplesEngine
from src.metrics_calculator import AllMetrics, MetricsCalculator
from src.panel_generator import PanelGenerator
from src.report_generator import generate_report, generate_master_report
from src.threshold_manager import ThresholdManager
from src.visualizations import build_all_charts, price_history_chart


# ---- Configure here ---------------------------------------------------------
TICKER = "KO"
FORCE_REFRESH = False
OPEN_REPORTS = False

# Existing-holding defaults used when user-specific values are unavailable.
EXISTING_PURCHASE_DATE = date.today() - timedelta(days=365)
EXISTING_SHARES_OWNED = 100.0
EXISTING_TOTAL_PORTFOLIO = 250_000.0

NOTES = (
    "Auto-generated pack: present/past/future reports for both new and existing "
    "holding modes."
)
# -----------------------------------------------------------------------------


def _build_metrics(stock: StockData, mode: str) -> AllMetrics:
    if mode == "existing":
        hist = stock.history
        if hist is not None and not hist.empty and "Close" in hist.columns:
            idx = hist.index
            target_dt = idx.tz_localize(None) if getattr(idx, "tz", None) else idx
            target_date = EXISTING_PURCHASE_DATE
            nearest_pos = min(range(len(target_dt)), key=lambda i: abs((target_dt[i].date() - target_date).days))
            inferred_cost_basis = float(hist["Close"].iloc[nearest_pos])
        else:
            inferred_cost_basis = None

        mc = MetricsCalculator(
            stock=stock,
            mode="existing",
            purchase_date=EXISTING_PURCHASE_DATE,
            cost_basis=inferred_cost_basis,
            shares_owned=EXISTING_SHARES_OWNED,
            total_portfolio_value=EXISTING_TOTAL_PORTFOLIO,
            notes=NOTES,
        )
        return mc.compute()

    return MetricsCalculator(stock=stock, mode="new", notes=NOTES).compute()


def _build_present_artifacts(metrics: AllMetrics, stock: StockData):
    tm = ThresholdManager(sector=metrics.q1.sector)

    metrics._history_df = stock.history
    metrics._divs_series = stock.dividends

    purchase_date_for_report = metrics.portfolio.purchase_date
    if metrics.mode == "existing":
        # Prevent Plotly categorical-vline bug in add_vline for some histories.
        metrics.portfolio.purchase_date = None

    charts = build_all_charts(metrics, tm)

    if metrics.mode == "existing":
        metrics.portfolio.purchase_date = purchase_date_for_report

    charts["price_history"] = price_history_chart(
        history=stock.history,
        ticker=metrics.ticker,
        purchase_date=None,
        cost_basis=metrics.portfolio.cost_basis,
        watermark=metrics.ticker,
    )

    pg = PanelGenerator(
        metrics=metrics,
        tm=tm,
        target_allocation=0.05,
        max_allocation=0.10,
        severe_overweight=0.15,
    )

    panel_bullish = pg.panel_bullish()
    panel_hold = pg.panel_hold()
    panel_bearish = pg.panel_bearish()
    panel_dividends = pg.panel_dividends(dividend_series=stock.dividends)
    panel_portfolio = pg.panel_portfolio_risk()
    panel_ml = pg.panel_ml_predictions()

    recommendation, explanation, audit = pg.final_recommendation(
        bullish=panel_bullish,
        hold=panel_hold,
        bearish=panel_bearish,
    )
    panel_audit = pg.panel_decision_audit(audit)

    panels = {
        "decision_audit": panel_audit,
        "bullish": panel_bullish,
        "hold": panel_hold,
        "bearish": panel_bearish,
        "dividends": panel_dividends,
    }
    if panel_portfolio and metrics.mode == "existing":
        panels["portfolio"] = panel_portfolio
    if panel_ml:
        panels["ml"] = panel_ml

    return panels, charts, recommendation, explanation


def main() -> None:
    print("\n" + "=" * 68)
    print(f"Generating full report pack for {TICKER}")
    print("=" * 68)

    stock = StockData(TICKER, force_refresh=FORCE_REFRESH)
    print(f"Data loaded: {stock.staleness_label()}")

    fp_engine = FirstPrinciplesEngine(
        force_refresh=FORCE_REFRESH,
        fred_api_key=os.getenv("FRED_API_KEY"),
    )
    fp_report = fp_engine.run(TICKER)
    fp_explanation = (
        f"First-principles signal mix: Strong {fp_report.synthesis.strong_count}, "
        f"Watch {fp_report.synthesis.watch_count}, Weak {fp_report.synthesis.weak_count}, "
        f"Insufficient {fp_report.synthesis.insufficient_count}."
    )

    outputs: Dict[Tuple[str, str], Path] = {}
    mode_artifacts = {}  # mode -> (metrics, panels, charts, recommendation, explanation)

    for mode in ("new", "existing"):
        print("\n" + "-" * 68)
        print(f"Mode: {mode}")
        print("-" * 68)

        metrics = _build_metrics(stock, mode)
        panels, charts, recommendation, explanation = _build_present_artifacts(metrics, stock)
        mode_artifacts[mode] = (metrics, panels, charts, recommendation, explanation)

        present_path = generate_report(
            metrics=metrics,
            panels=panels,
            recommendation=recommendation,
            recommendation_explanation=explanation,
            charts=charts,
            open_in_browser=OPEN_REPORTS,
            peer_metrics=None,
            notes=NOTES,
            report_view="present",
        )
        outputs[(mode, "present")] = present_path

        for view in ("past", "future"):
            view_path = generate_report(
                metrics=metrics,
                panels={},
                recommendation=fp_report.synthesis.decision,
                recommendation_explanation=fp_explanation,
                charts={},
                open_in_browser=OPEN_REPORTS,
                peer_metrics=None,
                notes=NOTES,
                report_view=view,
                first_principles_report=fp_report,
            )
            outputs[(mode, view)] = view_path

    print("\n" + "=" * 68)
    print("Generated report files:")
    for mode in ("new", "existing"):
        for view in ("present", "past", "future"):
            print(f"  {mode:8s} {view:7s} -> {outputs[(mode, view)]}")
    print("=" * 68 + "\n")

    # ── Master report ─────────────────────────────────────────────────────────
    print("-" * 68)
    print("Generating master report...")
    print("-" * 68)
    new_m, new_p, new_c, new_rec, new_exp = mode_artifacts["new"]
    ex_m, ex_p, ex_c, ex_rec, ex_exp = mode_artifacts["existing"]
    master_path = generate_master_report(
        new_metrics=new_m,
        new_panels=new_p,
        new_recommendation=new_rec,
        new_recommendation_explanation=new_exp,
        new_charts=new_c,
        existing_metrics=ex_m,
        existing_panels=ex_p,
        existing_recommendation=ex_rec,
        existing_recommendation_explanation=ex_exp,
        existing_charts=ex_c,
        first_principles_report=fp_report,
        notes=NOTES,
        open_in_browser=OPEN_REPORTS,
    )
    print(f"  master -> {master_path}\n")

if __name__ == "__main__":
    main()
