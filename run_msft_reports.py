"""
Runner: generates present, past, and future reports for MSFT.
Used for visual-improvements branch development work.
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env so FRED_API_KEY and other secrets are available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

TICKER         = "MSFT"
OWNERSHIP_MODE = "new"
FORCE_REFRESH  = False
OPEN_REPORT    = False   # set True to auto-open each report

NOTES = """
Microsoft has one of the strongest fundamental profiles in the market:
- Azure cloud platform growing ~30% YoY, gaining share on AWS
- Office 365 / M365 subscription model drives durable recurring revenue
- ~40% operating margins (one of the highest in large-cap tech)
- AAA credit rating — one of only two US companies with this rating
- Copilot AI integration across the entire product suite
- Net cash position; aggressive shareholder returns via buybacks + dividend
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from src.data_fetcher       import StockData, fetch_multiple
from src.metrics_calculator import MetricsCalculator
from src.threshold_manager  import ThresholdManager
from src.visualizations     import build_all_charts, price_history_chart
from src.panel_generator    import PanelGenerator
from src.report_generator   import generate_report
from src.first_principles   import FirstPrinciplesEngine
from src.utils              import parse_date

# ── Fetch data ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Fetching data for {TICKER}...")
print(f"{'='*60}")
stock = StockData(TICKER, force_refresh=FORCE_REFRESH)
print(f"✅ {stock.staleness_label()}")

metrics = MetricsCalculator(stock=stock, mode=OWNERSHIP_MODE).compute()
print(f"✅ Metrics computed")

tm = ThresholdManager(sector=metrics.q1.sector)

metrics._history_df = stock.history
metrics._divs_series = stock.dividends

charts = build_all_charts(metrics, tm)
charts["price_history"] = price_history_chart(
    history=stock.history,
    ticker=metrics.ticker,
    purchase_date=None,
    cost_basis=None,
    watermark=metrics.ticker,
)
print(f"✅ Charts built")

pg = PanelGenerator(
    metrics=metrics,
    tm=tm,
    target_allocation=None,
    max_allocation=None,
    severe_overweight=None,
)

panel_bullish   = pg.panel_bullish()
panel_hold      = pg.panel_hold()
panel_bearish   = pg.panel_bearish()
panel_dividends = pg.panel_dividends(dividend_series=stock.dividends)
panel_portfolio = pg.panel_portfolio_risk()
panel_ml        = pg.panel_ml_predictions()

recommendation, rec_explanation, rec_audit = pg.final_recommendation(
    bullish=panel_bullish,
    hold=panel_hold,
    bearish=panel_bearish,
)
panel_audit = pg.panel_decision_audit(rec_audit)

print(f"✅ Panels built — Recommendation: {recommendation}")
peer_metrics = {}

# ── PRESENT report ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Generating PRESENT report...")
print(f"{'='*60}")

panels = {
    "decision_audit": panel_audit,
    "bullish":        panel_bullish,
    "hold":           panel_hold,
    "bearish":        panel_bearish,
    "dividends":      panel_dividends,
}
if panel_ml:
    panels["ml"] = panel_ml

present_path = generate_report(
    metrics=metrics,
    panels=panels,
    recommendation=recommendation,
    recommendation_explanation=rec_explanation,
    charts=charts,
    open_in_browser=OPEN_REPORT,
    peer_metrics=None,
    notes=NOTES,
)
print(f"✅ PRESENT → {present_path}")

# ── PAST + FUTURE reports (FirstPrinciplesEngine) ────────────────────────────
fp_engine = FirstPrinciplesEngine(
    force_refresh=FORCE_REFRESH,
    fred_api_key=os.getenv("FRED_API_KEY"),
)

print(f"\n{'='*60}")
print(f"  Running FirstPrinciples engine for PAST + FUTURE...")
print(f"{'='*60}")
fp_report = fp_engine.run(TICKER)
print(f"✅ Engine complete — Decision: {fp_report.synthesis.decision}")

fp_explanation = (
    f"First-principles signal mix. "
    f"Strong {fp_report.synthesis.strong_count}, "
    f"Watch {fp_report.synthesis.watch_count}, "
    f"Weak {fp_report.synthesis.weak_count}, "
    f"Insufficient {fp_report.synthesis.insufficient_count}."
)

for view in ("past", "future"):
    print(f"\n  Generating {view.upper()} report...")
    path = generate_report(
        metrics=metrics,
        panels={},
        recommendation=fp_report.synthesis.decision,
        recommendation_explanation=fp_explanation,
        charts={},
        open_in_browser=OPEN_REPORT,
        peer_metrics=None,
        notes=NOTES,
        report_view=view,
        first_principles_report=fp_report,
    )
    print(f"  ✅ {view.upper()} → {path}")

print(f"\n{'='*60}")
print(f"  All 3 MSFT reports generated successfully!")
print(f"{'='*60}\n")
