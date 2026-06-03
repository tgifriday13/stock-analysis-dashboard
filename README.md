# Stock Fundamentals CEO Dashboard

A Python notebook that answers **6 hyper-specific fundamental questions** about any stock and generates a plain-English **HTML CEO Report** — for any sector, any company age, dividend or non-dividend.

## Quickstart

```bash
# 1. Clone the project
git clone https://github.com/tgifriday13/stock-analysis-dashboard.git
cd stock-analysis-dashboard

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt          # production only
pip install -r requirements-dev.txt     # add notebook + dev tools

# 4. Copy and fill in your environment variables
cp .env.example .env
# (edit .env with your SEC_USER_AGENT and FRED_API_KEY)

# 5. Open the notebook
jupyter notebook notebooks/stock_analysis.ipynb
# OR with JupyterLab:
jupyter lab notebooks/stock_analysis.ipynb
```

Then **edit Cell 1** (Configuration) and **Run All Cells**. A CEO Report opens in your browser automatically.

---

## The 6 Core Questions

These drive every panel, every visual, and the final recommendation:

| # | Question |
|---|----------|
| 1 | Is this business at a **meaningful scale** for my portfolio, and has that scale improved or deteriorated since I bought? |
| 2 | Is the business **growing** at a rate that will compound my portfolio value, and has growth accelerated, stayed stable, or slowed since I bought? |
| 3 | Are profits **high-quality, stable, and ideally expanding**, and have margins held or improved since I bought? |
| 4 | Is the company generating reliable **free cash flow** that can be returned to owners or reinvested at high returns, and has cash flow strength improved since I bought? |
| 5 | Is the **balance sheet strong** enough to survive shocks without hurting my portfolio, and has financial risk increased or decreased since I bought? |
| 6 | Is the current price **cheap enough** to provide a margin of safety for my portfolio, and is the stock cheaper or more expensive than when I bought? |

---

## Ownership Modes

### New Position
Set `OWNERSHIP_MODE = "new"` in Cell 1.

**Final recommendation:** `Strong Buy` / `Buy` / `Hold / Monitor` / `Avoid`

### Existing Holding
Set `OWNERSHIP_MODE = "existing"` and fill in:
- `PURCHASE_DATE` — when you bought (YYYY-MM-DD)
- `COST_BASIS` — your average cost per share
- `SHARES_OWNED` — number of shares
- `TOTAL_PORTFOLIO` — total portfolio value (for position weight calculation)

**Final recommendation:** `Buy More` / `Hold` / `Trim` / `Sell`

Every metric shows a **"now vs. when you bought"** delta, so you can see exactly how the business has changed since your purchase.

---

## Features

### Data
- Fetched from **yfinance** with automatic disk caching (`data/cache/`)
- Cache refreshes automatically if older than 24 hours
- Every metric is time-stamped: *"Data as of YYYY-MM-DD HH:MM PDT — last filing X days ago"*

### Smart Thresholds
- **Sector-adjusted defaults** — Technology gets higher P/E tolerance; Financials get higher debt tolerance; etc.
- **User overrides** — set any threshold in Cell 1 via `THRESHOLD_OVERRIDES`
- **Saved profiles** — save your personal threshold set with `tm.save("my_profile")` and reload with `THRESHOLD_PROFILE = "my_profile"`

### 5 Analysis Panels
1. **Why Buy / Why Buy More** — bullish signals from all 6 questions
2. **Why Hold** — neutral/mixed signals
3. **Why Sell / Why Trim** — bearish risk signals
4. **Dividends & Shareholder Returns** — yield, payout ratio, FCF coverage, consecutive years, buybacks; shows N/A gracefully for non-payers
5. **Portfolio Risk Summary** — position P&L, annualized return, portfolio weight, "now vs. purchase" delta table *(Existing Holding mode only)*

### ML Predictions Panel (optional)
Enable with `ENABLE_ML_PREDICTIONS = True`. Uses scikit-learn to estimate:
- **Fair-value price range** based on ROIC, growth, margins, valuation
- **Probability of 20%+ outperformance** over next 12 months

All outputs are **clearly labeled as predictions** — not financial advice.

### CEO Report
- One self-contained **HTML file** saved to `outputs/`
- Auto-opens in your browser
- Includes: executive summary, recommendation box, scorecard, all 6 questions with charts, all panels, peer comparison (optional), analyst notes
- Readable by a CEO who knows nothing about the code

### Multi-Stock Comparison
Optional Cell 10 compares a list of tickers side by side in a markdown table.

### First-Principles Past/Future Engine (new)
- Module path: `src/first_principles/`
- Purpose: run 12 first-principles questions (Past + Future) with strict SEC acceptance-date point-in-time rules.
- Free sources only: SEC EDGAR APIs, FRED macro series, yfinance (with Stooq fallback for prices).
- Includes Kenneth French factor ingestion (`RMW`, `CMA`) for profitability/investment regime reference bands.
- Notebook tab selector:
  - `REPORT_TAB = "present"` → classic CEO report
  - `REPORT_TAB = "past"` → first-principles `P1–P6` only
  - `REPORT_TAB = "future"` → first-principles `F1–F6` only
- Output artifacts:
  - JSON audit report with per-question metrics, falsification status, coverage score.
  - HTML audit-card report under `outputs/first_principles/`.

Quick usage:
```python
from src.first_principles import FirstPrinciplesEngine

engine = FirstPrinciplesEngine(force_refresh=False)
report = engine.run("MSFT")
paths = engine.save_report(report)
print(paths)
```

Or from the command line:
```bash
python scripts/run_all_reports.py   # all 6 report variants for configured ticker
python scripts/run_msft_reports.py  # MSFT present/past/future reports
# OR using make:
make run
make run-msft
```

Optional environment variables:
- `SEC_USER_AGENT` (recommended by SEC for API etiquette)
- `FRED_API_KEY` (for stable FRED API access)

### Session Logging
Every analysis is appended to `data/session_log.csv` for an audit trail.

---

## Project Structure

```
stock-analysis-dashboard/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         ← yfinance + caching
│   ├── metrics_calculator.py   ← All fundamental metrics (6 questions)
│   ├── threshold_manager.py    ← Smart, sector-aware thresholds
│   ├── visualizations.py       ← Plotly chart factory
│   ├── panel_generator.py      ← 5 analysis panels + ML panel
│   ├── report_generator.py     ← HTML CEO Report
│   ├── utils.py                ← Shared helpers
│   └── first_principles/       ← First-principles engine (SEC, FRED, yfinance)
├── tests/                      ← Test suite (mirrors src/ layout)
├── scripts/
│   ├── run_all_reports.py      ← Generate all 6 report variants
│   └── run_msft_reports.py     ← Generate MSFT reports (dev/demo)
├── notebooks/
│   └── stock_analysis.ipynb    ← Interactive analysis notebook
├── config/
│   └── default_thresholds.json ← Editable sector-aware thresholds
├── docs/
│   └── architecture.md         ← System architecture diagram
├── data/                       ← Runtime-generated (git-ignored)
│   ├── cache/
│   └── session_log.csv
├── outputs/                    ← HTML reports go here (git-ignored)
├── .env.example                ← Copy to .env and fill in secrets
├── CHANGELOG.md
├── LICENSE
├── Makefile                    ← make install / make test / make run
├── pyproject.toml              ← Build config + tool settings
├── requirements.txt            ← Production dependencies
├── requirements-dev.txt        ← Dev/notebook/test dependencies
└── README.md
```

---

## How to Add a New Metric

1. Add the raw data fetch (if needed) in `src/data_fetcher.py` → `StockData._load_all()`
2. Add a field to the appropriate dataclass in `src/metrics_calculator.py` (e.g., `Q3Profitability`)
3. Populate it in the corresponding `_q3_profitability()` method
4. Add a threshold entry in `config/default_thresholds.json`
5. Add a chart in `src/visualizations.py` → `build_all_charts()`
6. Reference it in the relevant panel in `src/panel_generator.py`
7. Add a row in the CEO Report section in `src/report_generator.py` → `_build_6q_sections()`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Charts not showing in notebook | Change `pio.renderers.default = "browser"` in Cell 2 |
| `ModuleNotFoundError: src` | Make sure you run the notebook from `notebooks/` or the project root |
| Data looks stale | Set `FORCE_REFRESH = True` in Cell 1 |
| `N/A` for many metrics | Some tickers (OTC, ETFs, new companies) have limited yfinance data — check the ticker spelling |
| scikit-learn missing | `pip install scikit-learn` or set `ENABLE_ML_PREDICTIONS = False` |

---

## Disclaimer

This tool is for **informational and educational purposes only**. It does not constitute financial advice. All data is sourced from public filings via yfinance. Always verify critical figures independently before making investment decisions.
