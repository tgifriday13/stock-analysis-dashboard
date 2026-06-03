# Architecture Overview

## High-Level Flow

```
User Input (ticker, mode, thresholds)
        │
        ▼
┌─────────────────┐     ┌──────────────────────────────┐
│  StockData      │     │  FirstPrinciplesEngine       │
│  data_fetcher   │     │  src/first_principles/       │
│  (yfinance +    │     │  (SEC EDGAR, FRED, yfinance) │
│   disk cache)   │     │  12 past/future questions    │
└────────┬────────┘     └─────────────┬────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐         ┌─────────────────┐
│ MetricsCalc     │         │ FP Report JSON  │
│ 6-question      │         │ + HTML audit    │
│ dataclasses     │         │   card          │
└────────┬────────┘         └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│ ThresholdMgr    │    │ Visualizations   │
│ sector-aware    │───▶│ Plotly charts    │
│ thresholds      │    └──────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PanelGenerator  │
│ 5 panels +      │
│ ML predictions  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ReportGenerator │
│ HTML CEO Report │
│ → outputs/      │
└─────────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/data_fetcher.py` | Fetch raw data from yfinance; 24-hour disk cache; staleness labels |
| `src/metrics_calculator.py` | Compute all 6 fundamental question metrics; `new` vs `existing` modes |
| `src/threshold_manager.py` | Sector-aware pass/fail thresholds; user override profiles |
| `src/visualizations.py` | Plotly chart factory keyed by question |
| `src/panel_generator.py` | Bullish/Hold/Bearish/Dividends/Portfolio panels; ML predictions; final recommendation |
| `src/report_generator.py` | Render full HTML CEO Report; `generate_master_report` combines all modes/views |
| `src/utils.py` | Shared helpers (date parsing, formatting) |
| `src/first_principles/` | First-principles engine — see below |

## First-Principles Sub-Package (`src/first_principles/`)

| Module | Responsibility |
|---|---|
| `engine.py` | Orchestrates all 12 P/F questions; returns `FPReport` |
| `questions.py` | Per-question logic and falsification tests |
| `sources.py` | SEC EDGAR + FRED + yfinance data fetching |
| `transforms.py` | Point-in-time metric transforms; French factor ingestion |
| `schemas.py` | Pydantic dataclasses for the report |
| `config.py` | Question definitions and threshold bands |
| `cli.py` | Command-line interface entry point |

## Directory Layout

```
stock-analysis-dashboard/
├── src/                        ← All importable library code
│   └── first_principles/       ← First-principles sub-package
├── tests/                      ← Mirrors src/ layout
│   └── first_principles/
├── scripts/                    ← Runnable entry-point scripts
├── notebooks/                  ← Jupyter exploration notebooks
├── config/                     ← Editable JSON threshold defaults
├── docs/                       ← This folder
├── data/                       ← Runtime-generated (git-ignored)
│   ├── cache/
│   └── session_log.csv
└── outputs/                    ← HTML reports (git-ignored by default)
```

## Data Flow: Caching

All external API calls go through `StockData`, which stores timestamped JSON/CSV
files in `data/cache/`. Cache is considered fresh for 24 hours; set
`force_refresh=True` to bypass.

The first-principles engine uses its own cache under `data/first_principles/cache/`.
