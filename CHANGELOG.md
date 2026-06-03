# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Professional repo structure: `scripts/`, `docs/`, `pyproject.toml`, `Makefile`, `LICENSE`, `.env.example`, `CHANGELOG.md`
- `requirements-dev.txt` split from `requirements.txt`

---

## [0.2.0] — 2024-Q4

### Added
- Master CEO report combining all ownership modes (`generate_master_report`)
- First-principles engine (`src/first_principles/`) — 12 P/F questions, SEC EDGAR point-in-time data
- Kenneth French factor ingestion (RMW, CMA reference bands)
- `REPORT_TAB` switcher: `present` / `past` / `future`
- Session audit trail saved to `data/session_log.csv`

---

## [0.1.0] — 2024-Q3

### Added
- Initial release: 6-question fundamental analysis dashboard
- yfinance data fetching with 24-hour disk cache
- Sector-aware thresholds with user override profiles
- Plotly chart factory and 5 analysis panels
- HTML CEO Report saved to `outputs/`
- ML fair-value + outperformance predictions (optional, scikit-learn)
- Multi-stock comparison table
- `new` and `existing` ownership modes
