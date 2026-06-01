"""Top-level orchestration for the first-principles past/future signal engine."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

# Load .env so FRED_API_KEY (and any other secrets) are available via os.getenv
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

from ..utils import outputs_dir
from .config import (
    DEFAULT_ACCEPTANCE_BUFFER_DAYS,
    FRED_SERIES,
    get_question_specs,
    is_financial_sector,
)
from .questions import FirstPrinciplesQuestionEngine
from .schemas import EngineMeta, FirstPrinciplesReport, SignalSynthesis
from .sources import FrenchDataSource, FredSource, PriceSource, SecEdgarSource
from .transforms import (
    build_annual_and_quarterly_panels,
    build_pit_view,
    build_research_view,
    build_ttm_flow_panel,
    freshness_snapshot,
    normalize_companyfacts,
)

logger = logging.getLogger("dashboard.first_principles.engine")


class FirstPrinciplesEngine:
    """End-to-end implementation of the first-principles signal framework."""

    def __init__(
        self,
        force_refresh: bool = False,
        sec_user_agent: Optional[str] = None,
        fred_api_key: Optional[str] = None,
    ):
        self.sec = SecEdgarSource(force_refresh=force_refresh, user_agent=sec_user_agent)
        self.fred = FredSource(api_key=fred_api_key, force_refresh=force_refresh)
        self.french = FrenchDataSource(force_refresh=force_refresh)
        self.price = PriceSource(force_refresh=force_refresh)
        self.force_refresh = force_refresh

    def run(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
        sector_override: Optional[str] = None,
    ) -> FirstPrinciplesReport:
        """Run the full pipeline for one ticker."""
        as_of = as_of_date or date.today()
        ticker_u = ticker.upper().strip()

        mapping = self.sec.resolve_ticker(ticker_u)
        cik = mapping["cik"]

        submissions = self.sec.fetch_submissions(cik)
        filing_meta = self.sec.filing_metadata_map(submissions)
        companyfacts = self.sec.fetch_companyfacts(cik)

        facts_df = normalize_companyfacts(cik=cik, ticker=ticker_u, companyfacts_json=companyfacts, filing_meta=filing_meta)

        research_view = build_research_view(facts_df)
        pit_view = build_pit_view(facts_df, as_of_date=as_of)

        annual_research, quarterly_research = build_annual_and_quarterly_panels(research_view)
        annual_pit, quarterly_pit = build_annual_and_quarterly_panels(pit_view)
        ttm = build_ttm_flow_panel(quarterly_pit)

        macro = self._fetch_macro_bundle(as_of=as_of, annual_panel=annual_pit)
        latest_price = self.price.latest_price(ticker_u)

        sector = sector_override or self._detect_sector_yfinance(ticker_u) or "Unknown"
        template = "financial" if is_financial_sector(sector) else "non_financial"

        specs = get_question_specs(template=template)
        question_engine = FirstPrinciplesQuestionEngine(specs=specs)
        question_results = question_engine.evaluate(
            annual_pit=annual_pit,
            annual_research=annual_research,
            quarterly_pit=quarterly_pit,
            macro=macro,
            latest_price=latest_price,
        )

        synthesis = self._synthesize(question_results)
        freshness = freshness_snapshot(facts_df)
        freshness["macro_latest"] = {
            k: (v.dropna().index.max().date().isoformat() if not v.empty else None)
            for k, v in macro.items()
        }
        freshness["latest_price"] = latest_price
        freshness["pit_num_rows"] = int(len(pit_view))
        freshness["research_num_rows"] = int(len(research_view))
        freshness["ttm_fields"] = sorted(ttm.index.tolist()) if not ttm.empty else []

        meta = EngineMeta(
            ticker=ticker_u,
            cik=str(cik).zfill(10),
            run_timestamp=datetime.utcnow(),
            as_of_date=as_of,
            accepted_cutoff_rule=f"acceptance_date + {DEFAULT_ACCEPTANCE_BUFFER_DAYS} business day",
            sector_template=template,
            data_freshness=freshness,
        )

        return FirstPrinciplesReport(meta=meta, questions=question_results, synthesis=synthesis)

    def save_report(
        self,
        report: FirstPrinciplesReport,
        basename: Optional[str] = None,
        time_view: str = "all",
    ) -> Dict[str, Path]:
        """Save legacy first-principles artifacts (JSON + compact HTML) under outputs/."""
        json_path = self.save_json_report(report=report, basename=basename, time_view=time_view)
        out_dir = json_path.parent
        base = json_path.stem

        view = self._normalize_time_view(time_view)
        html_path = out_dir / f"{base}.html"
        html_path.write_text(self._to_html(report, time_view=view), encoding="utf-8")

        return {"json": json_path, "html": html_path}

    def save_json_report(
        self,
        report: FirstPrinciplesReport,
        basename: Optional[str] = None,
        time_view: str = "all",
    ) -> Path:
        """Save JSON audit artifact under outputs/first_principles and return its path."""
        out_dir = outputs_dir() / "first_principles"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = report.meta.run_timestamp.strftime("%Y%m%d_%H%M%S")
        view = self._normalize_time_view(time_view)
        suffix = f"_{view}" if view != "all" else ""
        base = basename or f"{report.meta.ticker}_{stamp}{suffix}"

        json_path = out_dir / f"{base}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        return json_path

    @staticmethod
    def _normalize_time_view(time_view: str) -> str:
        v = (time_view or "all").strip().lower()
        if v not in {"all", "past", "future"}:
            return "all"
        return v

    def _filter_questions_by_time_view(self, questions, time_view: str):
        view = self._normalize_time_view(time_view)
        if view == "past":
            return [q for q in questions if q.question_id.upper().startswith("P")]
        if view == "future":
            return [q for q in questions if q.question_id.upper().startswith("F")]
        return list(questions)

    def _fetch_macro_bundle(self, as_of: date, annual_panel: pd.DataFrame) -> Dict[str, pd.Series]:
        if annual_panel.empty:
            start = date(as_of.year - 10, 1, 1)
        else:
            start = date(max(annual_panel.index.min() - 1, 1990), 1, 1)

        macro: Dict[str, pd.Series] = {}
        for key, series_id in FRED_SERIES.items():
            try:
                macro[key] = self.fred.fetch_series(series_id, start_date=start, end_date=as_of)
            except Exception as exc:
                logger.warning("Failed to fetch FRED series %s (%s): %s", key, series_id, exc)
                macro[key] = pd.Series(dtype=float)

        # Factor/regime reference bands (profitability/investment) from French library.
        try:
            ff5 = self.french.fetch_ff5_monthly()
            macro["ff5_rmw"] = ff5.get("RMW", pd.Series(dtype=float))
            macro["ff5_cma"] = ff5.get("CMA", pd.Series(dtype=float))
        except Exception as exc:
            logger.warning("Failed to fetch Kenneth French factor data: %s", exc)
            macro["ff5_rmw"] = pd.Series(dtype=float)
            macro["ff5_cma"] = pd.Series(dtype=float)
        return macro

    def _detect_sector_yfinance(self, ticker: str) -> Optional[str]:
        try:
            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector") or info.get("industry")
            if sector:
                return str(sector)
        except Exception:
            return None
        return None

    def _synthesize(self, questions) -> SignalSynthesis:
        strong = sum(1 for q in questions if q.signal == "Strong")
        watch = sum(1 for q in questions if q.signal == "Watch")
        weak = sum(1 for q in questions if q.signal == "Weak")
        insuff = sum(1 for q in questions if q.signal == "Insufficient")

        hard_fail = any(q.falsified for q in questions)
        reasons: List[str] = []
        for q in questions:
            if q.falsified:
                reasons.append(f"{q.question_id}: {q.falsification_reason}")
            elif q.signal == "Weak":
                reasons.append(f"{q.question_id}: Weak")

        if hard_fail or weak >= 4:
            decision = "Avoid/Trim"
        elif insuff >= 5:
            decision = "Monitor"
        elif strong >= 6 and weak == 0:
            decision = "Go"
        else:
            decision = "Monitor"

        return SignalSynthesis(
            decision=decision,
            hard_fail=hard_fail,
            reasons=reasons,
            strong_count=strong,
            watch_count=watch,
            weak_count=weak,
            insufficient_count=insuff,
        )

    def _to_html(self, report: FirstPrinciplesReport, time_view: str = "all") -> str:
        """Compact HTML artifact with audit-card style layout."""
        view = self._normalize_time_view(time_view)
        questions = self._filter_questions_by_time_view(report.questions, view)
        synthesis = self._synthesize(questions) if questions else report.synthesis

        view_title = {
            "all": "Past + Future",
            "past": "Past",
            "future": "Future",
        }[view]

        rows = []
        for q in questions:
            metrics_html = "".join(
                f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in q.metrics.items()
            )
            rows.append(
                f"""
                <section class=\"card\">
                    <h3>{q.question_id}: {q.question}</h3>
                    <p><strong>Signal:</strong> {q.signal} | <strong>Coverage:</strong> {q.coverage.coverage_ratio:.0%}</p>
                    <p><strong>Falsified:</strong> {q.falsified} {('- ' + q.falsification_reason) if q.falsification_reason else ''}</p>
                    <p>{q.decision_reason}</p>
                    <table>
                      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                      <tbody>{metrics_html}</tbody>
                    </table>
                </section>
                """
            )

        reasons = "".join(f"<li>{r}</li>" for r in synthesis.reasons) or "<li>None</li>"
        return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>First Principles Report - {view_title} - {report.meta.ticker}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #1b1f24; }}
    .top {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 10px; padding: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; margin-top: 16px; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 10px; padding: 12px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; border-top: 1px solid #d8dee4; padding: 6px 4px; font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"top\">
    <h1>First-Principles Signal Engine — {view_title} View</h1>
    <p><strong>Ticker:</strong> {report.meta.ticker} | <strong>CIK:</strong> {report.meta.cik}</p>
    <p><strong>As of:</strong> {report.meta.as_of_date} | <strong>Template:</strong> {report.meta.sector_template}</p>
    <p><strong>Decision:</strong> {synthesis.decision} | <strong>Hard fail:</strong> {synthesis.hard_fail}</p>
    <p><strong>Questions shown:</strong> {len(questions)}</p>
    <ul>{reasons}</ul>
  </div>
  <div class=\"grid\">{''.join(rows)}</div>
</body>
</html>
"""
