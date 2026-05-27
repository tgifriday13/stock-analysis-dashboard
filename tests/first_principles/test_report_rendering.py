from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.first_principles.config import get_question_specs
from src.first_principles.questions import FirstPrinciplesQuestionEngine
from src.first_principles.schemas import (
    ChartPayload,
    ChartSeries,
    EngineMeta,
    FirstPrinciplesReport,
    PanelCoverage,
    QuestionResult,
    SignalSynthesis,
)
from src.metrics_calculator import AllMetrics
from src.report_generator import _build_html


def _macro_bundle() -> dict:
    idx = pd.date_range("2018-01-01", periods=100, freq="ME")
    return {
        "cpi": pd.Series(np.linspace(250, 320, len(idx)), index=idx),
        "recession": pd.Series(np.zeros(len(idx)), index=idx),
        "baa_spread": pd.Series(np.linspace(1.6, 2.4, len(idx)), index=idx),
        "risk_free_10y": pd.Series(np.linspace(1.5, 4.0, len(idx)), index=idx),
        "ff5_rmw": pd.Series(np.linspace(0.001, 0.004, len(idx)), index=idx),
        "ff5_cma": pd.Series(np.linspace(0.001, 0.003, len(idx)), index=idx),
    }


def _base_panel() -> pd.DataFrame:
    years = pd.Index([2019, 2020, 2021, 2022, 2023, 2024], name="fiscal_year")
    return pd.DataFrame(
        {
            "revenue": [100, 108, 118, 130, 142, 158],
            "net_income": [10, 11, 13, 15, 18, 22],
            "operating_income": [14, 15, 17, 20, 23, 28],
            "operating_cash_flow": [15, 16, 18, 22, 25, 31],
            "capex": [3, 3, 4, 4, 5, 6],
            "diluted_shares": [10, 10, 10, 10, 10, 10],
            "total_assets": [120, 128, 138, 152, 168, 185],
            "total_debt": [35, 34, 33, 32, 31, 30],
            "cash_and_equivalents": [8, 9, 10, 11, 12, 13],
            "total_equity": [55, 60, 66, 74, 82, 92],
            "current_assets": [24, 25, 27, 29, 31, 34],
            "current_liabilities": [18, 18, 19, 20, 20, 21],
            "interest_expense": [2.0, 1.9, 1.8, 1.8, 1.7, 1.6],
            "income_tax_expense": [2.1, 2.2, 2.4, 2.8, 3.1, 3.5],
            "dividends_paid": [1.2, 1.2, 1.3, 1.4, 1.5, 1.6],
            "repurchases": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        },
        index=years,
    )


def _metrics() -> AllMetrics:
    m = AllMetrics(ticker="MSFT", company_name="Microsoft Corp.", mode="new")
    m.q1.market_cap = 2_900_000_000_000.0
    m.q1.sector = "Technology"
    m.q1.cap_category = "Mega Cap"
    m.q6.current_price = 410.0
    m.staleness_label = "Data as of test run"
    m.overall_score = 7.1
    return m


def _question(qid: str, chart: bool = True) -> QuestionResult:
    metrics_by_qid = {
        "P1": {
            "real_owner_earnings_cagr_5y": 0.07,
            "real_fcf_per_share_cagr_5y": 0.06,
            "real_revenue_per_share_cagr_5y": 0.05,
            "cumulative_real_owner_earnings_change": 0.30,
        },
        "P2": {"median_incremental_roic_5y": 0.13, "incremental_roic_years_below_watch": 1.0},
        "F1": {"forward_incremental_roic_spread": 0.05, "base_incremental_roic": 0.14},
        "F2": {"growth_assumptions": {"bear": -0.01, "base": 0.04, "bull": 0.09}, "bear_drawdown": -0.20},
    }

    payload = None
    if chart:
        payload = ChartPayload(
            chart_type="line",
            title=f"Chart {qid}",
            x_label="Year",
            y_label="Value",
            unit="ratio",
            series=[ChartSeries(name="Series", x=["2023", "2024"], y=[0.1, 0.2])],
        )
    return QuestionResult(
        question_id=qid,
        question=f"Question {qid}",
        metric_family=["m"],
        signal="Strong",
        falsified=False,
        falsification_reason=None,
        decision_reason="ok",
        thresholds={},
        metrics=metrics_by_qid.get(qid, {"x": 1.0}),
        percentile_overlay={},
        coverage=PanelCoverage(required_fields=["a"], available_fields=["a"], missing_fields=[], coverage_ratio=1.0),
        takeaway=f"Takeaway {qid}",
        chart_payload=payload,
    )


def _fp_report() -> FirstPrinciplesReport:
    questions = [_question("P1"), _question("P2"), _question("F1"), _question("F2")]
    return FirstPrinciplesReport(
        meta=EngineMeta(
            ticker="MSFT",
            cik="0000789019",
            run_timestamp=datetime.now(timezone.utc),
            as_of_date=date(2026, 5, 27),
            accepted_cutoff_rule="acceptance_date + 1 business day",
            sector_template="non_financial",
            data_freshness={},
        ),
        questions=questions,
        synthesis=SignalSynthesis(
            decision="Go",
            hard_fail=False,
            reasons=[],
            strong_count=4,
            watch_count=0,
            weak_count=0,
            insufficient_count=0,
        ),
    )


def test_question_engine_emits_chart_payload_contract():
    specs = get_question_specs("non_financial")
    engine = FirstPrinciplesQuestionEngine(specs)
    annual = _base_panel()

    results = engine.evaluate(annual, annual, {}, _macro_bundle(), latest_price=100.0)
    assert len(results) == 12

    for result in results:
        assert result.takeaway
        assert result.chart_payload is not None
        assert result.chart_payload.series
        for s in result.chart_payload.series:
            assert len(s.x) == len(s.y)


def test_report_view_renders_shared_frame_for_past_and_future():
    fp = _fp_report()
    m = _metrics()

    past_html = _build_html(
        metrics=m,
        panels={},
        recommendation="Hold",
        recommendation_explanation="baseline",
        charts={},
        report_view="past",
        first_principles_report=fp,
    )
    future_html = _build_html(
        metrics=m,
        panels={},
        recommendation="Hold",
        recommendation_explanation="baseline",
        charts={},
        report_view="future",
        first_principles_report=fp,
    )

    for html in [past_html, future_html]:
        assert "Stock Fundamentals CEO Dashboard" in html
        assert "informational purposes only" in html
        assert "Metric Interpretation Guide" in html
        assert "Technical metrics (expand)" in html
        assert "Main takeaway:" in html
        # Two cards per view, each with guide+technical dropdowns.
        assert html.count("<details") >= 4

    assert "P1: Question P1" in past_html
    assert "F1: Question F1" not in past_html
    assert "F1: Question F1" in future_html
    assert "P1: Question P1" not in future_html


def test_missing_chart_data_gracefully_degrades():
    fp = _fp_report()
    fp.questions = [_question("P1", chart=False)]

    html = _build_html(
        metrics=_metrics(),
        panels={},
        recommendation="Hold",
        recommendation_explanation="baseline",
        charts={},
        report_view="past",
        first_principles_report=fp,
    )
    assert "Chart data unavailable." in html


def test_present_mode_keeps_guide_and_adds_technical_details():
    html = _build_html(
        metrics=_metrics(),
        panels={},
        recommendation="Hold",
        recommendation_explanation="baseline",
        charts={},
        report_view="present",
        first_principles_report=None,
    )
    assert "Metric Interpretation Guide" in html
    assert "Main takeaway:" in html
    assert "Technical metrics (expand)" in html
    # Six present sections, each with guide+technical dropdowns.
    assert html.count("<details") >= 12
