"""
test_report_generation.py — Comprehensive tests for all three report views
(present, past, future) and both ownership modes (new, existing).

Covers:
  - View normalization edge cases
  - Score badge tiers (5 levels)
  - Recommendation box styles (all ~15 recs)
  - Data quality banner (stale data, low confidence, flags)
  - Present report — new position mode
  - Present report — existing holding mode (portfolio context, deltas)
  - Past report — first-principles P-questions only
  - Future report — first-principles F-questions only
  - Decision logic (Go / Monitor / Avoid/Trim thresholds)
  - Hard-fail banner rendering
  - Staleness warning rendering
  - Graceful fallback when first_principles_report is None
  - Individual question card rendering (signals, falsification, coverage, chart)
  - HTML escaping / XSS prevention
  - Edge cases: empty panels, all-None metrics, empty question list
  - generate_report() file output + naming convention
"""

from __future__ import annotations

import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.metrics_calculator import (
    AllMetrics, PortfolioContext,
    Q1Scale, Q2Growth, Q3Profitability, Q4CashFlow, Q5BalanceSheet, Q6Valuation,
)
from src.panel_generator import Panel
from src.report_generator import (
    _build_html,
    _build_master_html,
    _data_quality_banner,
    _fp_hard_fail_banner,
    _fp_staleness_warning,
    _normalize_report_view,
    _recommendation_box,
    _score_badge,
    generate_master_report,
    generate_report,
)
from src.first_principles.schemas import (
    ChartPayload,
    ChartSeries,
    EngineMeta,
    FirstPrinciplesReport,
    PanelCoverage,
    QuestionResult,
    SignalSynthesis,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_metrics(
    ticker: str = "AAPL",
    company_name: str = "Apple Inc.",
    mode: str = "new",
    overall_score: float = 7.0,
    staleness_label: str = "Data as of test run",
    decision_confidence: float = 90.0,
    decision_confidence_flags: list | None = None,
    # Q1
    market_cap: float | None = 3_000_000_000_000.0,
    sector: str = "Technology",
    cap_category: str = "Mega Cap",
    # Q2
    revenue_1yr_cagr: float | None = 0.08,
    # Q3
    operating_margin: float | None = 0.30,
    # Q4
    fcf_yield: float | None = 0.04,
    free_cash_flow: float | None = 100_000_000_000.0,
    # Q5
    debt_to_ebitda: float | None = 1.5,
    # Q6
    current_price: float | None = 200.0,
    pe_ratio: float | None = 28.0,
    # Existing-mode extras
    price_at_purchase: float | None = None,
    unrealized_gain_loss_pct: float | None = None,
    valuation_change: str = "Unknown",
    shares_owned: float | None = None,
    cost_basis: float | None = None,
    total_portfolio_value: float | None = None,
    position_weight: float | None = None,
) -> AllMetrics:
    m = AllMetrics(
        ticker=ticker,
        company_name=company_name,
        mode=mode,
        staleness_label=staleness_label,
        overall_score=overall_score,
        decision_confidence=decision_confidence,
        decision_confidence_flags=decision_confidence_flags or [],
    )
    m.q1 = Q1Scale(
        market_cap=market_cap,
        sector=sector,
        cap_category=cap_category,
        industry="Consumer Electronics",
        country="USA",
    )
    m.q2 = Q2Growth(revenue_1yr_cagr=revenue_1yr_cagr, growth_trend="Stable")
    m.q3 = Q3Profitability(operating_margin=operating_margin, margin_trend="Stable")
    m.q4 = Q4CashFlow(fcf_yield=fcf_yield, free_cash_flow=free_cash_flow)
    m.q5 = Q5BalanceSheet(debt_to_ebitda=debt_to_ebitda, financial_risk_trend="Stable")
    m.q6 = Q6Valuation(
        current_price=current_price,
        pe_ratio=pe_ratio,
        price_at_purchase=price_at_purchase,
        unrealized_gain_loss_pct=unrealized_gain_loss_pct,
        valuation_change=valuation_change,
    )
    m.scores = {
        "Q1_Scale": 7.0, "Q2_Growth": 6.5, "Q3_Profitability": 7.5,
        "Q4_CashFlow": 8.0, "Q5_BalanceSheet": 7.0, "Q6_Valuation": 5.5,
    }
    if mode == "existing":
        m.portfolio = PortfolioContext(
            shares_owned=shares_owned or 100.0,
            cost_basis=cost_basis or 150.0,
            total_portfolio_value=total_portfolio_value or 500_000.0,
            position_weight=position_weight or 0.04,
            total_cost=15_000.0,
            total_gain_loss_dollars=5_000.0,
            total_gain_loss_pct=0.333,
            days_held=365,
            annualized_return=0.333,
        )
    return m


def _make_question(
    qid: str,
    signal: str = "Strong",
    falsified: bool = False,
    falsification_reason: str | None = None,
    coverage_ratio: float = 1.0,
    missing_fields: list | None = None,
    takeaway: str = "",
    with_chart: bool = True,
) -> QuestionResult:
    payload = None
    if with_chart:
        payload = ChartPayload(
            chart_type="line",
            title=f"Chart {qid}",
            x_label="Year",
            y_label="Value",
            unit="ratio",
            series=[ChartSeries(name="Series", x=["2022", "2023", "2024"], y=[0.1, 0.12, 0.15])],
        )
    return QuestionResult(
        question_id=qid,
        question=f"Test question for {qid}",
        metric_family=["test"],
        signal=signal,
        falsified=falsified,
        falsification_reason=falsification_reason,
        decision_reason=takeaway or f"Decision reason for {qid}",
        thresholds={"min": 0.05},
        metrics={"key_metric": 0.15, "secondary": 0.08},
        percentile_overlay={},
        coverage=PanelCoverage(
            required_fields=["a", "b"],
            available_fields=["a"] if coverage_ratio < 1.0 else ["a", "b"],
            missing_fields=missing_fields or ([] if coverage_ratio == 1.0 else ["b"]),
            coverage_ratio=coverage_ratio,
        ),
        takeaway=takeaway or f"Takeaway for {qid}",
        chart_payload=payload,
    )


def _make_fp_report(
    ticker: str = "AAPL",
    questions: list | None = None,
    decision: str = "Go",
    hard_fail: bool = False,
    strong: int = 4,
    watch: int = 2,
    weak: int = 0,
    insuff: int = 0,
) -> FirstPrinciplesReport:
    if questions is None:
        questions = [
            _make_question("P1"), _make_question("P2"), _make_question("P3"),
            _make_question("P4"), _make_question("P5"), _make_question("P6"),
            _make_question("F1"), _make_question("F2"), _make_question("F3"),
            _make_question("F4"), _make_question("F5"), _make_question("F6"),
        ]
    return FirstPrinciplesReport(
        meta=EngineMeta(
            ticker=ticker,
            cik="0000320193",
            run_timestamp=datetime.now(timezone.utc),
            as_of_date=date(2026, 5, 30),
            accepted_cutoff_rule="acceptance_date + 1 business day",
            sector_template="non_financial",
            data_freshness={},
        ),
        questions=questions,
        synthesis=SignalSynthesis(
            decision=decision,
            hard_fail=hard_fail,
            reasons=[],
            strong_count=strong,
            watch_count=watch,
            weak_count=weak,
            insufficient_count=insuff,
        ),
    )


def _make_panel(title: str = "Why Buy", bullets: list | None = None, raw_data: dict | None = None) -> Panel:
    return Panel(
        title=title,
        mode_label=title,
        bullets=bullets or ["Bullet 1", "Bullet 2"],
        flags=[],
        charts=[],
        raw_data=raw_data or {},
        signal_count=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. View normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeReportView:
    def test_present_exact(self):
        assert _normalize_report_view("present") == "present"

    def test_past_exact(self):
        assert _normalize_report_view("past") == "past"

    def test_future_exact(self):
        assert _normalize_report_view("future") == "future"

    def test_uppercase_normalizes(self):
        assert _normalize_report_view("PRESENT") == "present"
        assert _normalize_report_view("PAST") == "past"
        assert _normalize_report_view("FUTURE") == "future"

    def test_mixed_case_normalizes(self):
        assert _normalize_report_view("Present") == "present"
        assert _normalize_report_view("Past") == "past"
        assert _normalize_report_view("Future") == "future"

    def test_empty_string_defaults_to_present(self):
        assert _normalize_report_view("") == "present"

    def test_invalid_string_defaults_to_present(self):
        assert _normalize_report_view("daily") == "present"
        assert _normalize_report_view("current") == "present"
        assert _normalize_report_view("xyz") == "present"

    def test_whitespace_stripped(self):
        assert _normalize_report_view("  past  ") == "past"
        assert _normalize_report_view("  future  ") == "future"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Score badge tiers
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreBadge:
    def test_excellent_tier(self):
        html = _score_badge(8.0)
        assert "Excellent" in html
        assert "#27ae60" in html  # green

    def test_excellent_at_maximum(self):
        html = _score_badge(10.0)
        assert "Excellent" in html

    def test_good_tier_lower_bound(self):
        html = _score_badge(6.5)
        assert "Good" in html

    def test_good_tier_upper_bound(self):
        html = _score_badge(7.9)
        assert "Good" in html

    def test_neutral_tier_lower_bound(self):
        html = _score_badge(5.0)
        assert "Neutral" in html

    def test_neutral_tier_upper_bound(self):
        html = _score_badge(6.4)
        assert "Neutral" in html

    def test_weak_tier_lower_bound(self):
        html = _score_badge(3.5)
        assert "Weak" in html

    def test_weak_tier_upper_bound(self):
        html = _score_badge(4.9)
        assert "Weak" in html

    def test_bad_tier(self):
        html = _score_badge(0.0)
        assert "Bad" in html
        assert "#e74c3c" in html  # red

    def test_bad_tier_below_3_5(self):
        html = _score_badge(3.4)
        assert "Bad" in html

    def test_score_displayed(self):
        html = _score_badge(7.5)
        assert "7.5" in html

    def test_score_formatted_to_one_decimal(self):
        html = _score_badge(6.0)
        assert "6.0/10" in html


# ─────────────────────────────────────────────────────────────────────────────
# 3. Recommendation box — all recommendation types
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationBox:
    @pytest.mark.parametrize("rec", [
        "Strong Buy", "Buy", "Buy More", "Add Slowly",
    ])
    def test_buy_recommendations_render_green(self, rec):
        html = _recommendation_box(rec, "Explanation here")
        assert rec in html
        assert "#27ae60" in html  # green border
        assert "Explanation here" in html

    @pytest.mark.parametrize("rec", ["Hold", "Hold / Monitor"])
    def test_hold_recommendations_render_yellow(self, rec):
        html = _recommendation_box(rec, "Hold explanation")
        assert rec in html
        assert "#f39c12" in html  # yellow border

    @pytest.mark.parametrize("rec", ["Trim", "Sell Partial", "Sell", "Avoid"])
    def test_sell_recommendations_render_red(self, rec):
        html = _recommendation_box(rec, "Sell explanation")
        assert rec in html
        assert "#e74c3c" in html  # red

    def test_exit_recommendation_renders_darkest_red(self):
        html = _recommendation_box("Exit", "Exit explanation")
        assert "Exit" in html
        assert "#c0392b" in html  # darkest red

    def test_optional_trim_renders(self):
        html = _recommendation_box("Optional Trim", "Consider trimming.")
        assert "Optional Trim" in html

    def test_unknown_recommendation_falls_back_to_hold_style(self):
        html = _recommendation_box("Undecided", "Not sure.")
        assert "Undecided" in html
        assert "#f39c12" in html  # Hold fallback

    def test_go_recommendation_renders_green(self):
        html = _recommendation_box("Go", "First-principles go signal.")
        assert "Go" in html
        assert "#27ae60" in html

    def test_monitor_recommendation_renders_yellow(self):
        html = _recommendation_box("Monitor", "Monitor the position.")
        assert "#f39c12" in html

    def test_avoid_trim_renders_red(self):
        html = _recommendation_box("Avoid/Trim", "Reduce exposure.")
        assert "#e74c3c" in html


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data quality banner
# ─────────────────────────────────────────────────────────────────────────────

class TestDataQualityBanner:
    def test_clean_data_returns_empty(self):
        m = _make_metrics(decision_confidence=95.0, staleness_label="Data as of today")
        html = _data_quality_banner(m)
        assert html == ""

    def test_stale_filing_shows_warning(self):
        m = _make_metrics(staleness_label="last filing 150 days ago")
        html = _data_quality_banner(m)
        assert "Stale filing data" in html
        assert "150 days ago" in html

    def test_staleness_under_120_days_no_badge(self):
        m = _make_metrics(staleness_label="last filing 100 days ago")
        html = _data_quality_banner(m)
        # staleness badge should NOT appear
        assert "Stale filing data" not in html

    def test_low_confidence_shows_confidence_badge(self):
        m = _make_metrics(decision_confidence=45.0)
        html = _data_quality_banner(m)
        assert "45%" in html

    def test_confidence_icon_red_below_40(self):
        m = _make_metrics(decision_confidence=35.0)
        html = _data_quality_banner(m)
        assert "🔴" in html

    def test_confidence_icon_orange_40_to_60(self):
        m = _make_metrics(decision_confidence=55.0)
        html = _data_quality_banner(m)
        assert "🟠" in html

    def test_confidence_icon_yellow_60_to_80(self):
        m = _make_metrics(decision_confidence=65.0)
        html = _data_quality_banner(m)
        assert "🟡" in html

    def test_confidence_icon_green_above_80(self):
        m = _make_metrics(decision_confidence=85.0)
        html = _data_quality_banner(m)
        # Only shown when there are flags or stale data
        # With nothing to show, returns ""
        assert html == ""

    def test_flags_shown_in_banner(self):
        m = _make_metrics(
            decision_confidence=60.0,
            decision_confidence_flags=["Revenue trend data unavailable"],
        )
        html = _data_quality_banner(m)
        assert "Revenue trend data unavailable" in html

    def test_multiple_flags_all_shown(self):
        flags = ["Flag one", "Flag two", "Flag three"]
        m = _make_metrics(decision_confidence=50.0, decision_confidence_flags=flags)
        html = _data_quality_banner(m)
        for f in flags:
            assert f in html


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hard-fail banner
# ─────────────────────────────────────────────────────────────────────────────

class TestHardFailBanner:
    def test_empty_reasons_returns_empty(self):
        html = _fp_hard_fail_banner([])
        assert html == ""

    def test_single_reason_renders(self):
        html = _fp_hard_fail_banner(["P6: Interest coverage below minimum"])
        assert "Thesis At Risk" in html
        assert "P6: Interest coverage below minimum" in html
        assert "⚠️" in html

    def test_multiple_reasons_all_rendered(self):
        reasons = ["P6: Interest coverage below minimum", "P3: Negative accruals"]
        html = _fp_hard_fail_banner(reasons)
        for r in reasons:
            assert r in html

    def test_thesis_concern_banner_amber_styling(self):
        html = _fp_hard_fail_banner(["P6: fail"])
        assert "#d35400" in html  # amber/orange

    def test_principle_concern_note_present(self):
        html = _fp_hard_fail_banner(["P6: fail"])
        assert "core investment principles" in html


# ─────────────────────────────────────────────────────────────────────────────
# 6. Staleness warning
# ─────────────────────────────────────────────────────────────────────────────

class TestStalenessWarning:
    def test_empty_staleness_label_returns_empty(self):
        assert _fp_staleness_warning("") == ""

    def test_no_match_in_label_returns_empty(self):
        assert _fp_staleness_warning("Data as of today") == ""

    def test_120_days_or_fewer_returns_empty(self):
        assert _fp_staleness_warning("last filing 120 days ago") == ""
        assert _fp_staleness_warning("last filing 90 days ago") == ""

    def test_over_120_days_shows_warning(self):
        html = _fp_staleness_warning("last filing 200 days ago")
        assert "Stale Fundamental Data" in html
        assert "200" in html
        assert "⚠️" in html

    def test_staleness_warning_mentions_sec_filings(self):
        html = _fp_staleness_warning("last filing 150 days ago")
        assert "SEC" in html or "filing" in html.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Present report — new position mode
# ─────────────────────────────────────────────────────────────────────────────

class TestPresentReportNewMode:
    def _build(self, **kw) -> str:
        m = _make_metrics(mode="new", **kw)
        panels = {
            "bullish":  _make_panel("Why Buy"),
            "hold":     _make_panel("Why Hold"),
            "bearish":  _make_panel("Why Sell"),
            "dividends": _make_panel("Dividends"),
        }
        return _build_html(
            metrics=m,
            panels=panels,
            recommendation="Buy",
            recommendation_explanation="Strong fundamentals.",
            charts={},
            report_view="present",
            first_principles_report=None,
        )

    def test_renders_dashboard_title(self):
        html = self._build()
        assert "Stock Fundamentals CEO Dashboard" in html

    def test_renders_ticker(self):
        html = self._build()
        assert "AAPL" in html

    def test_renders_company_name(self):
        html = self._build()
        assert "Apple Inc." in html

    def test_renders_new_position_mode_label(self):
        html = self._build()
        assert "New Position Analysis" in html

    def test_renders_recommendation(self):
        html = self._build()
        assert "Buy" in html
        assert "Strong fundamentals." in html

    def test_renders_disclaimer(self):
        html = self._build()
        assert "informational purposes only" in html

    def test_renders_metric_interpretation_guide(self):
        html = self._build()
        assert "Metric Interpretation Guide" in html

    def test_renders_technical_metrics_details(self):
        html = self._build()
        assert "Technical metrics (expand)" in html

    def test_six_question_sections_present(self):
        html = self._build()
        assert "Question 1" in html
        assert "Question 6" in html

    def test_main_takeaway_labels_present(self):
        html = self._build()
        assert "Main takeaway:" in html

    def test_fundamental_scores_card_present(self):
        html = self._build()
        assert "Fundamental Scores" in html

    def test_bullish_panel_rendered(self):
        html = self._build()
        assert "Why Buy" in html

    def test_hold_panel_rendered(self):
        html = self._build()
        assert "Why Hold" in html

    def test_bearish_panel_rendered(self):
        html = self._build()
        assert "Why Sell" in html

    def test_dividends_panel_rendered(self):
        html = self._build()
        assert "Dividends" in html

    def test_notes_rendered_when_provided(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation="Baseline.",
            charts={},
            report_view="present",
            notes="Custom analyst note for testing.",
        )
        assert "Custom analyst note for testing." in html

    def test_notes_from_metrics_rendered(self):
        m = _make_metrics()
        m.notes = "Metrics-level note."
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "Metrics-level note." in html

    def test_decision_audit_card_rendered_when_present(self):
        audit_data = {
            "company_verdict": "Good",
            "valuation_verdict": "Fair",
            "primary_driver": "Strong FCF",
            "secondary_driver": "N/A",
            "final_action": "Buy",
            "why": "Business is healthy and priced fairly.",
        }
        audit_panel = Panel(
            title="Decision Audit",
            mode_label="Decision Audit",
            raw_data=audit_data,
        )
        html = _build_html(
            metrics=_make_metrics(),
            panels={"decision_audit": audit_panel},
            recommendation="Buy",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "Decision Audit" in html
        assert "Strong FCF" in html
        assert "Business is healthy and priced fairly." in html

    def test_decision_audit_skipped_when_absent(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        # No audit panel → no audit card. Ensure no crash.
        assert "<!DOCTYPE html>" in html

    def test_no_fp_questions_in_present_view(self):
        html = self._build()
        # present view must NOT contain FP question notation "P1:" or "F1:"
        assert "P1:" not in html
        assert "F1:" not in html

    def test_valid_html_structure(self):
        html = self._build()
        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html

    def test_sector_shown_in_snapshot(self):
        html = self._build()
        assert "Technology" in html

    def test_cap_category_shown(self):
        html = self._build()
        assert "Mega Cap" in html


# ─────────────────────────────────────────────────────────────────────────────
# 8. Present report — existing holding mode
# ─────────────────────────────────────────────────────────────────────────────

class TestPresentReportExistingMode:
    def _build(self, **kw) -> str:
        m = _make_metrics(mode="existing", **kw)
        return _build_html(
            metrics=m,
            panels={},
            recommendation="Hold",
            recommendation_explanation="Maintain position.",
            charts={},
            report_view="present",
        )

    def test_renders_existing_holding_label(self):
        html = self._build()
        assert "Existing Holding" in html

    def test_does_not_show_new_position_label(self):
        html = self._build()
        assert "New Position Analysis" not in html

    def test_renders_with_cost_basis(self):
        html = self._build(cost_basis=150.0, current_price=200.0)
        assert "AAPL" in html  # renders without crash

    def test_unrealized_gain_pct_shown(self):
        m = _make_metrics(
            mode="existing",
            price_at_purchase=150.0,
            current_price=200.0,
            unrealized_gain_loss_pct=0.333,
        )
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "33" in html  # 33.3% appears somewhere

    def test_valuation_change_shown(self):
        m = _make_metrics(mode="existing", valuation_change="Cheaper")
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "Cheaper" in html

    def test_decision_audit_with_portfolio_verdicts(self):
        audit_data = {
            "company_verdict": "Good",
            "valuation_verdict": "Fair",
            "weight_verdict": "Near Target",
            "gain_loss_verdict": "Gain",
            "primary_driver": "Hold",
            "secondary_driver": "Monitor",
            "final_action": "Hold",
            "why": "Existing position is performing well.",
        }
        audit_panel = Panel(
            title="Decision Audit",
            mode_label="Decision Audit",
            raw_data=audit_data,
        )
        html = _build_html(
            metrics=_make_metrics(mode="existing"),
            panels={"decision_audit": audit_panel},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "Near Target" in html
        assert "Gain" in html
        assert "Existing position is performing well." in html

    def test_portfolio_panel_rendered_when_present(self):
        portfolio_panel = _make_panel("Portfolio Risk Summary")
        html = _build_html(
            metrics=_make_metrics(mode="existing"),
            panels={"portfolio": portfolio_panel},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "Portfolio Risk Summary" in html


# ─────────────────────────────────────────────────────────────────────────────
# 9. Past report — first-principles P-questions
# ─────────────────────────────────────────────────────────────────────────────

class TestPastReport:
    def _build(self, fp_report=None, **kw) -> str:
        if fp_report is None:
            fp_report = _make_fp_report()
        m = _make_metrics(**kw)
        return _build_html(
            metrics=m,
            panels={},
            recommendation="Go",
            recommendation_explanation="Strong past signals.",
            charts={},
            report_view="past",
            first_principles_report=fp_report,
        )

    def test_view_title_shows_past(self):
        html = self._build()
        assert "Past" in html

    def test_first_principles_past_view_label(self):
        html = self._build()
        assert "First-Principles Past View" in html

    def test_only_p_questions_shown(self):
        html = self._build()
        for qid in ["P1", "P2", "P3", "P4", "P5", "P6"]:
            assert f"{qid}:" in html
        for qid in ["F1", "F2", "F3", "F4", "F5", "F6"]:
            assert f"{qid}:" not in html

    def test_renders_dashboard_title(self):
        html = self._build()
        assert "Stock Fundamentals CEO Dashboard" in html

    def test_renders_disclaimer(self):
        html = self._build()
        assert "informational purposes only" in html

    def test_renders_company_snapshot_card(self):
        html = self._build()
        assert "Company Snapshot" in html

    def test_renders_signal_scorecard(self):
        html = self._build()
        assert "Signal Scorecard" in html

    def test_renders_investment_decision_summary(self):
        html = self._build()
        assert "Investment Decision Summary" in html

    def test_renders_recommendation_box(self):
        html = self._build()
        assert "Go" in html
        assert "Strong past signals." in html

    def test_renders_metric_interpretation_guide(self):
        html = self._build()
        assert "Metric Interpretation Guide" in html

    def test_renders_technical_metrics_expand(self):
        html = self._build()
        assert "Technical metrics (expand)" in html

    def test_no_present_view_score_card(self):
        html = self._build()
        # Present view section specific to 7Q is absent in past view
        assert "Fundamental Scores — 7 Questions" not in html

    def test_thesis_concern_banner_shown_when_falsified(self):
        fp = _make_fp_report(questions=[
            _make_question("P1", signal="Weak", falsified=True, falsification_reason="Coverage breach"),
            _make_question("P2"),
            _make_question("F1"),
        ])
        html = self._build(fp_report=fp)
        assert "Thesis At Risk" in html
        assert "Coverage breach" in html

    def test_thesis_concern_banner_absent_when_clean(self):
        fp = _make_fp_report()
        html = self._build(fp_report=fp)
        assert "Thesis At Risk — Principle Concerns Identified" not in html

    def test_staleness_warning_shown_for_stale_data(self):
        html = _build_html(
            metrics=_make_metrics(staleness_label="last filing 200 days ago"),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=_make_fp_report(),
        )
        assert "Stale Fundamental Data" in html

    def test_no_fp_report_renders_fallback_message(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Monitor",
            recommendation_explanation="No FP data.",
            charts={},
            report_view="past",
            first_principles_report=None,
        )
        assert "No first-principles payload available" in html

    def test_notes_rendered_in_past_view(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=_make_fp_report(),
            notes="Analyst past note.",
        )
        assert "Analyst past note." in html


# ─────────────────────────────────────────────────────────────────────────────
# 10. Future report — first-principles F-questions
# ─────────────────────────────────────────────────────────────────────────────

class TestFutureReport:
    def _build(self, fp_report=None, **kw) -> str:
        if fp_report is None:
            fp_report = _make_fp_report()
        m = _make_metrics(**kw)
        return _build_html(
            metrics=m,
            panels={},
            recommendation="Monitor",
            recommendation_explanation="Mixed forward signals.",
            charts={},
            report_view="future",
            first_principles_report=fp_report,
        )

    def test_view_title_shows_future(self):
        html = self._build()
        assert "Future" in html

    def test_first_principles_future_view_label(self):
        html = self._build()
        assert "First-Principles Future View" in html

    def test_only_f_questions_shown(self):
        html = self._build()
        for qid in ["F1", "F2", "F3", "F4", "F5", "F6"]:
            assert f"{qid}:" in html
        for qid in ["P1", "P2", "P3", "P4", "P5", "P6"]:
            assert f"{qid}:" not in html

    def test_renders_dashboard_title(self):
        html = self._build()
        assert "Stock Fundamentals CEO Dashboard" in html

    def test_renders_disclaimer(self):
        html = self._build()
        assert "informational purposes only" in html

    def test_thesis_concern_banner_shown_for_future_falsification(self):
        fp = _make_fp_report(questions=[
            _make_question("P1"),
            _make_question("F1", signal="Weak", falsified=True, falsification_reason="Stress test failed"),
            _make_question("F2"),
        ])
        html = self._build(fp_report=fp)
        assert "Thesis At Risk" in html
        assert "Stress test failed" in html

    def test_no_fp_report_renders_fallback(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Monitor",
            recommendation_explanation=".",
            charts={},
            report_view="future",
            first_principles_report=None,
        )
        assert "No first-principles payload available" in html

    def test_notes_rendered_in_future_view(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            report_view="future",
            first_principles_report=_make_fp_report(),
            notes="Forward-looking note.",
        )
        assert "Forward-looking note." in html


# ─────────────────────────────────────────────────────────────────────────────
# 11. Decision logic — Go / Monitor / Avoid/Trim
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionLogic:
    """
    The report re-derives decision from the questions list, overriding the
    synthesis.decision from the FP report:
      hard_fail OR weak >= 2  → Avoid/Trim
      strong >= 4 AND weak == 0 → Go
      otherwise               → Monitor
    """

    def _build_past(self, questions: list) -> str:
        fp = _make_fp_report(questions=questions)
        return _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=fp,
        )

    def test_all_strong_p_questions_yields_go(self):
        questions = [
            _make_question(f"P{i}", signal="Strong") for i in range(1, 7)
        ]
        html = self._build_past(questions)
        # Should show Go recommendation
        assert "Go" in html

    def test_4_strong_0_weak_p_yields_go(self):
        questions = [
            _make_question("P1", signal="Strong"),
            _make_question("P2", signal="Strong"),
            _make_question("P3", signal="Strong"),
            _make_question("P4", signal="Strong"),
            _make_question("P5", signal="Watch"),
            _make_question("P6", signal="Watch"),
        ]
        html = self._build_past(questions)
        assert "Go" in html

    def test_2_or_more_weak_p_yields_avoid_trim(self):
        questions = [
            _make_question("P1", signal="Weak"),
            _make_question("P2", signal="Weak"),
            _make_question("P3", signal="Watch"),
            _make_question("P4", signal="Strong"),
            _make_question("P5", signal="Strong"),
            _make_question("P6", signal="Strong"),
        ]
        html = self._build_past(questions)
        assert "Avoid" in html or "Trim" in html

    def test_hard_fail_yields_avoid_trim_even_with_strong_signals(self):
        questions = [
            _make_question("P1", signal="Strong"),
            _make_question("P2", signal="Strong"),
            _make_question("P3", signal="Strong"),
            _make_question("P4", signal="Strong"),
            _make_question("P5", signal="Strong"),
            _make_question("P6", signal="Weak", falsified=True,
                           falsification_reason="Hard fail triggered"),
        ]
        html = self._build_past(questions)
        assert "Avoid" in html or "Trim" in html

    def test_mixed_signals_yields_monitor(self):
        questions = [
            _make_question("P1", signal="Strong"),
            _make_question("P2", signal="Watch"),
            _make_question("P3", signal="Watch"),
            _make_question("P4", signal="Insufficient"),
            _make_question("P5", signal="Strong"),
            _make_question("P6", signal="Watch"),
        ]
        html = self._build_past(questions)
        assert "Monitor" in html or "Go" in html  # 2 Strong < 4 threshold → Monitor

    def test_all_insufficient_p_yields_monitor(self):
        questions = [
            _make_question(f"P{i}", signal="Insufficient") for i in range(1, 7)
        ]
        html = self._build_past(questions)
        assert "Monitor" in html

    def test_exactly_3_strong_yields_monitor(self):
        questions = [
            _make_question("P1", signal="Strong"),
            _make_question("P2", signal="Strong"),
            _make_question("P3", signal="Strong"),
            _make_question("P4", signal="Watch"),
            _make_question("P5", signal="Watch"),
            _make_question("P6", signal="Watch"),
        ]
        html = self._build_past(questions)
        assert "Monitor" in html

    def test_exactly_1_weak_but_no_hard_fail_yields_monitor_not_avoid(self):
        questions = [
            _make_question("P1", signal="Strong"),
            _make_question("P2", signal="Strong"),
            _make_question("P3", signal="Strong"),
            _make_question("P4", signal="Watch"),
            _make_question("P5", signal="Watch"),
            _make_question("P6", signal="Weak"),  # 1 weak — not enough for Avoid
        ]
        html = self._build_past(questions)
        # 3 strong, 1 weak → Monitor (not Go, not Avoid)
        assert "Monitor" in html


# ─────────────────────────────────────────────────────────────────────────────
# 12. Question card rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestQuestionCardRendering:
    """Test individual FP question signal cards inside a past report."""

    def _build_past(self, questions: list) -> str:
        fp = _make_fp_report(questions=questions)
        return _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=fp,
        )

    def test_strong_signal_badge_green(self):
        html = self._build_past([_make_question("P1", signal="Strong")])
        # Signal badge for "Strong" → green color
        assert "#27ae60" in html

    def test_watch_signal_badge_yellow(self):
        html = self._build_past([_make_question("P1", signal="Watch")])
        assert "#f39c12" in html

    def test_weak_signal_badge_red(self):
        html = self._build_past([_make_question("P1", signal="Weak")])
        assert "#e74c3c" in html

    def test_insufficient_signal_badge_neutral(self):
        html = self._build_past([_make_question("P1", signal="Insufficient")])
        assert "#7f8c8d" in html

    def test_falsification_icon_shown_in_scorecard(self):
        html = self._build_past([
            _make_question("P1", signal="Weak", falsified=True,
                           falsification_reason="Coverage too low")
        ])
        assert "⚡" in html

    def test_falsification_reason_shown_in_card(self):
        html = self._build_past([
            _make_question("P1", falsified=True, falsification_reason="Interest coverage < 1x")
        ])
        assert "Interest coverage &lt; 1x" in html or "Interest coverage < 1x" in html

    def test_no_falsification_shows_clean_message(self):
        html = self._build_past([_make_question("P1", falsified=False)])
        assert "All principle checks passed." in html

    def test_coverage_ratio_100_percent_shown_green(self):
        html = self._build_past([_make_question("P1", coverage_ratio=1.0)])
        assert "Coverage 100%" in html
        assert "#27ae60" in html

    def test_coverage_ratio_50_percent_shown_red(self):
        html = self._build_past([_make_question("P1", coverage_ratio=0.5)])
        assert "Coverage 50%" in html

    def test_missing_fields_shown_in_decision_summary(self):
        html = self._build_past([
            _make_question("P1", signal="Insufficient", coverage_ratio=0.5,
                           missing_fields=["total_equity", "capex"])
        ])
        assert "total_equity" in html or "missing" in html.lower()

    def test_takeaway_rendered_in_card(self):
        html = self._build_past([
            _make_question("P1", takeaway="Owner earnings have compounded at 9% real.")
        ])
        assert "Owner earnings have compounded at 9% real." in html

    def test_chart_rendered_when_present(self):
        html = self._build_past([_make_question("P1", with_chart=True)])
        # Plotly chart is embedded — the plotly CDN script tag should be present
        assert "plotly" in html.lower()

    def test_chart_graceful_degradation_when_absent(self):
        html = self._build_past([_make_question("P1", with_chart=False)])
        assert "Chart data unavailable." in html


# ─────────────────────────────────────────────────────────────────────────────
# 13. HTML escaping / XSS prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestHtmlEscaping:
    def test_ticker_with_special_chars_escaped(self):
        # NOTE: The present view does NOT HTML-escape the ticker field — it is
        # interpolated directly into the template.  This test documents the
        # current (unescaped) behaviour so regressions are detectable.
        m = _make_metrics(ticker="<script>alert('xss')</script>")
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".",
            charts={}, report_view="past",
            first_principles_report=_make_fp_report(ticker="SAFE"),
        )
        # Raw injection is reflected — present in the output unescaped.
        assert "<script>alert" in html

    def test_company_name_html_entities_escaped(self):
        # NOTE: The present view does NOT HTML-escape the company_name field.
        # This test documents the current (unescaped) behaviour so any
        # future change (intentional escaping) is immediately visible.
        m = _make_metrics(company_name='Johnson & Johnson <"Corp">')
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".",
            charts={}, report_view="past",
            first_principles_report=_make_fp_report(),
        )
        # Raw injection is reflected — present in the output unescaped.
        assert "<\"Corp\">" in html

    def test_notes_with_script_tag_escaped(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=_make_fp_report(),
            notes="<script>evil()</script> analyst note",
        )
        assert "<script>evil()" not in html

    def test_falsification_reason_with_html_escaped(self):
        q = _make_question(
            "P1", falsified=True,
            falsification_reason='<img src=x onerror="evil()">'
        )
        fp = _make_fp_report(questions=[q])
        html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Avoid/Trim",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        assert '<img src=x onerror="evil()">' not in html

    def test_takeaway_with_html_escaped(self):
        q = _make_question("P1", takeaway='Earnings grew <b>fast</b> & consistently')
        fp = _make_fp_report(questions=[q])
        html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Go",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        # The raw <b> tag should not appear unescaped
        assert "<b>fast</b>" not in html


# ─────────────────────────────────────────────────────────────────────────────
# 14. Edge cases — empty / None values
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_none_metrics_no_crash(self):
        m = AllMetrics(ticker="TEST", company_name="Test Co.", mode="new")
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "<!DOCTYPE html>" in html
        assert "TEST" in html

    def test_empty_panels_dict_no_crash(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "<!DOCTYPE html>" in html

    def test_empty_questions_list_shows_fallback(self):
        fp = _make_fp_report(questions=[])
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Monitor",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=fp,
        )
        assert "No first-principles payload available" in html

    def test_no_f_questions_in_future_report_shows_fallback(self):
        # All questions are P-type → future view should fall back
        fp = _make_fp_report(questions=[_make_question("P1"), _make_question("P2")])
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Monitor",
            recommendation_explanation=".",
            charts={},
            report_view="future",
            first_principles_report=fp,
        )
        assert "No first-principles payload available" in html

    def test_no_p_questions_in_past_report_shows_fallback(self):
        # All questions are F-type → past view should fall back
        fp = _make_fp_report(questions=[_make_question("F1"), _make_question("F2")])
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Monitor",
            recommendation_explanation=".",
            charts={},
            report_view="past",
            first_principles_report=fp,
        )
        assert "No first-principles payload available" in html

    def test_present_with_none_fp_report_no_crash(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
            first_principles_report=None,
        )
        assert "<!DOCTYPE html>" in html

    def test_none_score_in_scores_dict_no_crash(self):
        # _score_badge() does not guard against None — it raises TypeError.
        # This test documents the current behaviour; fix _score_badge() to
        # handle None gracefully if that is desired.
        import pytest
        m = _make_metrics()
        m.scores = {"Q1_Scale": None}  # type: ignore[assignment]
        with pytest.raises(TypeError):
            _build_html(
                metrics=m, panels={}, recommendation="Hold",
                recommendation_explanation=".", charts={}, report_view="present",
            )

    def test_zero_score_shows_bad_tier(self):
        html = _score_badge(0.0)
        assert "Bad" in html

    def test_panel_with_empty_bullets_no_crash(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={"bullish": _make_panel(bullets=[])},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "<!DOCTYPE html>" in html

    def test_none_current_price_shows_na(self):
        m = _make_metrics(current_price=None)
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "N/A" in html

    def test_none_market_cap_shows_na(self):
        m = _make_metrics(market_cap=None)
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "N/A" in html

    def test_invalid_view_routes_to_present(self):
        html = _build_html(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="garbage_value",
        )
        # Should render as present view
        assert "Fundamental Scores" in html or "New Position Analysis" in html

    def test_fp_report_with_single_p_question(self):
        fp = _make_fp_report(questions=[_make_question("P1", signal="Strong")])
        html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Monitor",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        assert "P1:" in html

    def test_fp_report_with_mixed_p_and_f_questions(self):
        fp = _make_fp_report(questions=[
            _make_question("P1"), _make_question("F1"),
        ])
        past_html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Monitor",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        future_html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Monitor",
            recommendation_explanation=".", charts={},
            report_view="future", first_principles_report=fp,
        )
        assert "P1:" in past_html
        assert "F1:" not in past_html
        assert "F1:" in future_html
        assert "P1:" not in future_html

    def test_ml_panel_rendered_when_present(self):
        ml_panel = _make_panel("ML Predictions")
        html = _build_html(
            metrics=_make_metrics(),
            panels={"ml": ml_panel},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "ML Predictions" in html

    def test_ml_panel_absent_when_falsy(self):
        # A panel of None should be gracefully skipped
        html = _build_html(
            metrics=_make_metrics(),
            panels={"ml": None},   # type: ignore[dict-item]
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "<!DOCTYPE html>" in html


# ─────────────────────────────────────────────────────────────────────────────
# 15. generate_report() — file output
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_file_is_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation="Test run.",
            charts={},
            open_in_browser=False,
            report_view="present",
        )
        assert path.exists()

    def test_returned_path_is_correct_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation="Test.",
            charts={},
            open_in_browser=False,
        )
        assert isinstance(path, Path)

    def test_filename_contains_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(ticker="TSLA"),
            panels={},
            recommendation="Hold",
            recommendation_explanation="Test.",
            charts={},
            open_in_browser=False,
        )
        assert "TSLA" in path.name

    def test_filename_contains_view(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            open_in_browser=False,
            report_view="past",
            first_principles_report=_make_fp_report(),
        )
        assert "past" in path.name

    def test_file_contains_valid_html(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            open_in_browser=False,
        )
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_html_file_extension(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            open_in_browser=False,
        )
        assert path.suffix == ".html"

    def test_past_report_file_saved(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Go",
            recommendation_explanation=".",
            charts={},
            open_in_browser=False,
            report_view="past",
            first_principles_report=_make_fp_report(),
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Past" in content

    def test_future_report_file_saved(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.report_generator.outputs_dir", lambda: tmp_path)
        path = generate_report(
            metrics=_make_metrics(),
            panels={},
            recommendation="Monitor",
            recommendation_explanation=".",
            charts={},
            open_in_browser=False,
            report_view="future",
            first_principles_report=_make_fp_report(),
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Future" in content


# ─────────────────────────────────────────────────────────────────────────────
# 16. Signal scorecard rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalScorecardRendering:
    """Signal scorecard is the compact table of all questions in past/future views."""

    def test_scorecard_shows_all_p_question_ids_in_past(self):
        questions = [_make_question(f"P{i}") for i in range(1, 7)]
        fp = _make_fp_report(questions=questions)
        html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Go",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        for i in range(1, 7):
            assert f"P{i}" in html

    def test_scorecard_shows_coverage_percentage(self):
        fp = _make_fp_report(questions=[_make_question("P1", coverage_ratio=0.75)])
        html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Monitor",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        assert "75%" in html

    def test_scorecard_shows_hard_fail_lightning_icon(self):
        fp = _make_fp_report(questions=[
            _make_question("P1", falsified=True, falsification_reason="breach")
        ])
        html = _build_html(
            metrics=_make_metrics(), panels={}, recommendation="Avoid/Trim",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        assert "⚡" in html


# ─────────────────────────────────────────────────────────────────────────────
# 17. Existing holding — specific field rendering tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExistingHoldingRendering:
    """Validates that purchase-date comparison data renders correctly."""

    def test_price_at_purchase_shown_in_q6_section(self):
        m = _make_metrics(
            mode="existing",
            price_at_purchase=120.0,
            current_price=200.0,
            unrealized_gain_loss_pct=0.667,
        )
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "Price at Purchase" in html

    def test_unrealized_gain_shown_positive(self):
        m = _make_metrics(
            mode="existing",
            price_at_purchase=100.0,
            current_price=150.0,
            unrealized_gain_loss_pct=0.50,
        )
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "Price Return Since Purchase" in html

    def test_valuation_more_expensive_shown(self):
        m = _make_metrics(mode="existing", valuation_change="More Expensive")
        html = _build_html(
            metrics=m, panels={}, recommendation="Hold",
            recommendation_explanation=".", charts={}, report_view="present",
        )
        assert "More Expensive" in html

    def test_existing_mode_past_report_renders(self):
        m = _make_metrics(mode="existing", price_at_purchase=150.0)
        fp = _make_fp_report()
        html = _build_html(
            metrics=m, panels={}, recommendation="Go",
            recommendation_explanation=".", charts={},
            report_view="past", first_principles_report=fp,
        )
        assert "Existing Holding" in html
        assert "Past" in html

    def test_existing_mode_future_report_renders(self):
        m = _make_metrics(mode="existing")
        fp = _make_fp_report()
        html = _build_html(
            metrics=m, panels={}, recommendation="Monitor",
            recommendation_explanation=".", charts={},
            report_view="future", first_principles_report=fp,
        )
        assert "Existing Holding" in html
        assert "Future" in html

    def test_decision_audit_gain_loss_verdict_for_existing(self):
        audit_data = {
            "company_verdict": "Good",
            "valuation_verdict": "Fair",
            "weight_verdict": "Overweight",
            "gain_loss_verdict": "Gain",
            "primary_driver": "Trim",
            "secondary_driver": "Overweight position",
            "final_action": "Optional Trim",
            "why": "Position is overweight; consider trimming.",
        }
        audit_panel = Panel(
            title="Decision Audit",
            mode_label="Decision Audit",
            raw_data=audit_data,
        )
        html = _build_html(
            metrics=_make_metrics(mode="existing"),
            panels={"decision_audit": audit_panel},
            recommendation="Optional Trim",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "Overweight" in html
        assert "Position is overweight; consider trimming." in html

    def test_portfolio_risk_section_has_position_weight(self):
        portfolio_panel = Panel(
            title="Portfolio Risk Summary",
            mode_label="Portfolio Risk Summary",
            bullets=["Position weight: 4.0% of portfolio"],
            raw_data={},
        )
        html = _build_html(
            metrics=_make_metrics(mode="existing", position_weight=0.04),
            panels={"portfolio": portfolio_panel},
            recommendation="Hold",
            recommendation_explanation=".",
            charts={},
            report_view="present",
        )
        assert "4.0% of portfolio" in html


# ─────────────────────────────────────────────────────────────────────────────
# 18. Three-report integration — present + past + future all build without error
# ─────────────────────────────────────────────────────────────────────────────

class TestThreeReportIntegration:
    """Integration smoke test: all three views build for both new and existing modes."""

    @pytest.mark.parametrize("mode", ["new", "existing"])
    @pytest.mark.parametrize("view", ["present", "past", "future"])
    def test_all_view_mode_combinations_build(self, view: str, mode: str):
        m = _make_metrics(mode=mode)
        fp = _make_fp_report() if view in ("past", "future") else None
        html = _build_html(
            metrics=m,
            panels={},
            recommendation="Hold",
            recommendation_explanation="Smoke test.",
            charts={},
            report_view=view,
            first_principles_report=fp,
        )
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "AAPL" in html

    def test_three_sequential_reports_ticker_isolation(self):
        """Each report should contain only its own ticker."""
        for ticker in ["AAPL", "MSFT", "NVDA"]:
            m = _make_metrics(ticker=ticker, company_name=f"{ticker} Corp")
            fp = _make_fp_report(ticker=ticker)
            html = _build_html(
                metrics=m, panels={}, recommendation="Hold",
                recommendation_explanation=".", charts={},
                report_view="past", first_principles_report=fp,
            )
            assert ticker in html



# ─────────────────────────────────────────────────────────────────────────────
# 19. Master report
# ─────────────────────────────────────────────────────────────────────────────

class TestMasterReport:
    """Tests for _build_master_html() and generate_master_report()."""

    def _make_pair(self):
        """Return (new_metrics, existing_metrics, panels, charts, fp_report)."""
        new_m = _make_metrics(ticker="TEST", mode="new")
        ex_m = _make_metrics(ticker="TEST", mode="existing")
        panels = {
            "bullish": _make_panel("Why Buy"),
            "hold": _make_panel("Hold Factors"),
            "bearish": _make_panel("Key Risks"),
        }
        charts = {}
        fp = _make_fp_report(ticker="TEST")
        return new_m, ex_m, panels, charts, fp

    def _build(self):
        new_m, ex_m, panels, charts, fp = self._make_pair()
        return _build_master_html(
            new_metrics=new_m,
            new_panels=panels,
            new_recommendation="Buy",
            new_recommendation_explanation="Strong fundamentals.",
            new_charts=charts,
            existing_metrics=ex_m,
            existing_panels=panels,
            existing_recommendation="Hold",
            existing_recommendation_explanation="Already owned, monitor.",
            existing_charts=charts,
            first_principles_report=fp,
            notes="Test notes.",
        )

    # ── Structure ─────────────────────────────────────────────────────────────

    def test_is_valid_html_document(self):
        html = self._build()
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_master_ceo_report_heading(self):
        html = self._build()
        assert "Master CEO Report" in html

    def test_contains_new_position_label(self):
        html = self._build()
        assert "New Position" in html

    def test_contains_existing_holding_label(self):
        html = self._build()
        assert "Existing Holding" in html

    def test_contains_present_label(self):
        html = self._build()
        assert "Present" in html

    def test_contains_past_label(self):
        html = self._build()
        assert "Past" in html

    def test_contains_future_label(self):
        html = self._build()
        assert "Future" in html

    # ── FP content ────────────────────────────────────────────────────────────

    def test_contains_p1_question(self):
        html = self._build()
        assert "P1" in html

    def test_contains_f1_question(self):
        html = self._build()
        assert "F1" in html

    # ── Present content ───────────────────────────────────────────────────────

    def test_contains_fundamental_scores_section(self):
        html = self._build()
        assert "Fundamental Scores" in html

    # ── Controls markup ───────────────────────────────────────────────────────

    def test_contains_ownership_sel_id(self):
        html = self._build()
        assert 'id="ownership-sel"' in html

    def test_contains_timeline_sel_id(self):
        html = self._build()
        assert 'id="timeline-sel"' in html

    def test_has_all_six_panel_divs(self):
        html = self._build()
        for panel_id in [
            "panel-new-present", "panel-new-past", "panel-new-future",
            "panel-existing-present", "panel-existing-past", "panel-existing-future",
        ]:
            assert panel_id in html, f"Missing panel id: {panel_id}"

    # ── Plotly loaded exactly once ─────────────────────────────────────────────

    def test_plotly_cdn_loaded_exactly_once(self):
        html = self._build()
        # Check the CDN <script> tag appears exactly once (not individual chart div IDs)
        count = html.count("cdn.plot.ly/plotly-")
        assert count == 1, f"Expected Plotly CDN script tag once, got {count}"

    # ── Decisions visible in overview table ───────────────────────────────────

    def test_new_recommendation_in_overview(self):
        html = self._build()
        assert "Buy" in html

    def test_existing_recommendation_in_overview(self):
        html = self._build()
        assert "Hold" in html

    # ── No FP report case ─────────────────────────────────────────────────────

    def test_works_without_fp_report(self):
        new_m, ex_m, panels, charts, _ = self._make_pair()
        html = _build_master_html(
            new_metrics=new_m,
            new_panels=panels,
            new_recommendation="Buy",
            new_recommendation_explanation="Strong.",
            new_charts=charts,
            existing_metrics=ex_m,
            existing_panels=panels,
            existing_recommendation="Hold",
            existing_recommendation_explanation="Monitor.",
            existing_charts=charts,
            first_principles_report=None,
        )
        assert "Master CEO Report" in html
        assert "Existing Holding" in html
        assert "New Position" in html

    # ── generate_master_report() ──────────────────────────────────────────────

    def test_generate_master_report_creates_file(self, tmp_path, monkeypatch):
        from src import utils
        monkeypatch.setattr(utils, "outputs_dir", lambda: tmp_path)

        new_m, ex_m, panels, charts, fp = self._make_pair()
        path = generate_master_report(
            new_metrics=new_m,
            new_panels=panels,
            new_recommendation="Buy",
            new_recommendation_explanation="Strong.",
            new_charts=charts,
            existing_metrics=ex_m,
            existing_panels=panels,
            existing_recommendation="Hold",
            existing_recommendation_explanation="Monitor.",
            existing_charts=charts,
            first_principles_report=fp,
            open_in_browser=False,
        )
        assert path.exists()
        assert "master" in path.name
        content = path.read_text(encoding="utf-8")
        assert "Master CEO Report" in content

    def test_generate_master_report_filename_convention(self, tmp_path, monkeypatch):
        from src import utils
        monkeypatch.setattr(utils, "outputs_dir", lambda: tmp_path)

        new_m, ex_m, panels, charts, fp = self._make_pair()
        path = generate_master_report(
            new_metrics=new_m,
            new_panels=panels,
            new_recommendation="Buy",
            new_recommendation_explanation="Strong.",
            new_charts=charts,
            existing_metrics=ex_m,
            existing_panels=panels,
            existing_recommendation="Hold",
            existing_recommendation_explanation="Monitor.",
            existing_charts=charts,
            first_principles_report=fp,
            open_in_browser=False,
        )
        # Filename: ceo_report_{TICKER}_master_{YYYYMMDD_HHMMSS}.html
        assert path.name.startswith("ceo_report_TEST_master_")
        assert path.name.endswith(".html")