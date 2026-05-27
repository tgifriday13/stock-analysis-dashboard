from __future__ import annotations

from datetime import date, datetime, timezone

from src.first_principles.engine import FirstPrinciplesEngine
from src.first_principles.schemas import (
    EngineMeta,
    FirstPrinciplesReport,
    PanelCoverage,
    QuestionResult,
    SignalSynthesis,
)


def _q(qid: str) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question=f"Question {qid}",
        metric_family=["m"],
        signal="Strong",
        falsified=False,
        falsification_reason=None,
        decision_reason="ok",
        thresholds={},
        metrics={"x": 1.0},
        percentile_overlay={},
        coverage=PanelCoverage(required_fields=["a"], available_fields=["a"], missing_fields=[], coverage_ratio=1.0),
    )


def test_filter_questions_by_time_view():
    engine = FirstPrinciplesEngine(force_refresh=False)
    qs = [_q("P1"), _q("P2"), _q("F1"), _q("F2")]

    past = engine._filter_questions_by_time_view(qs, "past")
    future = engine._filter_questions_by_time_view(qs, "future")
    all_q = engine._filter_questions_by_time_view(qs, "all")

    assert [q.question_id for q in past] == ["P1", "P2"]
    assert [q.question_id for q in future] == ["F1", "F2"]
    assert [q.question_id for q in all_q] == ["P1", "P2", "F1", "F2"]


def test_html_title_changes_with_view():
    engine = FirstPrinciplesEngine(force_refresh=False)
    report = FirstPrinciplesReport(
        meta=EngineMeta(
            ticker="MSFT",
            cik="0000789019",
            run_timestamp=datetime.now(timezone.utc),
            as_of_date=date(2026, 5, 27),
            accepted_cutoff_rule="acceptance_date + 1 business day",
            sector_template="non_financial",
            data_freshness={},
        ),
        questions=[_q("P1"), _q("F1")],
        synthesis=SignalSynthesis(
            decision="Go",
            hard_fail=False,
            reasons=[],
            strong_count=2,
            watch_count=0,
            weak_count=0,
            insufficient_count=0,
        ),
    )

    past_html = engine._to_html(report, time_view="past")
    future_html = engine._to_html(report, time_view="future")

    assert "Past View" in past_html
    assert "Future View" in future_html
    assert "P1:" in past_html
    assert "F1:" not in past_html
    assert "F1:" in future_html
    assert "P1:" not in future_html
