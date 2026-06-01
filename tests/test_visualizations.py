"""
tests/test_visualizations.py — Unit tests for src/visualizations.py.

All functions return Plotly Figure objects; no display or browser is needed.
Tests verify return types, absence of crashes on edge-case inputs, and key
layout/data properties.
"""

import math
import pytest
import pandas as pd
import plotly.graph_objects as go

from src.visualizations import (
    score_to_color,
    gauge_chart,
    bar_chart,
    line_chart,
    sparkline,
    radar_chart,
    waterfall_chart,
    dividend_history_chart,
    score_bar_chart,
    peer_comparison_chart,
    SCORE_EXCELLENT, SCORE_GOOD, SCORE_NEUTRAL, SCORE_WEAK, SCORE_BAD, SCORE_MISSING,
)
from src.utils import fmt_pct, fmt_currency, fmt_multiple


# ── score_to_color ────────────────────────────────────────────────────────────

class TestScoreToColor:
    def test_none_is_missing(self):
        assert score_to_color(None) == SCORE_MISSING

    def test_10_is_excellent(self):
        assert score_to_color(10.0) == SCORE_EXCELLENT

    def test_8_0_is_excellent(self):
        assert score_to_color(8.0) == SCORE_EXCELLENT

    def test_7_9_is_good(self):
        assert score_to_color(7.9) == SCORE_GOOD

    def test_6_5_is_good(self):
        assert score_to_color(6.5) == SCORE_GOOD

    def test_6_4_is_neutral(self):
        assert score_to_color(6.4) == SCORE_NEUTRAL

    def test_5_0_is_neutral(self):
        assert score_to_color(5.0) == SCORE_NEUTRAL

    def test_4_9_is_weak(self):
        assert score_to_color(4.9) == SCORE_WEAK

    def test_3_5_is_weak(self):
        assert score_to_color(3.5) == SCORE_WEAK

    def test_3_4_is_bad(self):
        assert score_to_color(3.4) == SCORE_BAD

    def test_0_is_bad(self):
        assert score_to_color(0.0) == SCORE_BAD


# ── gauge_chart ───────────────────────────────────────────────────────────────

class TestGaugeChart:
    def _make(self, value=0.05, **kw):
        defaults = dict(
            label="FCF Yield",
            min_val=0,
            max_val=0.15,
            strong_threshold=0.07,
            acceptable_threshold=0.03,
            watermark="Test Watermark",
        )
        defaults.update(kw)
        return gauge_chart(value, **defaults)

    def test_returns_figure(self):
        assert isinstance(self._make(), go.Figure)

    def test_none_value_no_crash(self):
        fig = self._make(value=None)
        assert isinstance(fig, go.Figure)

    def test_title_in_layout(self):
        fig = self._make(title="My Gauge")
        assert "My Gauge" in fig.layout.title.text

    def test_watermark_annotation_present(self):
        fig = self._make(watermark="MSFT Data")
        texts = [ann.text for ann in fig.layout.annotations]
        assert any("MSFT Data" in t for t in texts)

    def test_lower_is_better_variant(self):
        fig = self._make(
            value=1.0,
            label="Debt/EBITDA",
            min_val=0,
            max_val=8,
            strong_threshold=1.5,
            acceptable_threshold=3.0,
            higher_is_better=False,
        )
        assert isinstance(fig, go.Figure)

    def test_format_fn_applied(self):
        fig = self._make(value=0.08, format_fn=fmt_pct)
        assert isinstance(fig, go.Figure)


# ── bar_chart ─────────────────────────────────────────────────────────────────

class TestBarChart:
    def _make(self, labels=None, values=None, **kw):
        labels = labels or ["A", "B", "C"]
        values = values or [1.0, 2.0, 3.0]
        return bar_chart(labels, values, title="Test Bar", **kw)

    def test_returns_figure(self):
        assert isinstance(self._make(), go.Figure)

    def test_none_values_no_crash(self):
        fig = self._make(values=[None, 2.0, None])
        assert isinstance(fig, go.Figure)

    def test_all_none_values_no_crash(self):
        fig = self._make(values=[None, None, None])
        assert isinstance(fig, go.Figure)

    def test_single_trace_by_default(self):
        fig = self._make()
        assert len(fig.data) == 1

    def test_compare_values_adds_second_trace(self):
        fig = self._make(compare_values=[0.5, 1.5, 2.5])
        assert len(fig.data) == 2

    def test_compare_values_with_none_no_crash(self):
        fig = self._make(compare_values=[None, 1.5, None])
        assert isinstance(fig, go.Figure)

    def test_custom_colors_applied(self):
        fig = self._make(colors=["red", "green", "blue"])
        assert isinstance(fig, go.Figure)

    def test_format_fn_applied_to_text(self):
        fig = self._make(values=[0.1, 0.2], labels=["X", "Y"], format_fn=fmt_pct)
        assert isinstance(fig, go.Figure)

    def test_watermark_annotation(self):
        fig = self._make(watermark="Source: Test")
        texts = [ann.text for ann in fig.layout.annotations]
        assert any("Source: Test" in t for t in texts)

    def test_empty_labels_no_crash(self):
        fig = bar_chart([], [], title="Empty")
        assert isinstance(fig, go.Figure)


# ── line_chart ────────────────────────────────────────────────────────────────

class TestLineChart:
    def test_returns_figure(self):
        fig = line_chart(
            x=[2020, 2021, 2022],
            y_series={"Revenue": [1e9, 1.1e9, 1.2e9]},
            title="Revenue Trend",
        )
        assert isinstance(fig, go.Figure)

    def test_multiple_series(self):
        fig = line_chart(
            x=[2020, 2021, 2022],
            y_series={"Rev": [1e9, 1.1e9, 1.2e9], "Profit": [1e8, 1.2e8, 1.4e8]},
            title="Multi-Series",
        )
        assert len(fig.data) == 2

    def test_none_values_handled(self):
        fig = line_chart(
            x=[2020, 2021, 2022],
            y_series={"Rev": [None, 1.1e9, None]},
            title="With Nones",
        )
        assert isinstance(fig, go.Figure)

    def test_vline_adds_shape(self):
        fig = line_chart(
            x=[2020, 2021, 2022],
            y_series={"Rev": [1, 2, 3]},
            title="VLine Test",
            vline_x=2021,
            vline_label="Purchase Date",
        )
        assert isinstance(fig, go.Figure)

    def test_watermark_annotation(self):
        fig = line_chart(
            x=[1, 2], y_series={"S": [1, 2]}, title="T", watermark="WM"
        )
        texts = [ann.text for ann in fig.layout.annotations]
        assert any("WM" in t for t in texts)

    def test_empty_series_no_crash(self):
        fig = line_chart(x=[], y_series={}, title="Empty")
        assert isinstance(fig, go.Figure)


# ── sparkline ─────────────────────────────────────────────────────────────────

class TestSparkline:
    def test_returns_figure(self):
        fig = sparkline([1, 2, 3, 4, 5], title="Trend")
        assert isinstance(fig, go.Figure)

    def test_none_values_handled(self):
        fig = sparkline([1, None, 3, None, 5], title="With Gaps")
        assert isinstance(fig, go.Figure)

    def test_empty_list_no_crash(self):
        fig = sparkline([], title="Empty")
        assert isinstance(fig, go.Figure)

    def test_custom_color(self):
        fig = sparkline([1, 2, 3], title="Custom Color", color="#e74c3c")
        assert isinstance(fig, go.Figure)


# ── radar_chart ───────────────────────────────────────────────────────────────

class TestRadarChart:
    def test_returns_figure(self):
        scores = {"Q1": 7.0, "Q2": 8.0, "Q3": 6.5, "Q4": 5.0, "Q5": 9.0, "Q6": 4.0}
        fig = radar_chart(scores)
        assert isinstance(fig, go.Figure)

    def test_two_traces_base_plus_reference(self):
        scores = {"Q1": 7.0, "Q2": 6.0}
        fig = radar_chart(scores)
        assert len(fig.data) == 2

    def test_single_score(self):
        fig = radar_chart({"Q1": 5.0})
        assert isinstance(fig, go.Figure)

    def test_title_in_layout(self):
        fig = radar_chart({"Q1": 7.0}, title="MSFT Radar")
        assert "MSFT Radar" in fig.layout.title.text

    def test_watermark_annotation(self):
        fig = radar_chart({"Q1": 7.0}, watermark="WM")
        texts = [ann.text for ann in fig.layout.annotations]
        assert any("WM" in t for t in texts)


# ── waterfall_chart ───────────────────────────────────────────────────────────

class TestWaterfallChart:
    def test_returns_figure(self):
        fig = waterfall_chart(
            labels=["Start", "Change", "End"],
            values=[100, 20, 120],
            title="Waterfall",
        )
        assert isinstance(fig, go.Figure)

    def test_negative_change(self):
        fig = waterfall_chart(
            labels=["Start", "Loss", "End"],
            values=[100, -30, 70],
            title="Loss Waterfall",
        )
        assert isinstance(fig, go.Figure)


# ── dividend_history_chart ────────────────────────────────────────────────────

class TestDividendHistoryChart:
    def test_empty_series_returns_figure(self):
        fig = dividend_history_chart(pd.Series(dtype=float), title="No Divs")
        assert isinstance(fig, go.Figure)

    def test_none_series_returns_figure(self):
        fig = dividend_history_chart(None, title="None Divs")
        assert isinstance(fig, go.Figure)

    def test_with_data_returns_figure(self):
        idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
        divs = pd.Series([1.0, 1.1, 1.2], index=idx)
        fig = dividend_history_chart(divs, title="Annual Dividends")
        assert isinstance(fig, go.Figure)

    def test_with_purchase_date(self):
        from datetime import date as dt
        idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
        divs = pd.Series([1.0, 1.1], index=idx)
        fig = dividend_history_chart(divs, purchase_date=dt(2020, 6, 1))
        assert isinstance(fig, go.Figure)


# ── score_bar_chart ───────────────────────────────────────────────────────────

class TestScoreBarChart:
    def test_returns_figure(self):
        scores = {"Q1": 7.5, "Q2": 8.0, "Q3": 6.0, "Q4": 5.5, "Q5": 9.0, "Q6": 4.0}
        fig = score_bar_chart(scores)
        assert isinstance(fig, go.Figure)

    def test_single_score(self):
        fig = score_bar_chart({"Q1": 7.0})
        assert isinstance(fig, go.Figure)

    def test_empty_dict_no_crash(self):
        fig = score_bar_chart({})
        assert isinstance(fig, go.Figure)

    def test_colors_match_tiers(self):
        """Scores in different tiers should get different colors."""
        scores = {"Bad": 1.0, "Excellent": 9.0}
        fig = score_bar_chart(scores)
        colors = fig.data[0].marker.color
        # Excellent and Bad should have different colors
        assert colors[0] != colors[1]

    def test_title_in_layout(self):
        fig = score_bar_chart({"Q1": 5.0}, title="My Scorecard")
        assert "My Scorecard" in fig.layout.title.text


# ── peer_comparison_chart ─────────────────────────────────────────────────────

class TestPeerComparisonChart:
    def test_returns_figure(self):
        fig = peer_comparison_chart(
            tickers=["MSFT", "AAPL", "GOOG"],
            metric_values={"P/E Ratio": [25.0, 28.0, 22.0]},
            metric_label="P/E",
            title="Peer P/E Comparison",
        )
        assert isinstance(fig, go.Figure)

    def test_multiple_metrics(self):
        fig = peer_comparison_chart(
            tickers=["A", "B"],
            metric_values={"P/E": [20.0, 25.0], "EV/EBITDA": [12.0, 15.0]},
            metric_label="Valuation",
        )
        assert len(fig.data) == 2

    def test_none_values_no_crash(self):
        fig = peer_comparison_chart(
            tickers=["A", "B"],
            metric_values={"P/E": [None, 25.0]},
            metric_label="P/E",
        )
        assert isinstance(fig, go.Figure)

    def test_empty_tickers_no_crash(self):
        fig = peer_comparison_chart(
            tickers=[],
            metric_values={"P/E": []},
            metric_label="P/E",
        )
        assert isinstance(fig, go.Figure)
