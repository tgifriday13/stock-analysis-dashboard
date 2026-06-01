"""
tests/test_utils.py — Unit tests for src/utils.py pure helper functions.

All functions here are pure (no I/O, no network) so tests are fast and
deterministic.  File I/O helpers (load_json / save_json / log_session) are
tested with pytest's tmp_path fixture.
"""

import math
import json
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.utils import (
    fmt_currency,
    fmt_pct,
    fmt_multiple,
    fmt_number,
    safe_divide,
    safe_float,
    safe_pct_change,
    score_metric,
    delta_arrow,
    delta_color,
    parse_date,
    days_since,
    load_json,
    save_json,
    log_session,
    project_root,
    outputs_dir,
    config_dir,
)


# ── fmt_currency ─────────────────────────────────────────────────────────────

class TestFmtCurrency:
    def test_none_returns_na(self):
        assert fmt_currency(None) == "N/A"

    def test_nan_returns_na(self):
        assert fmt_currency(float("nan")) == "N/A"

    def test_trillions(self):
        assert fmt_currency(2.5e12) == "$2.50T"

    def test_billions(self):
        assert fmt_currency(3.2e9) == "$3.20B"

    def test_millions(self):
        assert fmt_currency(4.5e6) == "$4.50M"

    def test_thousands(self):
        assert fmt_currency(1500) == "$1.50K"

    def test_small_value(self):
        assert fmt_currency(9.99) == "$9.99"

    def test_negative_billions(self):
        result = fmt_currency(-2.3e9)
        assert result == "-$2.30B"

    def test_negative_small(self):
        assert fmt_currency(-5.0) == "-$5.00"

    def test_zero(self):
        assert fmt_currency(0) == "$0.00"

    def test_custom_decimals(self):
        assert fmt_currency(1e9, decimals=0) == "$1B"

    def test_exact_boundary_billion(self):
        assert fmt_currency(1e9) == "$1.00B"

    def test_exact_boundary_million(self):
        assert fmt_currency(1e6) == "$1.00M"

    def test_exact_boundary_thousand(self):
        assert fmt_currency(1e3) == "$1.00K"


# ── fmt_pct ──────────────────────────────────────────────────────────────────

class TestFmtPct:
    def test_none_returns_na(self):
        assert fmt_pct(None) == "N/A"

    def test_nan_returns_na(self):
        assert fmt_pct(float("nan")) == "N/A"

    def test_positive_ratio(self):
        assert fmt_pct(0.123) == "12.3%"

    def test_negative_ratio(self):
        assert fmt_pct(-0.05) == "-5.0%"

    def test_zero(self):
        assert fmt_pct(0.0) == "0.0%"

    def test_one(self):
        assert fmt_pct(1.0) == "100.0%"

    def test_custom_decimals(self):
        assert fmt_pct(0.1234, decimals=2) == "12.34%"

    def test_over_one_hundred(self):
        assert fmt_pct(1.5) == "150.0%"


# ── fmt_multiple ─────────────────────────────────────────────────────────────

class TestFmtMultiple:
    def test_none_returns_na(self):
        assert fmt_multiple(None) == "N/A"

    def test_nan_returns_na(self):
        assert fmt_multiple(float("nan")) == "N/A"

    def test_normal(self):
        assert fmt_multiple(15.5) == "15.5x"

    def test_zero(self):
        assert fmt_multiple(0.0) == "0.0x"

    def test_negative(self):
        assert fmt_multiple(-3.2) == "-3.2x"

    def test_custom_suffix(self):
        assert fmt_multiple(2.5, suffix="×") == "2.5×"

    def test_custom_decimals(self):
        assert fmt_multiple(15.55, decimals=2) == "15.55x"


# ── fmt_number ───────────────────────────────────────────────────────────────

class TestFmtNumber:
    def test_none_returns_na(self):
        assert fmt_number(None) == "N/A"

    def test_nan_returns_na(self):
        assert fmt_number(float("nan")) == "N/A"

    def test_trillions(self):
        assert fmt_number(1.5e12) == "1.50T"

    def test_billions(self):
        assert fmt_number(2.0e9) == "2.00B"

    def test_millions(self):
        assert fmt_number(3.0e6) == "3.00M"

    def test_thousands(self):
        assert fmt_number(4000) == "4.00K"

    def test_small(self):
        assert fmt_number(5.0) == "5.00"

    def test_negative(self):
        assert fmt_number(-2e9) == "-2.00B"


# ── safe_divide ──────────────────────────────────────────────────────────────

class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10.0, 4.0) == pytest.approx(2.5)

    def test_zero_denominator_returns_none(self):
        assert safe_divide(10.0, 0) is None

    def test_none_denominator_returns_none(self):
        assert safe_divide(10.0, None) is None

    def test_none_numerator_returns_none(self):
        assert safe_divide(None, 5.0) is None

    def test_both_none(self):
        assert safe_divide(None, None) is None

    def test_string_inputs_return_none(self):
        assert safe_divide("a", "b") is None

    def test_negative_result(self):
        assert safe_divide(-10.0, 2.0) == pytest.approx(-5.0)

    def test_small_denominator(self):
        result = safe_divide(1.0, 0.001)
        assert result == pytest.approx(1000.0)


# ── safe_float ───────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_int_input(self):
        assert safe_float(42) == 42.0

    def test_float_input(self):
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_string_number(self):
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_none_returns_none(self):
        assert safe_float(None) is None

    def test_non_numeric_string(self):
        assert safe_float("abc") is None

    def test_nan_returns_none(self):
        assert safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert safe_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert safe_float(float("-inf")) is None

    def test_zero(self):
        assert safe_float(0) == 0.0

    def test_negative_string(self):
        assert safe_float("-5.5") == pytest.approx(-5.5)

    def test_empty_string(self):
        assert safe_float("") is None


# ── safe_pct_change ──────────────────────────────────────────────────────────

class TestSafePctChange:
    def test_positive_change(self):
        assert safe_pct_change(110, 100) == pytest.approx(0.10)

    def test_negative_change(self):
        assert safe_pct_change(90, 100) == pytest.approx(-0.10)

    def test_none_new_val(self):
        assert safe_pct_change(None, 100) is None

    def test_none_old_val(self):
        assert safe_pct_change(110, None) is None

    def test_zero_old_val(self):
        assert safe_pct_change(110, 0) is None

    def test_both_none(self):
        assert safe_pct_change(None, None) is None

    def test_negative_base_uses_abs(self):
        # (90 - (-100)) / abs(-100) = 190/100 = 1.9
        assert safe_pct_change(90, -100) == pytest.approx(1.9)

    def test_same_value_is_zero(self):
        assert safe_pct_change(100, 100) == pytest.approx(0.0)


# ── score_metric ─────────────────────────────────────────────────────────────

class TestScoreMetric:
    # higher_is_better
    def test_strong_higher_is_better(self):
        assert score_metric(0.25, strong=0.20, acceptable=0.10) == 2

    def test_acceptable_higher_is_better(self):
        assert score_metric(0.15, strong=0.20, acceptable=0.10) == 1

    def test_weak_higher_is_better(self):
        assert score_metric(0.05, strong=0.20, acceptable=0.10) == 0

    def test_exactly_strong_boundary(self):
        assert score_metric(0.20, strong=0.20, acceptable=0.10) == 2

    def test_exactly_acceptable_boundary(self):
        assert score_metric(0.10, strong=0.20, acceptable=0.10) == 1

    def test_none_returns_0(self):
        assert score_metric(None, strong=0.20, acceptable=0.10) == 0

    # lower_is_better
    def test_strong_lower_is_better(self):
        assert score_metric(1.0, strong=1.5, acceptable=2.5, higher_is_better=False) == 2

    def test_acceptable_lower_is_better(self):
        assert score_metric(2.0, strong=1.5, acceptable=2.5, higher_is_better=False) == 1

    def test_weak_lower_is_better(self):
        assert score_metric(3.0, strong=1.5, acceptable=2.5, higher_is_better=False) == 0

    def test_exactly_strong_boundary_lower_better(self):
        assert score_metric(1.5, strong=1.5, acceptable=2.5, higher_is_better=False) == 2

    def test_none_lower_is_better_returns_0(self):
        assert score_metric(None, strong=1.5, acceptable=2.5, higher_is_better=False) == 0


# ── delta_arrow ──────────────────────────────────────────────────────────────

class TestDeltaArrow:
    def test_none_returns_dash(self):
        assert delta_arrow(None) == "—"

    def test_zero_returns_dash(self):
        assert delta_arrow(0) == "—"

    def test_positive_higher_is_better(self):
        assert delta_arrow(0.05, higher_is_better=True) == "▲"

    def test_negative_higher_is_better(self):
        assert delta_arrow(-0.05, higher_is_better=True) == "▼"

    def test_positive_lower_is_better(self):
        assert delta_arrow(0.05, higher_is_better=False) == "▼ (worse)"

    def test_negative_lower_is_better(self):
        assert delta_arrow(-0.05, higher_is_better=False) == "▲ (better)"


# ── delta_color ──────────────────────────────────────────────────────────────

class TestDeltaColor:
    def test_none_returns_grey(self):
        assert delta_color(None) == "#888888"

    def test_positive_higher_is_better_is_green(self):
        assert delta_color(0.1, higher_is_better=True) == "#27ae60"

    def test_negative_higher_is_better_is_red(self):
        assert delta_color(-0.1, higher_is_better=True) == "#e74c3c"

    def test_positive_lower_is_better_is_red(self):
        assert delta_color(0.1, higher_is_better=False) == "#e74c3c"

    def test_negative_lower_is_better_is_green(self):
        assert delta_color(-0.1, higher_is_better=False) == "#27ae60"

    def test_near_zero_below_threshold_is_grey(self):
        # 0.0005 < 0.001 threshold → neutral
        assert delta_color(0.0005) == "#888888"

    def test_near_zero_negative_is_grey(self):
        assert delta_color(-0.0005) == "#888888"


# ── parse_date ───────────────────────────────────────────────────────────────

class TestParseDate:
    def test_iso_format(self):
        assert parse_date("2023-01-15") == date(2023, 1, 15)

    def test_us_format(self):
        assert parse_date("01/15/2023") == date(2023, 1, 15)

    def test_eu_format(self):
        assert parse_date("15/01/2023") == date(2023, 1, 15)

    def test_compact_format(self):
        assert parse_date("20230115") == date(2023, 1, 15)

    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert parse_date("") is None

    def test_invalid_string_returns_none(self):
        assert parse_date("not-a-date") is None

    def test_partial_date_returns_none(self):
        assert parse_date("2023-01") is None


# ── days_since ───────────────────────────────────────────────────────────────

class TestDaysSince:
    def test_none_returns_none(self):
        assert days_since(None) is None

    def test_today_is_zero(self):
        assert days_since(date.today()) == 0

    def test_thirty_days_ago(self):
        d = date.today() - timedelta(days=30)
        assert days_since(d) == 30

    def test_one_year_ago(self):
        d = date.today() - timedelta(days=365)
        assert days_since(d) == 365


# ── load_json / save_json ─────────────────────────────────────────────────────

class TestJsonHelpers:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        data = {"x": 1}
        path = tmp_path / "deep" / "nested" / "file.json"
        save_json(data, path)
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "missing.json")


# ── path helpers ─────────────────────────────────────────────────────────────

class TestPathHelpers:
    def test_project_root_is_directory(self):
        assert project_root().is_dir()

    def test_config_dir_has_default_thresholds(self):
        assert (config_dir() / "default_thresholds.json").exists()

    def test_outputs_dir_type(self):
        # Just ensure it returns a Path object
        assert isinstance(outputs_dir(), Path)
