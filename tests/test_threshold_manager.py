"""
tests/test_threshold_manager.py — Unit tests for src/threshold_manager.py.

ThresholdManager reads config/default_thresholds.json at import time, so all
tests require the file to exist (it does in the repo).  Save/load tests use
monkeypatch to redirect USER_THRESHOLDS_DIR to a tmp directory.
"""

import json
import pytest
from pathlib import Path

import src.threshold_manager as tm_module
from src.threshold_manager import ThresholdManager


# ── Basic instantiation & get ─────────────────────────────────────────────────

class TestThresholdManagerBasic:
    def test_instantiation_no_sector(self):
        tm = ThresholdManager()
        assert tm is not None

    def test_get_known_growth_key(self):
        tm = ThresholdManager()
        val = tm.get("growth.revenue_cagr_1yr_strong")
        assert isinstance(val, float)
        assert val > 0

    def test_get_known_valuation_key(self):
        tm = ThresholdManager()
        val = tm.get("valuation.pe_cheap")
        assert isinstance(val, (int, float))
        assert val > 0

    def test_get_missing_key_returns_default(self):
        tm = ThresholdManager()
        assert tm.get("nonexistent.key", default=99) == 99

    def test_get_missing_key_none_default(self):
        tm = ThresholdManager()
        assert tm.get("nonexistent.key") is None

    def test_get_balance_sheet_keys(self):
        tm = ThresholdManager()
        assert tm.get("balance_sheet.debt_to_ebitda_strong") is not None
        assert tm.get("balance_sheet.debt_to_ebitda_danger") is not None

    def test_get_profitability_keys(self):
        tm = ThresholdManager()
        assert tm.get("profitability.operating_margin_strong") is not None
        assert tm.get("profitability.roic_strong") is not None

    def test_get_cashflow_keys(self):
        tm = ThresholdManager()
        assert tm.get("cashflow.fcf_yield_strong") is not None
        assert tm.get("cashflow.fcf_margin_strong") is not None

    def test_get_dividend_keys(self):
        tm = ThresholdManager()
        assert tm.get("dividends.payout_ratio_safe") is not None
        assert tm.get("dividends.fcf_coverage_strong") is not None


# ── Overrides ─────────────────────────────────────────────────────────────────

class TestOverrides:
    def test_override_single_key(self):
        tm = ThresholdManager()
        tm.override("valuation.pe_cheap", 10)
        assert tm.get("valuation.pe_cheap") == 10

    def test_override_takes_precedence_over_base(self):
        tm = ThresholdManager()
        original = tm.get("valuation.pe_cheap")
        tm.override("valuation.pe_cheap", original + 100)
        assert tm.get("valuation.pe_cheap") == original + 100

    def test_override_many(self):
        tm = ThresholdManager()
        tm.override_many({"valuation.pe_cheap": 5, "valuation.pe_fair": 10})
        assert tm.get("valuation.pe_cheap") == 5
        assert tm.get("valuation.pe_fair") == 10

    def test_reset_overrides_reverts_to_base(self):
        tm = ThresholdManager()
        original = tm.get("valuation.pe_cheap")
        tm.override("valuation.pe_cheap", 999)
        tm.reset_overrides()
        assert tm.get("valuation.pe_cheap") == original

    def test_override_new_key_not_in_base(self):
        tm = ThresholdManager()
        tm.override("custom.my_key", 42)
        assert tm.get("custom.my_key") == 42

    def test_reset_clears_all_overrides(self):
        tm = ThresholdManager()
        tm.override_many({"a.b": 1, "c.d": 2})
        tm.reset_overrides()
        assert tm.get("a.b") is None
        assert tm.get("c.d") is None


# ── Save / Load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm_module, "USER_THRESHOLDS_DIR", tmp_path)
        tm = ThresholdManager()
        tm.override("valuation.pe_cheap", 12)
        path = tm.save("test_set")
        assert path.exists()

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm_module, "USER_THRESHOLDS_DIR", tmp_path)
        tm1 = ThresholdManager()
        tm1.override("valuation.pe_cheap", 12)
        tm1.save("my_prefs")

        tm2 = ThresholdManager.load("my_prefs")
        assert tm2.get("valuation.pe_cheap") == 12

    def test_save_persists_sector(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm_module, "USER_THRESHOLDS_DIR", tmp_path)
        tm = ThresholdManager(sector="Technology")
        tm.save("tech_set")

        tm2 = ThresholdManager.load("tech_set")
        assert tm2._sector == "Technology"

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm_module, "USER_THRESHOLDS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            ThresholdManager.load("does_not_exist")

    def test_list_saved_empty_when_no_dir(self, tmp_path, monkeypatch):
        empty = tmp_path / "no_such_dir"
        monkeypatch.setattr(tm_module, "USER_THRESHOLDS_DIR", empty)
        assert ThresholdManager.list_saved() == []

    def test_list_saved_returns_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm_module, "USER_THRESHOLDS_DIR", tmp_path)
        tm = ThresholdManager()
        tm.save("alpha")
        tm.save("beta")
        names = ThresholdManager.list_saved()
        assert "alpha" in names
        assert "beta" in names


# ── Sector adjustments ────────────────────────────────────────────────────────

class TestSectorAdjustments:
    def test_technology_has_higher_pe(self):
        """Technology sector gets a PE multiplier > 1 → higher PE thresholds."""
        generic = ThresholdManager()
        tech = ThresholdManager(sector="Technology")
        assert tech.get("valuation.pe_cheap") > generic.get("valuation.pe_cheap")
        assert tech.get("valuation.pe_fair") > generic.get("valuation.pe_fair")
        assert tech.get("valuation.pe_expensive") > generic.get("valuation.pe_expensive")

    def test_technology_higher_gross_margin(self):
        generic = ThresholdManager()
        tech = ThresholdManager(sector="Technology")
        assert tech.get("profitability.gross_margin_strong") > generic.get("profitability.gross_margin_strong")

    def test_unknown_sector_same_as_generic(self):
        generic = ThresholdManager()
        unk = ThresholdManager(sector="UnknownSectorXYZ")
        assert unk.get("valuation.pe_cheap") == generic.get("valuation.pe_cheap")

    def test_no_sector_same_as_generic(self):
        generic = ThresholdManager()
        no_sector = ThresholdManager(sector=None)
        assert no_sector.get("valuation.pe_cheap") == generic.get("valuation.pe_cheap")

    def test_sector_overrides_still_take_precedence(self):
        """User overrides beat sector adjustments."""
        tech = ThresholdManager(sector="Technology")
        tech.override("valuation.pe_cheap", 1)
        assert tech.get("valuation.pe_cheap") == 1


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_returns_string(self):
        tm = ThresholdManager()
        s = tm.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_key_section_headers(self):
        tm = ThresholdManager()
        s = tm.summary()
        assert "GROWTH" in s
        assert "PROFITABILITY" in s
        assert "VALUATION" in s
        assert "BALANCE SHEET" in s

    def test_summary_reflects_sector(self):
        tm = ThresholdManager(sector="Technology")
        s = tm.summary()
        assert "Technology" in s

    def test_summary_shows_override_count(self):
        tm = ThresholdManager()
        tm.override("valuation.pe_cheap", 5)
        tm.override("valuation.pe_fair", 10)
        s = tm.summary()
        assert "2" in s  # 2 overrides listed


# ── _dotget (internal) ────────────────────────────────────────────────────────

class TestDotget:
    def test_simple_key(self):
        assert ThresholdManager._dotget({"a": 1}, "a") == 1

    def test_nested_key(self):
        d = {"outer": {"inner": 42}}
        assert ThresholdManager._dotget(d, "outer.inner") == 42

    def test_deeply_nested(self):
        d = {"a": {"b": {"c": "deep"}}}
        assert ThresholdManager._dotget(d, "a.b.c") == "deep"

    def test_missing_top_level(self):
        assert ThresholdManager._dotget({}, "missing", default="x") == "x"

    def test_missing_nested_level(self):
        d = {"outer": {}}
        assert ThresholdManager._dotget(d, "outer.inner", default=0) == 0

    def test_value_is_zero(self):
        """Returns 0, not the default, when the stored value is 0."""
        d = {"key": 0}
        assert ThresholdManager._dotget(d, "key", default=99) == 0
