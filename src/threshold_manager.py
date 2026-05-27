"""
threshold_manager.py — Manages smart, context-aware thresholds.

Provides sector-adjusted defaults and allows users to override any threshold.
User threshold sets can be saved/loaded by name for reuse across sessions.
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import config_dir, project_root, load_json, save_json

logger = logging.getLogger("dashboard.thresholds")

DEFAULT_THRESHOLDS_FILE = config_dir() / "default_thresholds.json"
USER_THRESHOLDS_DIR = project_root() / "config" / "user_thresholds"


def _load_defaults() -> Dict:
    """Load the default thresholds JSON from config/."""
    try:
        return load_json(DEFAULT_THRESHOLDS_FILE)
    except FileNotFoundError:
        logger.error(f"Default thresholds not found at {DEFAULT_THRESHOLDS_FILE}")
        return {}


class ThresholdManager:
    """
    Provides context-aware threshold values used by scoring and visualization.

    Usage:
        tm = ThresholdManager(sector="Technology")
        pe_cheap = tm.get("valuation.pe_cheap")   # → 24 (sector-adjusted)
        tm.override("valuation.pe_cheap", 20)      # user override
        tm.save("my_thresholds")                   # persist to file
        tm2 = ThresholdManager.load("my_thresholds")
    """

    def __init__(self, sector: Optional[str] = None, overrides: Optional[Dict] = None):
        self._base = _load_defaults()
        self._sector = sector
        self._overrides: Dict[str, Any] = overrides or {}
        self._effective = self._build_effective()

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve an effective threshold by dot-separated key.
        e.g. tm.get("valuation.pe_cheap")
        Precedence: user overrides → sector-adjusted base → raw base → default
        """
        if key in self._overrides:
            return self._overrides[key]
        return self._dotget(self._effective, key, default)

    def override(self, key: str, value: Any) -> None:
        """Set a user override for a specific threshold key."""
        self._overrides[key] = value
        logger.debug(f"Threshold override: {key} = {value}")

    def override_many(self, overrides: Dict[str, Any]) -> None:
        """Apply multiple overrides at once."""
        for k, v in overrides.items():
            self.override(k, v)

    def reset_overrides(self) -> None:
        """Clear all user overrides, reverting to sector-adjusted defaults."""
        self._overrides.clear()

    def save(self, name: str) -> Path:
        """Persist current overrides + sector to a named file under config/user_thresholds/."""
        os.makedirs(USER_THRESHOLDS_DIR, exist_ok=True)
        path = USER_THRESHOLDS_DIR / f"{name}.json"
        save_json({"sector": self._sector, "overrides": self._overrides}, path)
        logger.info(f"Threshold set '{name}' saved to {path}")
        return path

    @classmethod
    def load(cls, name: str) -> "ThresholdManager":
        """Load a previously saved threshold set by name."""
        path = USER_THRESHOLDS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No threshold set named '{name}' found at {path}")
        data = load_json(path)
        return cls(sector=data.get("sector"), overrides=data.get("overrides", {}))

    @classmethod
    def list_saved(cls) -> list:
        """Return list of names of saved threshold sets."""
        if not USER_THRESHOLDS_DIR.exists():
            return []
        return [p.stem for p in USER_THRESHOLDS_DIR.glob("*.json")]

    def summary(self) -> str:
        """Print a readable summary of key effective thresholds."""
        lines = [
            f"Thresholds — sector: {self._sector or 'Generic'}, "
            f"user overrides: {len(self._overrides)}",
            "",
            "  GROWTH",
            f"    Revenue CAGR (strong):    {self.get('growth.revenue_cagr_1yr_strong')*100:.0f}%",
            f"    Revenue CAGR (acceptable):{self.get('growth.revenue_cagr_1yr_acceptable')*100:.0f}%",
            "",
            "  PROFITABILITY",
            f"    Operating margin (strong):    {self.get('profitability.operating_margin_strong')*100:.0f}%",
            f"    Operating margin (acceptable):{self.get('profitability.operating_margin_acceptable')*100:.0f}%",
            f"    ROIC (strong):                {self.get('profitability.roic_strong')*100:.0f}%",
            "",
            "  CASH FLOW",
            f"    FCF yield (strong):     {self.get('cashflow.fcf_yield_strong')*100:.0f}%",
            f"    FCF margin (strong):    {self.get('cashflow.fcf_margin_strong')*100:.0f}%",
            "",
            "  BALANCE SHEET",
            f"    Debt/EBITDA (strong):   {self.get('balance_sheet.debt_to_ebitda_strong')}x",
            f"    Debt/EBITDA (danger):   {self.get('balance_sheet.debt_to_ebitda_danger')}x",
            f"    Current ratio (strong): {self.get('balance_sheet.current_ratio_strong')}",
            "",
            "  VALUATION",
            f"    P/E cheap: {self.get('valuation.pe_cheap')}, "
            f"fair: {self.get('valuation.pe_fair')}, "
            f"expensive: {self.get('valuation.pe_expensive')}",
            f"    EV/EBITDA cheap: {self.get('valuation.ev_ebitda_cheap')}, "
            f"fair: {self.get('valuation.ev_ebitda_fair')}, "
            f"expensive: {self.get('valuation.ev_ebitda_expensive')}",
        ]
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_effective(self) -> Dict:
        """Build effective thresholds = base defaults + sector adjustments."""
        effective = copy.deepcopy(self._base)
        if not self._sector:
            return effective

        adj = self._base.get("sector_adjustments", {}).get(self._sector, {})
        if not adj:
            return effective

        # Apply sector multipliers & boosts
        val = effective.get("valuation", {})
        prof = effective.get("profitability", {})
        div = effective.get("dividends", {})
        bs = effective.get("balance_sheet", {})

        if "pe_multiplier" in adj:
            m = adj["pe_multiplier"]
            for k in ["pe_cheap", "pe_fair", "pe_expensive"]:
                if k in val:
                    val[k] = round(val[k] * m, 1)

        if "ps_multiplier" in adj:
            m = adj["ps_multiplier"]
            for k in ["ps_cheap", "ps_fair", "ps_expensive"]:
                if k in val:
                    val[k] = round(val[k] * m, 1)

        if "gross_margin_boost" in adj:
            b = adj["gross_margin_boost"]
            for k in ["gross_margin_strong", "gross_margin_acceptable"]:
                if k in prof:
                    prof[k] = round(prof[k] + b, 3)

        if "roe_boost" in adj:
            b = adj["roe_boost"]
            for k in ["roe_strong", "roe_acceptable"]:
                if k in prof:
                    prof[k] = round(prof[k] + b, 3)

        if "debt_to_ebitda_multiplier" in adj:
            m = adj["debt_to_ebitda_multiplier"]
            for k in ["debt_to_ebitda_strong", "debt_to_ebitda_acceptable", "debt_to_ebitda_danger"]:
                if k in bs:
                    bs[k] = round(bs[k] * m, 1)

        if "dividend_yield_boost" in adj:
            b = adj["dividend_yield_boost"]
            for k in ["yield_strong", "yield_acceptable"]:
                if k in div:
                    div[k] = round(div[k] + b, 3)

        if "capex_intensity_high" in adj:
            effective["cashflow"]["capex_intensity_high"] = adj["capex_intensity_high"]

        effective["valuation"] = val
        effective["profitability"] = prof
        effective["dividends"] = div
        effective["balance_sheet"] = bs
        return effective

    @staticmethod
    def _dotget(d: Dict, key: str, default: Any = None) -> Any:
        """Navigate nested dict with dot-separated key."""
        parts = key.split(".")
        current = d
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
