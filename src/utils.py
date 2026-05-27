"""
utils.py — Shared helper functions used across all modules.
"""

import os
import json
import logging
import pytz
from datetime import datetime, date
from typing import Any, Optional, Union
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard")


# ── Path helpers ─────────────────────────────────────────────────────────────
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def cache_dir() -> Path:
    return project_root() / "data" / "cache"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def config_dir() -> Path:
    return project_root() / "config"


# ── Timestamp helpers ─────────────────────────────────────────────────────────
def now_pdt() -> datetime:
    """Return current datetime in US/Pacific (PDT/PST)."""
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz)


def timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return now_pdt().strftime(fmt)


def friendly_timestamp(dt: Optional[datetime] = None) -> str:
    """Human-readable timestamp string: 'YYYY-MM-DD HH:MM PDT'."""
    if dt is None:
        dt = now_pdt()
    elif dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(pytz.timezone("US/Pacific"))
    return dt.strftime("%Y-%m-%d %H:%M %Z")


# ── Number formatting ────────────────────────────────────────────────────────
def fmt_currency(value: Optional[float], decimals: int = 2) -> str:
    """Format a number as a dollar amount: $1.23B, $456.7M, $12.3K, $1.23."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "N/A"
    value = float(value)
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val/1e12:.{decimals}f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.{decimals}f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.{decimals}f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val/1e3:.{decimals}f}K"
    else:
        return f"{sign}${abs_val:.{decimals}f}"


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a ratio (0–1) as a percentage string: '12.3%'."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    return f"{float(value)*100:.{decimals}f}%"


def fmt_multiple(value: Optional[float], decimals: int = 1, suffix: str = "x") -> str:
    """Format a valuation multiple: '15.3x'."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    return f"{float(value):.{decimals}f}{suffix}"


def fmt_number(value: Optional[float], decimals: int = 2) -> str:
    """General number formatter with abbreviations."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    value = float(value)
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}{abs_val/1e12:.{decimals}f}T"
    elif abs_val >= 1e9:
        return f"{sign}{abs_val/1e9:.{decimals}f}B"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val/1e6:.{decimals}f}M"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val/1e3:.{decimals}f}K"
    else:
        return f"{sign}{abs_val:.{decimals}f}"


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Return numerator / denominator, or None if denominator is zero/None."""
    try:
        if denominator is None or denominator == 0:
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        result = float(value)
        import math
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def safe_pct_change(new_val: Optional[float], old_val: Optional[float]) -> Optional[float]:
    """Return (new - old) / abs(old) or None if not computable."""
    n, o = safe_float(new_val), safe_float(old_val)
    if n is None or o is None or o == 0:
        return None
    return (n - o) / abs(o)


# ── Scoring helpers ──────────────────────────────────────────────────────────
def score_metric(
    value: Optional[float],
    strong: float,
    acceptable: float,
    higher_is_better: bool = True,
) -> int:
    """
    Return 2 (strong), 1 (acceptable), or 0 (weak) based on thresholds.
    Works for both 'higher is better' (e.g., margins) and 'lower is better'
    (e.g., P/E ratio, debt/EBITDA).
    """
    if value is None:
        return 0
    v = float(value)
    if higher_is_better:
        if v >= strong:
            return 2
        elif v >= acceptable:
            return 1
        else:
            return 0
    else:  # lower is better
        if v <= strong:
            return 2
        elif v <= acceptable:
            return 1
        else:
            return 0


def delta_arrow(value: Optional[float], higher_is_better: bool = True) -> str:
    """Return ▲, ▼, or — based on sign of delta."""
    if value is None:
        return "—"
    if value > 0:
        return "▲" if higher_is_better else "▼ (worse)"
    elif value < 0:
        return "▼" if higher_is_better else "▲ (better)"
    return "—"


def delta_color(value: Optional[float], higher_is_better: bool = True) -> str:
    """Return CSS color for a delta value."""
    if value is None:
        return "#888888"
    positive_color = "#27ae60"
    negative_color = "#e74c3c"
    neutral_color = "#888888"
    if value > 0.001:
        return positive_color if higher_is_better else negative_color
    elif value < -0.001:
        return negative_color if higher_is_better else positive_color
    return neutral_color


# ── JSON helpers ─────────────────────────────────────────────────────────────
def load_json(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: Union[str, Path]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Date helpers ─────────────────────────────────────────────────────────────
def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse common date formats into a date object."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(str(date_str), fmt).date()
        except ValueError:
            continue
    return None


def days_since(d: Optional[date]) -> Optional[int]:
    if d is None:
        return None
    return (date.today() - d).days


# ── Session logging ──────────────────────────────────────────────────────────
def log_session(ticker: str, mode: str, recommendation: str, notes: str = "") -> None:
    """Append a row to data/session_log.csv for audit trail."""
    import csv
    log_path = project_root() / "data" / "session_log.csv"
    os.makedirs(log_path.parent, exist_ok=True)
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticker", "mode", "recommendation", "notes"])
        writer.writerow([friendly_timestamp(), ticker, mode, recommendation, notes])
