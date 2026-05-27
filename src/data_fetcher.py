"""
data_fetcher.py — Single source of truth for all raw market data.

Pulls from yfinance with automatic disk-based caching.
Cache is considered stale after MAX_CACHE_AGE_HOURS hours and refreshed automatically.
Every dataset is time-stamped; staleness relative to the latest filing is reported.
"""

import json
import os
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

from .utils import cache_dir, save_json, load_json, friendly_timestamp, safe_float

logger = logging.getLogger("dashboard.data_fetcher")

MAX_CACHE_AGE_HOURS = 24  # Refresh cache if older than this
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(ticker: str, data_type: str) -> Path:
    return cache_dir() / f"{ticker.upper()}_{data_type}.json"


def _is_cache_fresh(path: Path) -> bool:
    """Return True if the cache file exists and is younger than MAX_CACHE_AGE_HOURS."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=MAX_CACHE_AGE_HOURS)


def _serialize(obj: Any) -> Any:
    """Make an object JSON-serializable (handles DataFrames, Timestamps, NaN/Inf)."""
    if isinstance(obj, pd.DataFrame):
        # Convert index and columns to plain strings before calling to_dict
        # (to_dict does NOT accept date_format; we handle timestamps manually)
        df = obj.copy()
        df.index = [str(i) for i in df.index]
        df.columns = [str(c) for c in df.columns]
        return {str(k): _serialize(v) for k, v in df.to_dict(orient="index").items()}
    if isinstance(obj, pd.Series):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _save_cache(ticker: str, data_type: str, data: Any) -> None:
    path = _cache_path(ticker, data_type)
    os.makedirs(path.parent, exist_ok=True)
    payload = {
        "fetched_at": friendly_timestamp(),
        "ticker": ticker.upper(),
        "data_type": data_type,
        "data": _serialize(data),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.debug(f"Cached {data_type} for {ticker} → {path.name}")


def _load_cache(ticker: str, data_type: str) -> Optional[Dict]:
    path = _cache_path(ticker, data_type)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# ── Core fetcher ─────────────────────────────────────────────────────────────

def _fetch_with_retry(func, *args, **kwargs):
    """Call func(*args, **kwargs) up to RETRY_ATTEMPTS times."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying…")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"All {RETRY_ATTEMPTS} attempts failed: {e}")
                raise


class StockData:
    """
    Container for all raw data fetched for a single ticker.
    Provides clean, typed accessors used by metrics_calculator.py.
    """

    def __init__(self, ticker: str, force_refresh: bool = False):
        self.ticker = ticker.upper()
        self.force_refresh = force_refresh
        self.fetch_timestamp: str = friendly_timestamp()
        self._raw: Dict[str, Any] = {}
        self._yf: Optional[yf.Ticker] = None
        self._load_all()

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def info(self) -> Dict[str, Any]:
        return self._raw.get("info", {})

    @property
    def income_stmt(self) -> pd.DataFrame:
        return self._raw.get("income_stmt", pd.DataFrame())

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return self._raw.get("balance_sheet", pd.DataFrame())

    @property
    def cash_flow(self) -> pd.DataFrame:
        return self._raw.get("cash_flow", pd.DataFrame())

    @property
    def history(self) -> pd.DataFrame:
        """5-year daily price history."""
        return self._raw.get("history", pd.DataFrame())

    @property
    def dividends(self) -> pd.Series:
        return self._raw.get("dividends", pd.Series(dtype=float))

    @property
    def earnings_history(self) -> pd.DataFrame:
        return self._raw.get("earnings_history", pd.DataFrame())

    @property
    def recommendations(self) -> pd.DataFrame:
        return self._raw.get("recommendations", pd.DataFrame())

    def get_price_on_date(self, target_date) -> Optional[float]:
        """Return the closing price nearest to target_date."""
        hist = self.history
        if hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        target = pd.Timestamp(target_date)
        idx = hist.index.searchsorted(target)
        if idx >= len(hist):
            idx = len(hist) - 1
        return safe_float(hist["Close"].iloc[idx])

    def days_since_last_filing(self) -> Optional[int]:
        """Return days elapsed since the most recent quarterly filing."""
        income = self.income_stmt
        if income.empty:
            return None
        latest_col = income.columns[0] if not income.empty else None
        if latest_col is None:
            return None
        try:
            filing_date = pd.Timestamp(latest_col).date()
            return (datetime.today().date() - filing_date).days
        except Exception:
            return None

    def staleness_label(self) -> str:
        """Human-readable freshness label for display in visuals."""
        days = self.days_since_last_filing()
        if days is None:
            return f"Data as of {self.fetch_timestamp}"
        return f"Data as of {self.fetch_timestamp} — last filing {days} days ago"

    # ── Internal loader ───────────────────────────────────────────────────────

    def _load_all(self) -> None:
        logger.info(f"Loading data for {self.ticker}…")
        self._yf = yf.Ticker(self.ticker)

        datasets = {
            "info": self._fetch_info,
            "income_stmt": self._fetch_income,
            "balance_sheet": self._fetch_balance,
            "cash_flow": self._fetch_cashflow,
            "history": self._fetch_history,
            "dividends": self._fetch_dividends,
            "earnings_history": self._fetch_earnings_history,
            "recommendations": self._fetch_recommendations,
        }

        for name, fetcher in datasets.items():
            cache_key = name
            path = _cache_path(self.ticker, cache_key)

            if not self.force_refresh and _is_cache_fresh(path):
                cached = _load_cache(self.ticker, cache_key)
                if cached and "data" in cached:
                    self._raw[name] = self._deserialize(name, cached["data"])
                    logger.debug(f"  {name}: loaded from cache")
                    continue

            try:
                result = _fetch_with_retry(fetcher)
                self._raw[name] = result
                logger.debug(f"  {name}: fetched fresh")
            except Exception as e:
                logger.warning(f"  {name}: fetch failed ({e}), using empty fallback")
                self._raw[name] = self._empty_fallback(name)
                continue
            # Cache save is separate — a serialization error must NOT overwrite real data
            try:
                _save_cache(self.ticker, cache_key, result)
            except Exception as e:
                logger.debug(f"  {name}: cache save failed ({e})")

        self.fetch_timestamp = friendly_timestamp()
        logger.info(f"Data loaded for {self.ticker} — {self.staleness_label()}")

    def _fetch_info(self) -> Dict:
        info = self._yf.info or {}
        return {k: _serialize(v) for k, v in info.items()}

    def _fetch_income(self) -> pd.DataFrame:
        stmt = self._yf.financials  # annual
        if stmt is None or stmt.empty:
            stmt = self._yf.income_stmt
        return stmt if stmt is not None else pd.DataFrame()

    def _fetch_balance(self) -> pd.DataFrame:
        bs = self._yf.balance_sheet
        return bs if bs is not None else pd.DataFrame()

    def _fetch_cashflow(self) -> pd.DataFrame:
        cf = self._yf.cashflow
        return cf if cf is not None else pd.DataFrame()

    def _fetch_history(self) -> pd.DataFrame:
        hist = self._yf.history(period="5y", interval="1d", auto_adjust=True)
        return hist if hist is not None else pd.DataFrame()

    def _fetch_dividends(self) -> pd.Series:
        divs = self._yf.dividends
        return divs if divs is not None else pd.Series(dtype=float)

    def _fetch_earnings_history(self) -> pd.DataFrame:
        try:
            eh = self._yf.earnings_history
            return eh if eh is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _fetch_recommendations(self) -> pd.DataFrame:
        try:
            rec = self._yf.recommendations
            return rec if rec is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _deserialize(self, name: str, data: Any) -> Any:
        """Reconstruct DataFrames from cached JSON.

        Serialized with orient="index":
          - Financial stmts: outer keys = metric names (rows), inner keys = date strings (cols)
            → reconstruct as DF, then convert columns to DatetimeIndex
          - History:         outer keys = date strings  (rows), inner keys = OHLCV col names
            → reconstruct as DF, then convert index to DatetimeIndex
        """
        if name in ("income_stmt", "balance_sheet", "cash_flow", "earnings_history", "recommendations"):
            if isinstance(data, dict):
                try:
                    # outer keys = metric names → rows; inner keys = date strings → columns
                    df = pd.DataFrame.from_dict(data, orient="index")
                    df.columns = pd.to_datetime(df.columns, errors="coerce")
                    # Values from JSON are already numeric; just ensure float dtype where possible
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
                except Exception as e:
                    logger.debug(f"_deserialize({name}) failed: {e}")
                    return pd.DataFrame()
            return pd.DataFrame()
        elif name == "history":
            if isinstance(data, dict):
                try:
                    # outer keys = date strings → index; inner keys = OHLCV → columns
                    df = pd.DataFrame.from_dict(data, orient="index")
                    df.index = pd.to_datetime(df.index, errors="coerce")
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
                except Exception as e:
                    logger.debug(f"_deserialize({name}) failed: {e}")
                    return pd.DataFrame()
            return pd.DataFrame()
        elif name == "dividends":
            if isinstance(data, dict):
                try:
                    s = pd.Series(data).astype(float)
                    # Dividend dates include mixed timezone offsets (EDT/EST); utc=True normalises
                    s.index = pd.to_datetime(s.index, errors="coerce", utc=True)
                    return s
                except Exception as e:
                    logger.debug(f"_deserialize(dividends) failed: {e}")
                    return pd.Series(dtype=float)
            return pd.Series(dtype=float)
        elif name == "info":
            return data if isinstance(data, dict) else {}
        return data

    def _empty_fallback(self, name: str) -> Any:
        if name in ("income_stmt", "balance_sheet", "cash_flow", "earnings_history", "recommendations"):
            return pd.DataFrame()
        elif name == "history":
            return pd.DataFrame()
        elif name == "dividends":
            return pd.Series(dtype=float)
        elif name == "info":
            return {}
        return None


# ── Multi-ticker convenience ──────────────────────────────────────────────────

def fetch_multiple(tickers: List[str], force_refresh: bool = False) -> Dict[str, StockData]:
    """
    Fetch data for a list of tickers.
    Returns dict mapping ticker → StockData.
    """
    results = {}
    for t in tickers:
        try:
            results[t.upper()] = StockData(t, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Failed to fetch data for {t}: {e}")
    return results
