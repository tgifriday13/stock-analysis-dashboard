"""Free-data source connectors for SEC, FRED, and price history."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ..utils import project_root
from .config import (
    FRED_SERIES_URL,
    SEC_COMPANYFACTS_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKER_MAP_URL,
    STOOQ_PRICE_URL,
)

logger = logging.getLogger("dashboard.first_principles.sources")


_CACHE_ROOT = project_root() / "data" / "first_principles" / "cache"
FRENCH_FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"


def _next_business_day(d: date, n: int = 1) -> date:
    """Simple weekday-only business day increment."""
    out = d
    step = 0
    while step < n:
        out += timedelta(days=1)
        if out.weekday() < 5:
            step += 1
    return out


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _parse_acceptance_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


@dataclass
class CachedHttpClient:
    """HTTP client with disk cache and TTL."""

    cache_dir: Path
    force_refresh: bool = False
    timeout_seconds: int = 25

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_cached_payload(cache_path: Path) -> Dict[str, Any]:
        with open(cache_path, "r") as f:
            payload = json.load(f)
        # Backward-compatible with any legacy cache entries that might store raw JSON.
        return payload["data"] if isinstance(payload, dict) and "data" in payload else payload

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        ttl_hours: int = 24,
    ) -> Dict[str, Any]:
        params = params or {}
        cache_key = _hash_key(f"{url}|{json.dumps(params, sort_keys=True)}")
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists() and not self.force_refresh:
            age = datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)
            if age.total_seconds() <= ttl_hours * 3600:
                return self._read_cached_payload(cache_path)

        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            with open(cache_path, "w") as f:
                json.dump({"fetched_at_utc": datetime.utcnow().isoformat(), "data": payload}, f)
            return payload
        except Exception as exc:
            if cache_path.exists() and not self.force_refresh:
                logger.warning(
                    "HTTP fetch failed for %s (%s); using stale cache %s",
                    url,
                    exc,
                    cache_path,
                )
                return self._read_cached_payload(cache_path)
            raise


class SecEdgarSource:
    """Connector for SEC ticker map, submissions, and companyfacts."""

    def __init__(self, force_refresh: bool = False, user_agent: Optional[str] = None):
        self.user_agent = user_agent or os.getenv(
            "SEC_USER_AGENT",
            "first-principles-research/1.0 (email: local@example.com)",
        )
        self.headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        self.client = CachedHttpClient(_CACHE_ROOT / "sec", force_refresh=force_refresh)

    @staticmethod
    def pad_cik(cik: str) -> str:
        digits = "".join(ch for ch in str(cik) if ch.isdigit())
        return digits.zfill(10)

    @staticmethod
    def normalize_accession(accession: Optional[str]) -> Optional[str]:
        if accession is None:
            return None
        return str(accession).replace("-", "").strip()

    def fetch_ticker_map(self) -> pd.DataFrame:
        raw = self.client.get_json(SEC_TICKER_MAP_URL, headers=self.headers, ttl_hours=24)
        fields = raw.get("fields", [])
        data = raw.get("data", [])
        if fields and data:
            df = pd.DataFrame(data, columns=fields)
        else:
            # Fall back to company_tickers shape if SEC payload changes
            df = pd.DataFrame(raw).T
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.upper()
        if "cik" in df.columns:
            df["cik"] = df["cik"].astype(str).str.zfill(10)
        elif "cik_str" in df.columns:
            df["cik"] = df["cik_str"].astype(str).str.zfill(10)
        return df

    def resolve_ticker(self, ticker: str) -> Dict[str, Any]:
        t = ticker.upper().strip()
        mapping = self.fetch_ticker_map()
        row = mapping[mapping["ticker"] == t]
        if row.empty:
            raise ValueError(f"Ticker '{ticker}' not found in SEC ticker map.")
        rec = row.iloc[0].to_dict()
        rec["ticker"] = t
        rec["cik"] = self.pad_cik(str(rec.get("cik", rec.get("cik_str", ""))))
        return rec

    def fetch_submissions(self, cik: str) -> Dict[str, Any]:
        url = SEC_SUBMISSIONS_URL.format(cik=self.pad_cik(cik))
        data = self.client.get_json(url, headers=self.headers, ttl_hours=8)
        time.sleep(0.12)
        return data

    def fetch_companyfacts(self, cik: str) -> Dict[str, Any]:
        url = SEC_COMPANYFACTS_URL.format(cik=self.pad_cik(cik))
        data = self.client.get_json(url, headers=self.headers, ttl_hours=8)
        time.sleep(0.12)
        return data

    def filing_metadata_map(self, submissions_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Return accession -> filing metadata with PIT availability date."""
        filings = submissions_json.get("filings", {}).get("recent", {})
        accessions = filings.get("accessionNumber", [])
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        acceptance_times = filings.get("acceptanceDateTime", [])

        metadata: Dict[str, Dict[str, Any]] = {}
        length = min(len(accessions), len(forms), len(filing_dates))
        for i in range(length):
            accession_raw = accessions[i]
            accession = self.normalize_accession(accession_raw)
            filing_date = _parse_date(filing_dates[i])
            accepted = _parse_acceptance_datetime(acceptance_times[i] if i < len(acceptance_times) else None)
            accepted_date = accepted.date() if accepted else filing_date
            pit_date = _next_business_day(accepted_date, 1) if accepted_date else None

            metadata[accession] = {
                "accession": accession_raw,
                "form": forms[i],
                "filing_date": filing_date,
                "acceptance_datetime": accepted,
                "pit_available_date": pit_date,
            }
        return metadata


class FredSource:
    """Connector for FRED macro series."""

    def __init__(self, api_key: Optional[str] = None, force_refresh: bool = False):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.client = CachedHttpClient(_CACHE_ROOT / "fred", force_refresh=force_refresh)

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        params: Dict[str, Any] = {
            "series_id": series_id,
            "file_type": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if start_date is not None:
            params["observation_start"] = start_date.isoformat()
        if end_date is not None:
            params["observation_end"] = end_date.isoformat()

        payload = self.client.get_json(FRED_SERIES_URL, params=params, ttl_hours=24)
        observations = payload.get("observations", [])
        if not observations:
            return pd.Series(dtype=float)

        rows = []
        for obs in observations:
            d = _parse_date(obs.get("date"))
            v = obs.get("value")
            if d is None or v in (None, "."):
                continue
            try:
                rows.append((d, float(v)))
            except ValueError:
                continue

        if not rows:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([r[0] for r in rows])
        vals = [r[1] for r in rows]
        return pd.Series(vals, index=idx, name=series_id).sort_index()


class PriceSource:
    """Free daily-close price source with yfinance primary and Stooq fallback."""

    def __init__(self, force_refresh: bool = False):
        self.client = CachedHttpClient(_CACHE_ROOT / "price", force_refresh=force_refresh)

    def _read_history_cache(self, cache_path: Path) -> pd.DataFrame:
        try:
            if cache_path.suffix == ".parquet":
                return pd.read_parquet(cache_path).sort_index()
            if cache_path.suffix == ".csv":
                return pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
        except Exception as exc:
            logger.warning("Failed reading price cache %s: %s", cache_path, exc)
        return pd.DataFrame()

    def _fetch_stooq(self, ticker: str) -> pd.DataFrame:
        params = {"s": f"{ticker.lower()}.us", "i": "d"}
        cache_key = _hash_key(f"stooq|{ticker.upper()}")
        csv_path = _CACHE_ROOT / "price" / f"{cache_key}.csv"

        if csv_path.exists() and not self.client.force_refresh:
            age = datetime.utcnow() - datetime.utcfromtimestamp(csv_path.stat().st_mtime)
            if age.total_seconds() <= 24 * 3600:
                return pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")

        response = requests.get(STOOQ_PRICE_URL, params=params, timeout=20)
        response.raise_for_status()
        text = response.text.strip()
        if "No data" in text or not text:
            return pd.DataFrame()
        csv_path.write_text(text)
        df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
        return df.sort_index()

    def fetch_history(self, ticker: str, period: str = "10y") -> pd.DataFrame:
        cache_key = _hash_key(f"yf|{ticker.upper()}|{period}")
        parquet_path = _CACHE_ROOT / "price" / f"{cache_key}.parquet"
        csv_path = _CACHE_ROOT / "price" / f"{cache_key}.csv"
        if not self.client.force_refresh:
            for cache_path in (parquet_path, csv_path):
                if cache_path.exists():
                    age = datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)
                    if age.total_seconds() <= 24 * 3600:
                        cached = self._read_history_cache(cache_path)
                        if not cached.empty:
                            return cached

        df = pd.DataFrame()
        try:
            hist = yf.Ticker(ticker.upper()).history(period=period, auto_adjust=False)
            if not hist.empty:
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as exc:
            logger.warning("yfinance history fetch failed for %s: %s", ticker, exc)

        if df.empty:
            try:
                df = self._fetch_stooq(ticker)
                if not df.empty and "Close" not in df.columns and "close" in df.columns:
                    df = df.rename(columns={"close": "Close"})
            except Exception as exc:
                logger.warning("Stooq fallback failed for %s: %s", ticker, exc)
                return pd.DataFrame()

        if not df.empty:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_parquet(parquet_path)
            except Exception as exc:
                logger.warning(
                    "Parquet price cache unavailable for %s (%s). Writing CSV cache fallback.",
                    ticker,
                    exc,
                )
                df.to_csv(csv_path, index_label="Date")
        return df

    def latest_price(self, ticker: str) -> Optional[float]:
        hist = self.fetch_history(ticker, period="6mo")
        if hist.empty or "Close" not in hist.columns:
            return None
        val = hist["Close"].dropna()
        if val.empty:
            return None
        return float(val.iloc[-1])


class FrenchDataSource:
    """Connector for Kenneth French factor data library (free/public)."""

    def __init__(self, force_refresh: bool = False):
        self.cache_dir = _CACHE_ROOT / "french"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_refresh = force_refresh

    def _read_ff5_cache(self, cache_path: Path) -> pd.DataFrame:
        try:
            if cache_path.suffix == ".parquet":
                return pd.read_parquet(cache_path).sort_index()
            if cache_path.suffix == ".csv":
                return pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
        except Exception as exc:
            logger.warning("Failed reading FF5 cache %s: %s", cache_path, exc)
        return pd.DataFrame()

    def fetch_ff5_monthly(self) -> pd.DataFrame:
        """
        Fetch and parse Fama-French 5-factor monthly data.

        Returns columns including Mkt-RF, SMB, HML, RMW, CMA, RF in decimal units.
        """
        parquet_path = self.cache_dir / "ff5_monthly.parquet"
        csv_path = self.cache_dir / "ff5_monthly.csv"
        if not self.force_refresh:
            for cache_path in (parquet_path, csv_path):
                if cache_path.exists():
                    age = datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)
                    if age.total_seconds() <= 7 * 24 * 3600:
                        cached = self._read_ff5_cache(cache_path)
                        if not cached.empty:
                            return cached

        response = requests.get(FRENCH_FF5_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Archive contains one CSV file.
            name = zf.namelist()[0]
            raw = zf.read(name).decode("utf-8", errors="ignore")

        lines = raw.splitlines()
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith(",Mkt-RF") or line.strip().startswith("Date,"):
                start_idx = i
                continue
            if start_idx is not None and line.strip() == "":
                end_idx = i
                break

        if start_idx is None:
            raise ValueError("Could not locate FF5 header in French data file.")
        end_idx = end_idx or len(lines)
        csv_payload = "\n".join(lines[start_idx:end_idx])
        df = pd.read_csv(io.StringIO(csv_payload))

        # Normalize date column and cast to decimal returns.
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
        df = df.set_index("Date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
        df = df.dropna(how="all").sort_index()

        try:
            df.to_parquet(parquet_path)
        except Exception as exc:
            logger.warning("Parquet FF5 cache unavailable (%s). Writing CSV cache fallback.", exc)
            df.to_csv(csv_path, index_label="Date")
        return df
