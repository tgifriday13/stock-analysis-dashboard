"""PIT-safe normalization and panel-building utilities for SEC companyfacts."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CANONICAL_TAG_PRIORITY, FLOW_FIELDS
from .sources import SecEdgarSource

logger = logging.getLogger("dashboard.first_principles.transforms")


def _safe_date(value: Any) -> Optional[date]:
    if value in (None, "", pd.NaT):
        return None
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _expected_unit_for_field(field: str) -> str:
    if "share" in field:
        return "shares"
    if field in {"current_ratio"}:
        return "pure"
    return "usd"


def _unit_rank(field: str, unit: str) -> int:
    u = (unit or "").lower()
    expected = _expected_unit_for_field(field)
    if expected == "shares":
        if "shares" in u:
            return 0
        return 5
    if expected == "pure":
        if "pure" in u:
            return 0
        return 5
    # USD expected
    if "usd" in u:
        return 0
    return 5


def normalize_companyfacts(
    cik: str,
    ticker: str,
    companyfacts_json: Dict[str, Any],
    filing_meta: Dict[str, Dict[str, Any]],
    tag_priority: Optional[Dict[str, Iterable[Dict[str, str]]]] = None,
) -> pd.DataFrame:
    """
    Convert SEC companyfacts to a canonical long-form DataFrame.

    Output columns are stable for both research-view and PIT-view assembly.
    """
    facts = companyfacts_json.get("facts", {})
    priorities = tag_priority or CANONICAL_TAG_PRIORITY

    rows = []
    for canonical_field, candidates in priorities.items():
        for tag_rank, candidate in enumerate(candidates):
            taxonomy = candidate["taxonomy"]
            tag = candidate["tag"]
            tag_payload = facts.get(taxonomy, {}).get(tag)
            if not tag_payload:
                continue

            units = tag_payload.get("units", {})
            ranked_units = sorted(units.keys(), key=lambda u: _unit_rank(canonical_field, u))

            for unit in ranked_units:
                unit_facts = units.get(unit, [])
                for entry in unit_facts:
                    period_end = _safe_date(entry.get("end"))
                    if period_end is None:
                        continue

                    try:
                        value = float(entry.get("val"))
                    except (TypeError, ValueError):
                        continue

                    accession_raw = entry.get("accn")
                    accession_norm = SecEdgarSource.normalize_accession(accession_raw)
                    meta = filing_meta.get(accession_norm, {}) if accession_norm else {}

                    rows.append(
                        {
                            "cik": str(cik).zfill(10),
                            "ticker": ticker.upper(),
                            "canonical_field": canonical_field,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "tag_rank": tag_rank,
                            "unit": unit,
                            "value": value,
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                            "form": entry.get("form"),
                            "period_start": _safe_date(entry.get("start")),
                            "period_end": period_end,
                            "filed_date": _safe_date(entry.get("filed")) or meta.get("filing_date"),
                            "accession": accession_raw,
                            "accession_norm": accession_norm,
                            "acceptance_datetime": meta.get("acceptance_datetime"),
                            "pit_available_date": meta.get("pit_available_date"),
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cik",
                "ticker",
                "canonical_field",
                "taxonomy",
                "tag",
                "tag_rank",
                "unit",
                "value",
                "fiscal_year",
                "fiscal_period",
                "form",
                "period_start",
                "period_end",
                "filed_date",
                "accession",
                "accession_norm",
                "acceptance_datetime",
                "pit_available_date",
            ]
        )

    df = pd.DataFrame(rows)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")
    df = df.dropna(subset=["fiscal_year"]).copy()
    df["fiscal_year"] = df["fiscal_year"].astype(int)

    # Drop obviously unusable units; keep shares/usd/pure focused observations.
    unit_mask = (
        (df["canonical_field"].str.contains("share") & df["unit"].str.lower().str.contains("shares"))
        | (~df["canonical_field"].str.contains("share") & df["unit"].str.lower().str.contains("usd|pure", regex=True))
    )
    df = df[unit_mask].copy()
    return df


def _dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_cols = ["canonical_field", "fiscal_year", "fiscal_period", "period_end"]
    tmp = df.copy()
    tmp["filed_sort"] = pd.to_datetime(tmp["filed_date"], errors="coerce")
    tmp["pit_sort"] = pd.to_datetime(tmp["pit_available_date"], errors="coerce")

    tmp = tmp.sort_values(
        sort_cols + ["filed_sort", "pit_sort", "tag_rank"],
        ascending=[True, True, True, True, False, False, True],
    )
    deduped = tmp.drop_duplicates(sort_cols, keep="first").drop(columns=["filed_sort", "pit_sort"])
    return deduped


def build_research_view(df_facts: pd.DataFrame) -> pd.DataFrame:
    """Research view = latest fact per period regardless of current as-of date."""
    return _dedupe_latest(df_facts)


def build_pit_view(df_facts: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
    """Point-in-time view honoring acceptance_date + 1 business day cutoff."""
    if df_facts.empty:
        return df_facts
    cutoff = pd.Timestamp(as_of_date).date()
    tmp = df_facts.copy()
    tmp = tmp[tmp["pit_available_date"].notna()].copy()
    tmp = tmp[tmp["pit_available_date"] <= cutoff]
    return _dedupe_latest(tmp)


def _latest_by_fy_fp(df: pd.DataFrame, field: str, fp: str) -> pd.Series:
    part = df[(df["canonical_field"] == field) & (df["fiscal_period"] == fp)].copy()
    if part.empty:
        return pd.Series(dtype=float)
    part["filed_sort"] = pd.to_datetime(part["filed_date"], errors="coerce")
    part = part.sort_values(["fiscal_year", "filed_sort", "tag_rank"], ascending=[True, False, True])
    part = part.drop_duplicates(["fiscal_year"], keep="first")
    return pd.Series(part["value"].values, index=part["fiscal_year"].values, name=field)


def _compute_quarter_from_ytd(series_q1: pd.Series, series_q2_ytd: pd.Series, series_q3_ytd: pd.Series, series_fy: pd.Series) -> pd.DataFrame:
    fy_index = sorted(set(series_q1.index).union(series_q2_ytd.index).union(series_q3_ytd.index).union(series_fy.index))
    rows = []
    for fy in fy_index:
        q1 = series_q1.get(fy, np.nan)
        q2_ytd = series_q2_ytd.get(fy, np.nan)
        q3_ytd = series_q3_ytd.get(fy, np.nan)
        fy_total = series_fy.get(fy, np.nan)

        q2 = q2_ytd - q1 if pd.notna(q2_ytd) and pd.notna(q1) else np.nan
        q3 = q3_ytd - q2_ytd if pd.notna(q3_ytd) and pd.notna(q2_ytd) else np.nan
        q4 = fy_total - q3_ytd if pd.notna(fy_total) and pd.notna(q3_ytd) else np.nan

        rows.append({"fiscal_year": fy, "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "FY": fy_total})

    return pd.DataFrame(rows).set_index("fiscal_year") if rows else pd.DataFrame()


def build_annual_and_quarterly_panels(df_view: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build annual panel and per-field quarterly flow panels.

    Annual panel uses FY for flow fields and FY/Q4 snapshot for stock fields.
    """
    if df_view.empty:
        return pd.DataFrame(), {}

    annual = pd.DataFrame()
    quarterly: Dict[str, pd.DataFrame] = {}

    fields = sorted(df_view["canonical_field"].unique())
    for field in fields:
        if field in FLOW_FIELDS:
            s_q1 = _latest_by_fy_fp(df_view, field, "Q1")
            s_q2 = _latest_by_fy_fp(df_view, field, "Q2")
            s_q3 = _latest_by_fy_fp(df_view, field, "Q3")
            s_fy = _latest_by_fy_fp(df_view, field, "FY")
            qtbl = _compute_quarter_from_ytd(s_q1, s_q2, s_q3, s_fy)
            quarterly[field] = qtbl
            if not qtbl.empty:
                annual[field] = qtbl["FY"]
        else:
            s_fy = _latest_by_fy_fp(df_view, field, "FY")
            if s_fy.empty:
                # fallback: use Q4 if available
                s_fy = _latest_by_fy_fp(df_view, field, "Q4")
            if not s_fy.empty:
                annual[field] = s_fy

    annual = annual.sort_index()
    annual.index.name = "fiscal_year"
    return annual, quarterly


def build_ttm_flow_panel(quarterly: Dict[str, pd.DataFrame]) -> pd.Series:
    """Compute latest TTM values from quarterly flow tables."""
    out: Dict[str, float] = {}
    for field, table in quarterly.items():
        if table.empty:
            continue
        qcols = ["Q1", "Q2", "Q3", "Q4"]
        stacked = table[qcols].stack()
        if stacked.empty:
            continue

        # Use last 4 available quarterly observations across years.
        last_four = stacked.iloc[-4:]
        if len(last_four) == 4:
            out[field] = float(last_four.sum())
        else:
            out[field] = np.nan
    return pd.Series(out, name="ttm")


def freshness_snapshot(df_facts: pd.DataFrame) -> Dict[str, Any]:
    """Simple freshness metadata for report headers."""
    if df_facts.empty:
        return {
            "latest_filed_date": None,
            "latest_pit_available_date": None,
            "num_facts": 0,
            "num_fields": 0,
        }

    latest_filed = pd.to_datetime(df_facts["filed_date"], errors="coerce").max()
    latest_pit = pd.to_datetime(df_facts["pit_available_date"], errors="coerce").max()
    return {
        "latest_filed_date": latest_filed.date().isoformat() if pd.notna(latest_filed) else None,
        "latest_pit_available_date": latest_pit.date().isoformat() if pd.notna(latest_pit) else None,
        "num_facts": int(len(df_facts)),
        "num_fields": int(df_facts["canonical_field"].nunique()),
    }
