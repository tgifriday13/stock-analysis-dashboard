"""Question evaluation engine for 12 first-principles past/future signals."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_PERCENTILE_WINSOR_LIMIT, QuestionSpec
from .schemas import ChartPayload, ChartReferenceLine, ChartSeries, PanelCoverage, QuestionResult

logger = logging.getLogger("dashboard.first_principles.questions")


def _safe_div(n: Any, d: Any) -> float:
    try:
        nf = float(n)
        df = float(d)
    except (TypeError, ValueError):
        return float("nan")
    if df == 0 or not np.isfinite(df):
        return float("nan")
    out = nf / df
    return float(out) if np.isfinite(out) else float("nan")


def _cagr(first: float, last: float, years: int) -> float:
    if years <= 0 or first <= 0 or last <= 0:
        return float("nan")
    return (last / first) ** (1 / years) - 1


def _last_n_valid(series: pd.Series, n: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[-n:] if len(s) >= n else s


def _winsorized_percentile_rank(value: float, population: Sequence[float], lower_q: float = DEFAULT_PERCENTILE_WINSOR_LIMIT) -> float:
    pop = np.array([x for x in population if np.isfinite(x)], dtype=float)
    if pop.size == 0 or not np.isfinite(value):
        return float("nan")
    lo = np.quantile(pop, lower_q)
    hi = np.quantile(pop, 1 - lower_q)
    clipped = np.clip(pop, lo, hi)
    v = float(np.clip(value, lo, hi))
    return float((clipped <= v).mean())


def _coverage(required_fields: Sequence[str], available_values: Dict[str, Any]) -> PanelCoverage:
    required = list(required_fields)
    available: List[str] = []
    missing: List[str] = []
    for field in required:
        val = available_values.get(field)
        ok = val is not None
        if isinstance(val, (float, np.floating)):
            ok = np.isfinite(val)
        if isinstance(val, pd.Series):
            ok = val.dropna().shape[0] > 0
        if ok:
            available.append(field)
        else:
            missing.append(field)

    ratio = (len(available) / len(required)) if required else 1.0
    return PanelCoverage(
        required_fields=required,
        available_fields=available,
        missing_fields=missing,
        coverage_ratio=float(ratio),
    )


def _signal_from_threshold(value: float, strong: float, watch: float, higher_is_better: bool = True) -> str:
    if not np.isfinite(value):
        return "Insufficient"
    if higher_is_better:
        if value >= strong:
            return "Strong"
        if value >= watch:
            return "Watch"
        return "Weak"
    if value <= strong:
        return "Strong"
    if value <= watch:
        return "Watch"
    return "Weak"


def _apply_percentile_overlay(signal: str, percentile: float, higher_is_better: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Hybrid absolute+percentile overlay for drift-resistant signals."""
    if signal == "Insufficient" or not np.isfinite(percentile):
        return signal, {"percentile": percentile, "overlay_applied": False}

    adjusted = signal
    if higher_is_better:
        if percentile < 0.20:
            adjusted = "Weak"
        elif percentile < 0.35 and signal == "Strong":
            adjusted = "Watch"
        elif percentile > 0.70 and signal == "Watch":
            adjusted = "Strong"
    else:
        # lower-is-better: invert logic
        inv = 1 - percentile
        if inv < 0.20:
            adjusted = "Weak"
        elif inv < 0.35 and signal == "Strong":
            adjusted = "Watch"
        elif inv > 0.70 and signal == "Watch":
            adjusted = "Strong"

    return adjusted, {"percentile": float(percentile), "overlay_applied": adjusted != signal, "base_signal": signal}


def _label_index_value(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _chart_series_from_series(name: str, series: pd.Series) -> ChartSeries:
    if series is None or len(series) == 0:
        return ChartSeries(name=name, x=[], y=[])
    s = pd.to_numeric(series, errors="coerce")
    mask = np.isfinite(s.values.astype(float))
    x = [_label_index_value(i) for i, keep in zip(s.index, mask) if keep]
    y = [float(v) for v, keep in zip(s.values, mask) if keep]
    return ChartSeries(name=name, x=x, y=y)


def _chart_series_from_points(name: str, x_vals: Sequence[Any], y_vals: Sequence[Any]) -> ChartSeries:
    xs: List[str] = []
    ys: List[float] = []
    for x, y in zip(x_vals, y_vals):
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yf):
            continue
        xs.append(_label_index_value(x))
        ys.append(yf)
    return ChartSeries(name=name, x=xs, y=ys)


def _has_chart_data(payload: Optional[ChartPayload]) -> bool:
    if payload is None:
        return False
    return any(len(s.x) > 0 and len(s.y) > 0 for s in payload.series)


class FirstPrinciplesQuestionEngine:
    """Evaluates the 12 first-principles questions from PIT-safe panels."""

    def __init__(self, specs: List[QuestionSpec]):
        self.specs = specs

    def evaluate(
        self,
        annual_pit: pd.DataFrame,
        annual_research: pd.DataFrame,
        quarterly_pit: Dict[str, pd.DataFrame],
        macro: Dict[str, pd.Series],
        latest_price: Optional[float],
    ) -> List[QuestionResult]:
        if annual_pit.empty:
            return [
                QuestionResult(
                    question_id=s.question_id,
                    question=s.question,
                    metric_family=s.metric_family,
                    signal="Insufficient",
                    falsified=False,
                    falsification_reason=None,
                    decision_reason="No PIT annual panel available.",
                    thresholds=s.thresholds,
                    metrics={},
                    percentile_overlay={},
                    coverage=PanelCoverage(s.required_fields, [], s.required_fields, 0.0),
                    audit={"note": "No PIT facts after acceptance-date cutoff."},
                )
                for s in self.specs
            ]

        ctx = self._build_context(annual_pit, annual_research, quarterly_pit, macro, latest_price)
        out = []
        for spec in self.specs:
            fn = getattr(self, f"_eval_{spec.question_id.lower()}")
            out.append(fn(spec, ctx))
        return out

    def _build_context(
        self,
        annual_pit: pd.DataFrame,
        annual_research: pd.DataFrame,
        quarterly_pit: Dict[str, pd.DataFrame],
        macro: Dict[str, pd.Series],
        latest_price: Optional[float],
    ) -> Dict[str, Any]:
        panel = annual_pit.copy().sort_index()
        hist = annual_research.copy().sort_index()

        revenue = panel.get("revenue", pd.Series(dtype=float))
        cfo = panel.get("operating_cash_flow", pd.Series(dtype=float))
        capex = panel.get("capex", pd.Series(dtype=float)).abs()
        net_income = panel.get("net_income", pd.Series(dtype=float))
        shares = panel.get("diluted_shares", pd.Series(dtype=float))

        owner_earnings = cfo - capex
        fcf = owner_earnings.copy()

        owner_earnings_ps = owner_earnings / shares
        fcf_ps = fcf / shares
        revenue_ps = revenue / shares

        cpi = macro.get("cpi", pd.Series(dtype=float))

        real_owner_ps = self._deflate_by_cpi(owner_earnings_ps, cpi)
        real_fcf_ps = self._deflate_by_cpi(fcf_ps, cpi)
        real_revenue_ps = self._deflate_by_cpi(revenue_ps, cpi)

        operating_income = panel.get("operating_income", pd.Series(dtype=float))
        tax_expense = panel.get("income_tax_expense", pd.Series(dtype=float))
        debt = panel.get("total_debt", pd.Series(dtype=float))
        cash = panel.get("cash_and_equivalents", pd.Series(dtype=float))
        equity = panel.get("total_equity", pd.Series(dtype=float))

        tax_rate = (tax_expense / operating_income.replace(0, np.nan)).clip(lower=0.0, upper=0.5).fillna(0.21)
        nopat = operating_income * (1 - tax_rate)
        invested_capital = debt + equity - cash
        incremental_roic = nopat.diff() / invested_capital.diff().replace(0, np.nan)

        net_debt = debt - cash

        current_assets = panel.get("current_assets", pd.Series(dtype=float))
        current_liabilities = panel.get("current_liabilities", pd.Series(dtype=float))

        working_capital = current_assets - current_liabilities
        working_capital_accrual = working_capital.diff() / panel.get("total_assets", pd.Series(dtype=float)).replace(0, np.nan)

        current_ratio = current_assets / current_liabilities.replace(0, np.nan)
        interest_expense = panel.get("interest_expense", pd.Series(dtype=float)).abs()
        interest_coverage = operating_income / interest_expense.replace(0, np.nan)

        dividends = panel.get("dividends_paid", pd.Series(dtype=float)).abs()
        repurchases = panel.get("repurchases", pd.Series(dtype=float)).abs()
        payout = dividends.add(repurchases, fill_value=0.0)

        latest_year = int(panel.index.max())

        market_cap = float("nan")
        if latest_price is not None and np.isfinite(latest_price):
            shares_latest = pd.to_numeric(shares, errors="coerce").dropna()
            if not shares_latest.empty:
                market_cap = float(latest_price * shares_latest.iloc[-1])

        ctx: Dict[str, Any] = {
            "annual": panel,
            "hist": hist,
            "quarterly": quarterly_pit,
            "macro": macro,
            "latest_price": latest_price,
            "latest_year": latest_year,
            "market_cap": market_cap,
            "revenue": revenue,
            "net_income": net_income,
            "cfo": cfo,
            "capex": capex,
            "fcf": fcf,
            "owner_earnings": owner_earnings,
            "shares": shares,
            "owner_earnings_ps": owner_earnings_ps,
            "fcf_ps": fcf_ps,
            "revenue_ps": revenue_ps,
            "real_owner_ps": real_owner_ps,
            "real_fcf_ps": real_fcf_ps,
            "real_revenue_ps": real_revenue_ps,
            "operating_income": operating_income,
            "tax_expense": tax_expense,
            "tax_rate": tax_rate,
            "nopat": nopat,
            "debt": debt,
            "cash": cash,
            "equity": equity,
            "net_debt": net_debt,
            "invested_capital": invested_capital,
            "incremental_roic": incremental_roic,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "working_capital_accrual": working_capital_accrual,
            "current_ratio": current_ratio,
            "interest_expense": interest_expense,
            "interest_coverage": interest_coverage,
            "dividends": dividends,
            "repurchases": repurchases,
            "payout": payout,
        }
        return ctx

    def _deflate_by_cpi(self, nominal: pd.Series, cpi_series: pd.Series) -> pd.Series:
        if nominal.empty or cpi_series.empty:
            return nominal.copy() * np.nan

        cpi_yearly = cpi_series.resample("YE").mean()
        cpi_by_year = pd.Series(cpi_yearly.values, index=cpi_yearly.index.year)
        base_year = int(max(cpi_by_year.index))
        base_cpi = float(cpi_by_year.loc[base_year])

        real = nominal.copy().astype(float)
        for year, val in nominal.items():
            cpi_val = cpi_by_year.get(int(year), np.nan)
            if pd.notna(cpi_val) and cpi_val > 0:
                real.loc[year] = float(val) * (base_cpi / float(cpi_val))
            else:
                real.loc[year] = np.nan
        return real

    def _basic_result(
        self,
        spec: QuestionSpec,
        metrics: Dict[str, Any],
        base_signal: str,
        percentile_series: Optional[pd.Series],
        percentile_value: Optional[float],
        coverage: PanelCoverage,
        falsified: bool,
        falsification_reason: Optional[str],
        decision_reason: str,
        takeaway: str,
        chart_payload: Optional[ChartPayload] = None,
        higher_is_better: bool = True,
    ) -> QuestionResult:
        signal = base_signal
        percentile_overlay: Dict[str, Any] = {}
        if percentile_series is not None and percentile_value is not None and np.isfinite(percentile_value):
            pct = _winsorized_percentile_rank(float(percentile_value), percentile_series.dropna().values)
            signal, percentile_overlay = _apply_percentile_overlay(signal, pct, higher_is_better=higher_is_better)
        else:
            percentile_overlay = {"percentile": float("nan"), "overlay_applied": False}

        if falsified:
            signal = "Weak"

        if coverage.coverage_ratio < spec.min_coverage:
            signal = "Insufficient"
            decision_reason = f"Coverage {coverage.coverage_ratio:.2f} below required {spec.min_coverage:.2f}."
            takeaway = "Insufficient evidence due to missing required point-in-time fields."

        if not _has_chart_data(chart_payload):
            chart_payload = None

        return QuestionResult(
            question_id=spec.question_id,
            question=spec.question,
            metric_family=spec.metric_family,
            signal=signal,
            falsified=falsified,
            falsification_reason=falsification_reason,
            decision_reason=decision_reason,
            thresholds=dict(spec.thresholds),
            metrics=metrics,
            percentile_overlay=percentile_overlay,
            coverage=coverage,
            audit={"coverage": asdict(coverage), "falsification_rule": spec.falsification},
            takeaway=takeaway,
            chart_payload=chart_payload,
        )

    def _eval_p1(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        real_owner = _last_n_valid(c["real_owner_ps"], 6)
        real_fcf = _last_n_valid(c["real_fcf_ps"], 6)
        real_rev = _last_n_valid(c["real_revenue_ps"], 6)

        real_owner_cagr = float("nan")
        decline = float("nan")
        if len(real_owner) >= 2:
            years = int(real_owner.index[-1] - real_owner.index[0])
            real_owner_cagr = _cagr(float(real_owner.iloc[0]), float(real_owner.iloc[-1]), max(years, 1))
            decline = _safe_div(real_owner.iloc[-1], real_owner.iloc[0]) - 1

        real_fcf_cagr = float("nan")
        if len(real_fcf) >= 2:
            years = int(real_fcf.index[-1] - real_fcf.index[0])
            real_fcf_cagr = _cagr(float(real_fcf.iloc[0]), float(real_fcf.iloc[-1]), max(years, 1))

        real_rev_cagr = float("nan")
        if len(real_rev) >= 2:
            years = int(real_rev.index[-1] - real_rev.index[0])
            real_rev_cagr = _cagr(float(real_rev.iloc[0]), float(real_rev.iloc[-1]), max(years, 1))

        coverage = _coverage(
            spec.required_fields,
            {
                "net_income": c["net_income"],
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
                "diluted_shares": c["shares"],
                "revenue": c["revenue"],
                "cpi": c["macro"].get("cpi"),
            },
        )

        falsified = bool(np.isfinite(real_owner_cagr) and real_owner_cagr <= 0) or bool(
            np.isfinite(decline) and decline <= spec.thresholds["decline_breach"]
        )
        fals_reason = spec.falsification if falsified else None

        base_signal = _signal_from_threshold(real_owner_cagr, spec.thresholds["strong"], spec.thresholds["watch"], True)
        metrics = {
            "real_owner_earnings_cagr_5y": real_owner_cagr,
            "real_fcf_per_share_cagr_5y": real_fcf_cagr,
            "real_revenue_per_share_cagr_5y": real_rev_cagr,
            "cumulative_real_owner_earnings_change": decline,
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Real Per-Share Compounding",
            x_label="Fiscal Year",
            y_label="Real USD / share",
            unit="usd_per_share_real",
            series=[
                _chart_series_from_series("Owner Earnings / Share", real_owner),
                _chart_series_from_series("FCF / Share", real_fcf),
                _chart_series_from_series("Revenue / Share", real_rev),
            ],
        )
        takeaway = (
            f"Real owner-earnings per share compounded at {real_owner_cagr:.1%} over the observed cycle."
            if np.isfinite(real_owner_cagr)
            else "Real per-share compounding cannot be assessed with current coverage."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=real_owner,
            percentile_value=float(real_owner.iloc[-1]) if len(real_owner) else None,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=fals_reason,
            decision_reason="Real per-share owner earnings trend classified using absolute+percentile thresholds.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _eval_p2(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        inc_roic = _last_n_valid(c["incremental_roic"], 5)
        med = float(inc_roic.median()) if not inc_roic.empty else float("nan")
        below_count = int((inc_roic < spec.thresholds["watch"]).sum()) if not inc_roic.empty else 0

        coverage = _coverage(
            spec.required_fields,
            {
                "operating_income": c["operating_income"],
                "income_tax_expense": c["tax_expense"],
                "total_equity": c["equity"],
                "total_debt": c["debt"],
                "cash_and_equivalents": c["cash"],
            },
        )

        falsified = below_count >= spec.thresholds["years_below_hurdle"]
        base_signal = _signal_from_threshold(med, spec.thresholds["strong"], spec.thresholds["watch"], True)
        metrics = {
            "median_incremental_roic_5y": med,
            "incremental_roic_years_below_watch": below_count,
            "incremental_roic_series": inc_roic.to_dict(),
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Incremental ROIC Trend",
            x_label="Fiscal Year",
            y_label="Incremental ROIC",
            unit="ratio",
            series=[_chart_series_from_series("Incremental ROIC", inc_roic)],
            reference_lines=[
                ChartReferenceLine(label="Strong", value=float(spec.thresholds["strong"])),
                ChartReferenceLine(label="Watch", value=float(spec.thresholds["watch"])),
            ],
        )
        takeaway = (
            f"Median incremental ROIC was {med:.1%}, indicating how effectively new capital translated into profit."
            if np.isfinite(med)
            else "Incremental capital-return signal is unavailable with current filings."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=inc_roic,
            percentile_value=med,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Incremental capital return profile evaluated over trailing 5 fiscal years.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _eval_p3(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        assets = c["annual"].get("total_assets", pd.Series(dtype=float))
        accrual_ratio = (c["net_income"] - c["cfo"]) / assets.rolling(2).mean().replace(0, np.nan)
        cfo_to_ni = c["cfo"] / c["net_income"].replace(0, np.nan)
        wc_proxy = c["working_capital_accrual"]

        recent_acc = _last_n_valid(accrual_ratio, 5)
        recent_cfo_ni = _last_n_valid(cfo_to_ni, 5)

        years_bad_accrual = int((recent_acc > spec.thresholds["accrual_fail"]).sum())
        cfo_ni_latest = float(recent_cfo_ni.iloc[-1]) if len(recent_cfo_ni) else float("nan")

        coverage = _coverage(
            spec.required_fields,
            {
                "net_income": c["net_income"],
                "operating_cash_flow": c["cfo"],
                "total_assets": assets,
                "current_assets": c["current_assets"],
                "current_liabilities": c["current_liabilities"],
            },
        )

        falsified = (years_bad_accrual >= 2) or (np.isfinite(cfo_ni_latest) and cfo_ni_latest < spec.thresholds["cfo_to_ni_fail"])

        acc_latest = float(recent_acc.iloc[-1]) if len(recent_acc) else float("nan")
        # lower accrual ratio is better
        if np.isfinite(acc_latest) and np.isfinite(cfo_ni_latest):
            if acc_latest <= spec.thresholds["accrual_strong"] and cfo_ni_latest >= spec.thresholds["cfo_to_ni_strong"]:
                base_signal = "Strong"
            elif acc_latest <= spec.thresholds["accrual_fail"] and cfo_ni_latest >= spec.thresholds["cfo_to_ni_fail"]:
                base_signal = "Watch"
            else:
                base_signal = "Weak"
        else:
            base_signal = "Insufficient"

        metrics = {
            "accrual_ratio_latest": acc_latest,
            "accrual_ratio_series": recent_acc.to_dict(),
            "cfo_to_ni_latest": cfo_ni_latest,
            "working_capital_accrual_proxy_latest": float(wc_proxy.dropna().iloc[-1]) if wc_proxy.dropna().shape[0] else float("nan"),
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Accrual Quality and Cash Conversion",
            x_label="Fiscal Year",
            y_label="Ratio",
            unit="ratio",
            series=[
                _chart_series_from_series("Accrual Ratio", recent_acc),
                _chart_series_from_series("CFO / Net Income", recent_cfo_ni),
            ],
            reference_lines=[
                ChartReferenceLine(label="Accrual Strong", value=float(spec.thresholds["accrual_strong"])),
                ChartReferenceLine(label="Accrual Fail", value=float(spec.thresholds["accrual_fail"])),
            ],
        )
        takeaway = (
            f"Accrual ratio is {acc_latest:.1%} and CFO/NI is {cfo_ni_latest:.2f}, indicating earnings cash backing quality."
            if np.isfinite(acc_latest) and np.isfinite(cfo_ni_latest)
            else "Cash-backing evidence is incomplete for the latest cycle."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=recent_acc,
            percentile_value=acc_latest,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Accrual intensity and cash-conversion quality evaluated jointly.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=False,
        )

    def _eval_p4(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        rev = _last_n_valid(c["revenue"], 6)
        shares = _last_n_valid(c["shares"], 6)
        net_debt = _last_n_valid(c["net_debt"], 6)
        fcf = _last_n_valid(c["fcf"], 6)

        rev_cagr = float("nan")
        share_cagr = float("nan")
        if len(rev) >= 2:
            rev_cagr = _cagr(float(rev.iloc[0]), float(rev.iloc[-1]), max(int(rev.index[-1] - rev.index[0]), 1))
        if len(shares) >= 2:
            share_cagr = _cagr(float(shares.iloc[0]), float(shares.iloc[-1]), max(int(shares.index[-1] - shares.index[0]), 1))

        dilution_adjusted = rev_cagr - share_cagr if np.isfinite(rev_cagr) and np.isfinite(share_cagr) else float("nan")
        fcf_strength = float(np.nanmedian(fcf.values)) if len(fcf) else float("nan")
        nd_trend = _safe_div(net_debt.iloc[-1] - net_debt.iloc[0], abs(net_debt.iloc[0])) if len(net_debt) >= 2 else float("nan")

        coverage = _coverage(
            spec.required_fields,
            {
                "revenue": c["revenue"],
                "diluted_shares": c["shares"],
                "total_debt": c["debt"],
                "cash_and_equivalents": c["cash"],
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
            },
        )

        weak_fcf = np.isfinite(fcf_strength) and fcf_strength <= 0
        leverage_deterioration = np.isfinite(nd_trend) and nd_trend > 0.25
        falsified = (np.isfinite(share_cagr) and share_cagr > spec.thresholds["watch_share_cagr"] and weak_fcf) or leverage_deterioration

        if np.isfinite(share_cagr):
            if share_cagr <= spec.thresholds["strong_share_cagr"] and not leverage_deterioration:
                base_signal = "Strong"
            elif share_cagr <= spec.thresholds["watch_share_cagr"]:
                base_signal = "Watch"
            else:
                base_signal = "Weak"
        else:
            base_signal = "Insufficient"

        metrics = {
            "revenue_cagr_5y": rev_cagr,
            "share_cagr_5y": share_cagr,
            "dilution_adjusted_growth": dilution_adjusted,
            "net_debt_trend": nd_trend,
        }
        shares_idx = shares / shares.iloc[0] * 100 if len(shares) and shares.iloc[0] != 0 else pd.Series(dtype=float)
        debt_idx = net_debt / abs(net_debt.iloc[0]) * 100 if len(net_debt) and net_debt.iloc[0] != 0 else pd.Series(dtype=float)
        chart_payload = ChartPayload(
            chart_type="line",
            title="Dilution and Leverage Trend (Indexed)",
            x_label="Fiscal Year",
            y_label="Index (start=100)",
            unit="index",
            series=[
                _chart_series_from_series("Share Count Index", shares_idx),
                _chart_series_from_series("Net Debt Index", debt_idx),
            ],
        )
        takeaway = (
            f"Share count CAGR was {share_cagr:.1%} with dilution-adjusted growth of {dilution_adjusted:.1%}."
            if np.isfinite(share_cagr) and np.isfinite(dilution_adjusted)
            else "Dilution-vs-growth funding quality is not fully observable from current filings."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=shares,
            percentile_value=float(shares.iloc[-1]) if len(shares) else None,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Growth funding mix assessed through dilution and leverage trajectories.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=False,
        )

    def _eval_p5(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        payout = _last_n_valid(c["payout"], 5)
        fcf = _last_n_valid(c["fcf"], 5)
        debt = _last_n_valid(c["debt"], 5)

        coverage_series = fcf / payout.replace(0, np.nan)
        coverage_latest = float(coverage_series.dropna().iloc[-1]) if coverage_series.dropna().shape[0] else float("nan")
        payout_rate = float(payout.sum() / fcf.sum()) if fcf.sum() != 0 else float("nan")

        debt_up = False
        if len(debt) >= 2 and np.isfinite(float(debt.iloc[0])) and debt.iloc[0] != 0:
            debt_up = ((debt.iloc[-1] - debt.iloc[0]) / abs(debt.iloc[0])) > 0.2

        falsified = (np.isfinite(coverage_latest) and coverage_latest < 1.0 and debt_up) or (
            np.isfinite(payout_rate) and payout_rate > 1.2
        )

        if np.isfinite(coverage_latest):
            base_signal = _signal_from_threshold(coverage_latest, spec.thresholds["coverage_strong"], spec.thresholds["coverage_watch"], True)
        else:
            base_signal = "Insufficient"

        coverage = _coverage(
            spec.required_fields,
            {
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
                "dividends_paid": c["dividends"],
                "repurchases": c["repurchases"],
                "total_debt": c["debt"],
                "cash_and_equivalents": c["cash"],
            },
        )

        metrics = {
            "fcf_payout_coverage_latest": coverage_latest,
            "payout_to_fcf_5y": payout_rate,
            "debt_growth_flag": debt_up,
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Payout Coverage vs FCF",
            x_label="Fiscal Year",
            y_label="Coverage",
            unit="ratio",
            series=[_chart_series_from_series("FCF / Payout", coverage_series)],
            reference_lines=[
                ChartReferenceLine(label="Strong", value=float(spec.thresholds["coverage_strong"])),
                ChartReferenceLine(label="Watch", value=float(spec.thresholds["coverage_watch"])),
            ],
        )
        takeaway = (
            f"Latest payout coverage is {coverage_latest:.2f}x, indicating whether capital returns were self-funded."
            if np.isfinite(coverage_latest)
            else "Payout sustainability cannot be determined with current coverage."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=coverage_series,
            percentile_value=coverage_latest,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Payout sustainability judged against internally generated free cash flow.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _eval_p6(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        fcf_margin = c["fcf"] / c["revenue"].replace(0, np.nan)
        icov = c["interest_coverage"]

        min_fcf_margin = float(fcf_margin.dropna().min()) if fcf_margin.dropna().shape[0] else float("nan")
        icov_trough = float(icov.dropna().min()) if icov.dropna().shape[0] else float("nan")

        # simple recovery speed: longest run below median owner earnings
        oe = c["owner_earnings"].dropna()
        recovery_years = float("nan")
        if len(oe) >= 3:
            med = float(oe.median())
            streak = 0
            max_streak = 0
            for val in oe.values:
                if val < med:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            recovery_years = float(max_streak)

        coverage = _coverage(
            spec.required_fields,
            {
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
                "revenue": c["revenue"],
                "operating_income": c["operating_income"],
                "interest_expense": c["interest_expense"],
                "recession": c["macro"].get("recession"),
            },
        )

        falsified = np.isfinite(icov_trough) and icov_trough < spec.thresholds["watch"]
        base_signal = _signal_from_threshold(icov_trough, spec.thresholds["strong"], spec.thresholds["watch"], True)

        metrics = {
            "min_fcf_margin": min_fcf_margin,
            "interest_coverage_trough": icov_trough,
            "recovery_speed_years": recovery_years,
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Resilience Trough Metrics",
            x_label="Fiscal Year",
            y_label="Ratio",
            unit="ratio",
            series=[
                _chart_series_from_series("Interest Coverage", icov),
                _chart_series_from_series("FCF Margin", fcf_margin),
            ],
            reference_lines=[
                ChartReferenceLine(label="Strong", value=float(spec.thresholds["strong"])),
                ChartReferenceLine(label="Watch", value=float(spec.thresholds["watch"])),
            ],
        )
        takeaway = (
            f"Interest coverage troughed at {icov_trough:.2f}x with minimum FCF margin of {min_fcf_margin:.1%} through the sample."
            if np.isfinite(icov_trough) and np.isfinite(min_fcf_margin)
            else "Stress-resilience evidence is incomplete from available history."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=icov.dropna(),
            percentile_value=icov_trough,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Stress resilience scored with solvency trough and cash margin durability.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _scenario_growth_triplet(self, series: pd.Series, macro: Dict[str, pd.Series]) -> Dict[str, float]:
        """History+macro conditioned growth triplet for base/bear/bull."""
        s = _last_n_valid(series, 6)
        if len(s) < 2:
            return {"bear": float("nan"), "base": float("nan"), "bull": float("nan")}

        # annual growth history
        growth = s.pct_change().dropna()
        if growth.empty:
            return {"bear": float("nan"), "base": float("nan"), "bull": float("nan")}

        recession = macro.get("recession", pd.Series(dtype=float))
        baa_spread = macro.get("baa_spread", pd.Series(dtype=float))

        rec_penalty = 0.0
        if not recession.empty:
            rec_penalty = 0.01 if float(recession.dropna().iloc[-1]) >= 1 else 0.0

        spread_penalty = 0.0
        if not baa_spread.empty and baa_spread.dropna().shape[0] >= 24:
            tail = baa_spread.dropna().iloc[-1]
            hist = baa_spread.dropna()
            z = _safe_div(tail - hist.mean(), hist.std())
            if np.isfinite(z) and z > 1:
                spread_penalty = min(0.03, 0.01 * z)

        # French factors: CMA (investment) and RMW (profitability) act as regime bands.
        cma = macro.get("ff5_cma", pd.Series(dtype=float)).dropna()
        rmw = macro.get("ff5_rmw", pd.Series(dtype=float)).dropna()
        factor_adjust = 0.0
        if len(cma) >= 24 and len(rmw) >= 24:
            cma_tail = float(cma.iloc[-1])
            rmw_tail = float(rmw.iloc[-1])
            # High CMA and weak RMW imply tighter forward assumptions.
            if cma_tail > cma.quantile(0.75):
                factor_adjust += 0.01
            if rmw_tail < rmw.quantile(0.25):
                factor_adjust += 0.01

        base = float(np.nanmedian(growth.values)) - rec_penalty - spread_penalty - factor_adjust
        bear = float(np.nanquantile(growth.values, 0.20)) - rec_penalty - spread_penalty - factor_adjust
        bull = float(np.nanquantile(growth.values, 0.80))
        return {"bear": bear, "base": base, "bull": bull}

    def _eval_f1(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        inc_roic = _last_n_valid(c["incremental_roic"], 5)
        growth_triplet = self._scenario_growth_triplet(c["revenue"], c["macro"])

        base_roic = float(np.nanmedian(inc_roic.values)) if len(inc_roic) else float("nan")
        hurdle = 0.08
        forward_spread = base_roic - hurdle if np.isfinite(base_roic) else float("nan")

        falsified = np.isfinite(forward_spread) and forward_spread < spec.thresholds["watch_spread"]
        base_signal = _signal_from_threshold(forward_spread, spec.thresholds["strong_spread"], spec.thresholds["watch_spread"], True)

        coverage = _coverage(
            spec.required_fields,
            {
                "operating_income": c["operating_income"],
                "income_tax_expense": c["tax_expense"],
                "total_equity": c["equity"],
                "total_debt": c["debt"],
                "cash_and_equivalents": c["cash"],
            },
        )

        metrics = {
            "forward_incremental_roic_spread": forward_spread,
            "base_incremental_roic": base_roic,
            "scenario_growth_triplet": growth_triplet,
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Forward Reinvestment Spread Inputs",
            x_label="Fiscal Year",
            y_label="Incremental ROIC",
            unit="ratio",
            series=[_chart_series_from_series("Incremental ROIC", inc_roic)],
            reference_lines=[
                ChartReferenceLine(label="Hurdle", value=0.08),
                ChartReferenceLine(label="Strong Spread", value=0.08 + float(spec.thresholds["strong_spread"])),
                ChartReferenceLine(label="Watch Spread", value=0.08 + float(spec.thresholds["watch_spread"])),
            ],
        )
        takeaway = (
            f"Forward incremental ROIC spread is {forward_spread:.1%} versus hurdle, using base historical capital efficiency."
            if np.isfinite(forward_spread)
            else "Forward reinvestment spread is unavailable from current data."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=inc_roic,
            percentile_value=base_roic,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Forward reinvestment spread derived from historical incremental ROIC and macro-conditioned growth.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _eval_f2(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        oe_ps = c["owner_earnings_ps"].dropna()
        growth = self._scenario_growth_triplet(oe_ps, c["macro"])
        last_oe = float(oe_ps.iloc[-1]) if len(oe_ps) else float("nan")

        def project_path(start: float, g: float, years: int = 5) -> List[float]:
            vals = [start]
            x = start
            for _ in range(years):
                x = x * (1 + g)
                vals.append(x)
            return vals

        base_path = project_path(last_oe, growth["base"]) if np.isfinite(last_oe) and np.isfinite(growth["base"]) else []
        bear_path = project_path(last_oe, growth["bear"]) if np.isfinite(last_oe) and np.isfinite(growth["bear"]) else []
        bull_path = project_path(last_oe, growth["bull"]) if np.isfinite(last_oe) and np.isfinite(growth["bull"]) else []

        bear_drawdown = float("nan")
        if bear_path:
            mn = min(bear_path)
            bear_drawdown = _safe_div(mn - last_oe, abs(last_oe))

        falsified = (bear_path and min(bear_path) <= 0) or (
            np.isfinite(bear_drawdown) and bear_drawdown < spec.thresholds["max_bear_drawdown"]
        )

        if base_path and bear_path:
            base_signal = "Strong" if min(bear_path) > 0 and growth["base"] > 0 else "Watch"
        else:
            base_signal = "Insufficient"

        coverage = _coverage(
            spec.required_fields,
            {
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
                "diluted_shares": c["shares"],
                "cpi": c["macro"].get("cpi"),
                "recession": c["macro"].get("recession"),
                "baa_spread": c["macro"].get("baa_spread"),
            },
        )

        metrics = {
            "growth_assumptions": growth,
            "owner_earnings_base_path": base_path,
            "owner_earnings_bear_path": bear_path,
            "owner_earnings_bull_path": bull_path,
            "bear_drawdown": bear_drawdown,
        }
        years = [f"Y{i}" for i in range(0, 6)]
        chart_payload = ChartPayload(
            chart_type="line",
            title="5Y Owner-Earnings Scenarios",
            x_label="Projection Year",
            y_label="Owner Earnings / Share",
            unit="usd_per_share",
            series=[
                _chart_series_from_points("Bear", years, bear_path),
                _chart_series_from_points("Base", years, base_path),
                _chart_series_from_points("Bull", years, bull_path),
            ],
        )
        takeaway = (
            f"Bear-case drawdown is {bear_drawdown:.1%}, with scenario paths anchored to history and macro regime."
            if np.isfinite(bear_drawdown)
            else "Scenario distribution cannot be formed with current owner-earnings history."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=oe_ps,
            percentile_value=last_oe,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="5Y owner-earnings distribution generated from history + macro-conditioned scenario bands.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _eval_f3(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        op_margin = (c["operating_income"] / c["revenue"].replace(0, np.nan)).dropna()
        if op_margin.empty:
            cv = float("nan")
            latest_margin = float("nan")
            p90 = float("nan")
        else:
            cv = float(op_margin.std() / op_margin.mean()) if op_margin.mean() != 0 else float("nan")
            latest_margin = float(op_margin.iloc[-1])
            p90 = float(np.nanquantile(op_margin.values, 0.9))

        required_margin = latest_margin
        falsified = np.isfinite(required_margin) and np.isfinite(p90) and required_margin > p90 * 1.10

        if np.isfinite(cv):
            base_signal = "Strong" if cv <= spec.thresholds["max_cv"] else "Watch"
            if cv > spec.thresholds["max_cv"] * 1.5:
                base_signal = "Weak"
        else:
            base_signal = "Insufficient"

        coverage = _coverage(
            spec.required_fields,
            {
                "revenue": c["revenue"],
                "operating_income": c["operating_income"],
                "cpi": c["macro"].get("cpi"),
            },
        )

        metrics = {
            "operating_margin_cv": cv,
            "operating_margin_latest": latest_margin,
            "historical_margin_p90": p90,
            "required_margin_base_case": required_margin,
        }
        chart_payload = ChartPayload(
            chart_type="line",
            title="Margin Durability Band",
            x_label="Fiscal Year",
            y_label="Operating Margin",
            unit="ratio",
            series=[_chart_series_from_series("Operating Margin", op_margin)],
            reference_lines=[ChartReferenceLine(label="Historical P90", value=p90)] if np.isfinite(p90) else [],
        )
        takeaway = (
            f"Operating-margin variability (CV) is {cv:.2f}, indicating the stability of pricing power through cycles."
            if np.isfinite(cv)
            else "Margin durability cannot be scored due to insufficient operating-margin history."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=op_margin,
            percentile_value=latest_margin,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Margin durability measured via persistence band and coefficient of variation.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _eval_f4(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        debt = c["debt"].dropna()
        cash = c["cash"].dropna()
        op_income = c["operating_income"].dropna()
        int_exp = c["interest_expense"].dropna()

        stress_ebit = float(op_income.iloc[-1] * 0.70) if len(op_income) else float("nan")
        stress_icov = _safe_div(stress_ebit, float(int_exp.iloc[-1])) if len(int_exp) else float("nan")

        latest_nd = float(debt.iloc[-1] - cash.iloc[-1]) if len(debt) and len(cash) else float("nan")
        latest_ebitda_proxy = float(op_income.iloc[-1]) if len(op_income) else float("nan")
        stress_nd_ebitda = _safe_div(latest_nd, latest_ebitda_proxy * 0.70 if np.isfinite(latest_ebitda_proxy) else np.nan)

        current_ratio = c["current_ratio"].dropna()
        liquidity_runway = float(current_ratio.iloc[-1]) if len(current_ratio) else float("nan")

        falsified = (np.isfinite(stress_icov) and stress_icov < spec.thresholds["watch"]) or (
            np.isfinite(liquidity_runway) and liquidity_runway < 1.0
        )
        base_signal = _signal_from_threshold(stress_icov, spec.thresholds["strong"], spec.thresholds["watch"], True)

        coverage = _coverage(
            spec.required_fields,
            {
                "total_debt": c["debt"],
                "cash_and_equivalents": c["cash"],
                "current_assets": c["current_assets"],
                "current_liabilities": c["current_liabilities"],
                "operating_income": c["operating_income"],
                "interest_expense": c["interest_expense"],
            },
        )

        metrics = {
            "stress_interest_coverage": stress_icov,
            "stress_net_debt_ebitda_proxy": stress_nd_ebitda,
            "liquidity_runway_current_ratio": liquidity_runway,
        }
        chart_payload = ChartPayload(
            chart_type="bar",
            title="Stress Funding and Liquidity",
            x_label="Stress Metric",
            y_label="Value",
            unit="ratio",
            series=[
                _chart_series_from_points(
                    "Stress",
                    ["Interest Coverage", "Net Debt / EBITDA", "Current Ratio"],
                    [stress_icov, stress_nd_ebitda, liquidity_runway],
                )
            ],
            reference_lines=[
                ChartReferenceLine(label="Strong Coverage", value=float(spec.thresholds["strong"])),
                ChartReferenceLine(label="Watch Coverage", value=float(spec.thresholds["watch"])),
            ],
        )
        takeaway = (
            f"Stress interest coverage is {stress_icov:.2f}x with liquidity runway (current ratio) at {liquidity_runway:.2f}."
            if np.isfinite(stress_icov) and np.isfinite(liquidity_runway)
            else "Funding-resilience stress metrics are partially unavailable."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=c["interest_coverage"].dropna(),
            percentile_value=stress_icov,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Downside funding stress test applied to solvency and liquidity metrics.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )

    def _implied_growth_reverse_dcf(self, fcf0: float, equity_value: float, discount_rate: float, terminal_growth: float, years: int = 10) -> float:
        if not all(np.isfinite(x) for x in [fcf0, equity_value, discount_rate, terminal_growth]):
            return float("nan")
        if fcf0 <= 0 or equity_value <= 0 or discount_rate <= terminal_growth:
            return float("nan")

        def pv_from_g(g: float) -> float:
            pv = 0.0
            fcf = fcf0
            for t in range(1, years + 1):
                fcf = fcf * (1 + g)
                pv += fcf / ((1 + discount_rate) ** t)
            terminal = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            pv += terminal / ((1 + discount_rate) ** years)
            return pv

        lo, hi = -0.50, 0.80
        for _ in range(80):
            mid = (lo + hi) / 2
            pv = pv_from_g(mid)
            if pv > equity_value:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def _eval_f5(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        price = c["latest_price"] if c["latest_price"] is not None else float("nan")
        shares = c["shares"].dropna()
        fcf = c["fcf"].dropna()
        risk_free = c["macro"].get("risk_free_10y", pd.Series(dtype=float)).dropna()

        shares_latest = float(shares.iloc[-1]) if len(shares) else float("nan")
        fcf0 = float(fcf.iloc[-1]) if len(fcf) else float("nan")
        equity_value = float(price * shares_latest) if np.isfinite(price) and np.isfinite(shares_latest) else float("nan")

        r = 0.09
        if len(risk_free):
            r = float(risk_free.iloc[-1] / 100.0 + 0.05)

        implied_growth = self._implied_growth_reverse_dcf(fcf0, equity_value, r, terminal_growth=0.025, years=10)

        hist_growth = _last_n_valid(c["fcf"], 6).pct_change().dropna()
        p60 = float(np.nanquantile(hist_growth.values, 0.60)) if len(hist_growth) else float("nan")
        p90 = float(np.nanquantile(hist_growth.values, 0.90)) if len(hist_growth) else float("nan")

        falsified = np.isfinite(implied_growth) and np.isfinite(p90) and implied_growth > p90

        if np.isfinite(implied_growth) and np.isfinite(p60) and np.isfinite(p90):
            if implied_growth <= p60:
                base_signal = "Strong"
            elif implied_growth <= p90:
                base_signal = "Watch"
            else:
                base_signal = "Weak"
        else:
            base_signal = "Insufficient"

        coverage = _coverage(
            spec.required_fields,
            {
                "price": price,
                "diluted_shares": c["shares"],
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
                "risk_free_10y": c["macro"].get("risk_free_10y"),
            },
        )

        metrics = {
            "implied_growth_reverse_dcf": implied_growth,
            "historical_growth_p60": p60,
            "historical_growth_p90": p90,
            "discount_rate_used": r,
        }
        chart_payload = ChartPayload(
            chart_type="bar",
            title="Implied Growth Feasibility",
            x_label="Growth Reference",
            y_label="Growth Rate",
            unit="ratio",
            series=[
                _chart_series_from_points(
                    "Growth",
                    ["Implied", "Historical p60", "Historical p90"],
                    [implied_growth, p60, p90],
                )
            ],
        )
        takeaway = (
            f"Market-implied growth is {implied_growth:.1%} versus historical feasible bands (p60 {p60:.1%}, p90 {p90:.1%})."
            if np.isfinite(implied_growth) and np.isfinite(p60) and np.isfinite(p90)
            else "Implied-growth feasibility is indeterminate with current inputs."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=hist_growth,
            percentile_value=implied_growth,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Reverse-DCF implied growth checked against feasible historical growth distribution.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=False,
        )

    def _eval_f6(self, spec: QuestionSpec, c: Dict[str, Any]) -> QuestionResult:
        market_cap = c["market_cap"]
        if not np.isfinite(market_cap) or market_cap <= 0:
            market_cap = float("nan")

        payout_latest = float(c["payout"].dropna().iloc[-1]) if c["payout"].dropna().shape[0] else float("nan")
        cash_yield = _safe_div(payout_latest, market_cap)

        real_oe = _last_n_valid(c["real_owner_ps"], 6)
        real_growth = float("nan")
        if len(real_oe) >= 2:
            years = int(real_oe.index[-1] - real_oe.index[0])
            real_growth = _cagr(float(real_oe.iloc[0]), float(real_oe.iloc[-1]), max(years, 1))

        shares = _last_n_valid(c["shares"], 6)
        dilution_drag = float("nan")
        if len(shares) >= 2:
            years = int(shares.index[-1] - shares.index[0])
            dilution_drag = -_cagr(float(shares.iloc[0]), float(shares.iloc[-1]), max(years, 1))

        expected_return = (
            (cash_yield if np.isfinite(cash_yield) else 0.0)
            + (real_growth if np.isfinite(real_growth) else 0.0)
            + (dilution_drag if np.isfinite(dilution_drag) else 0.0)
        )

        implied_growth_ref = self._eval_f5(
            next(s for s in self.specs if s.question_id == "F5"), c
        ).metrics.get("implied_growth_reverse_dcf", float("nan"))

        # If implied growth is very high and expected return only clears threshold,
        # it likely depends on valuation/multiple support.
        needs_multiple = np.isfinite(implied_growth_ref) and implied_growth_ref > 0.12 and expected_return < 0.12
        falsified = needs_multiple

        base_signal = _signal_from_threshold(expected_return, spec.thresholds["strong"], spec.thresholds["watch"], True)

        coverage = _coverage(
            spec.required_fields,
            {
                "price": c["latest_price"],
                "diluted_shares": c["shares"],
                "dividends_paid": c["dividends"],
                "repurchases": c["repurchases"],
                "operating_cash_flow": c["cfo"],
                "capex": c["capex"],
                "cpi": c["macro"].get("cpi"),
            },
        )

        metrics = {
            "cash_yield": cash_yield,
            "real_growth": real_growth,
            "dilution_friction": dilution_drag,
            "expected_return_no_multiple_expansion": expected_return,
            "implied_growth_reference": implied_growth_ref,
        }
        chart_payload = ChartPayload(
            chart_type="bar",
            title="Return Decomposition (No Multiple Expansion)",
            x_label="Return Component",
            y_label="Contribution",
            unit="ratio",
            series=[
                _chart_series_from_points(
                    "Contribution",
                    ["Cash Yield", "Real Growth", "Dilution/Friction", "Expected Return"],
                    [cash_yield, real_growth, dilution_drag, expected_return],
                )
            ],
            reference_lines=[
                ChartReferenceLine(label="Strong", value=float(spec.thresholds["strong"])),
                ChartReferenceLine(label="Watch", value=float(spec.thresholds["watch"])),
            ],
        )
        takeaway = (
            f"Base expected return is {expected_return:.1%} before any multiple expansion."
            if np.isfinite(expected_return)
            else "Return decomposition cannot be completed from available coverage."
        )
        return self._basic_result(
            spec,
            metrics,
            base_signal,
            percentile_series=real_oe.pct_change().dropna(),
            percentile_value=expected_return,
            coverage=coverage,
            falsified=falsified,
            falsification_reason=spec.falsification if falsified else None,
            decision_reason="Shareholder return decomposition excludes multiple expansion by construction.",
            takeaway=takeaway,
            chart_payload=chart_payload,
            higher_is_better=True,
        )
