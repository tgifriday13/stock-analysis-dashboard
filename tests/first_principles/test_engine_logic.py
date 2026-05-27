from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.first_principles.config import get_question_specs
from src.first_principles.questions import FirstPrinciplesQuestionEngine
from src.first_principles.transforms import build_pit_view, build_research_view


def _macro_bundle() -> dict:
    idx = pd.date_range("2018-01-01", periods=100, freq="ME")
    return {
        "cpi": pd.Series(np.linspace(250, 320, len(idx)), index=idx),
        "recession": pd.Series(np.zeros(len(idx)), index=idx),
        "baa_spread": pd.Series(np.linspace(1.6, 2.4, len(idx)), index=idx),
        "risk_free_10y": pd.Series(np.linspace(1.5, 4.0, len(idx)), index=idx),
        "ff5_rmw": pd.Series(np.linspace(0.001, 0.004, len(idx)), index=idx),
        "ff5_cma": pd.Series(np.linspace(0.001, 0.003, len(idx)), index=idx),
    }


def _base_panel() -> pd.DataFrame:
    years = pd.Index([2019, 2020, 2021, 2022, 2023, 2024], name="fiscal_year")
    return pd.DataFrame(
        {
            "revenue": [100, 108, 118, 130, 142, 158],
            "net_income": [10, 11, 13, 15, 18, 22],
            "operating_income": [14, 15, 17, 20, 23, 28],
            "operating_cash_flow": [15, 16, 18, 22, 25, 31],
            "capex": [3, 3, 4, 4, 5, 6],
            "diluted_shares": [10, 10, 10, 10, 10, 10],
            "total_assets": [120, 128, 138, 152, 168, 185],
            "total_debt": [35, 34, 33, 32, 31, 30],
            "cash_and_equivalents": [8, 9, 10, 11, 12, 13],
            "total_equity": [55, 60, 66, 74, 82, 92],
            "current_assets": [24, 25, 27, 29, 31, 34],
            "current_liabilities": [18, 18, 19, 20, 20, 21],
            "interest_expense": [2.0, 1.9, 1.8, 1.8, 1.7, 1.6],
            "income_tax_expense": [2.1, 2.2, 2.4, 2.8, 3.1, 3.5],
            "dividends_paid": [1.2, 1.2, 1.3, 1.4, 1.5, 1.6],
            "repurchases": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        },
        index=years,
    )


def test_pit_view_enforces_acceptance_cutoff():
    facts = pd.DataFrame(
        [
            {
                "canonical_field": "revenue",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_end": date(2024, 12, 31),
                "filed_date": date(2025, 1, 25),
                "tag_rank": 0,
                "pit_available_date": date(2025, 1, 26),
                "value": 150.0,
            },
            {
                "canonical_field": "revenue",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_end": date(2024, 12, 31),
                "filed_date": date(2025, 2, 15),
                "tag_rank": 0,
                "pit_available_date": date(2025, 2, 16),
                "value": 160.0,
            },
        ]
    )

    pit = build_pit_view(facts, as_of_date=date(2025, 1, 31))
    assert len(pit) == 1
    assert float(pit.iloc[0]["value"]) == 150.0


def test_research_view_keeps_latest_restatement():
    facts = pd.DataFrame(
        [
            {
                "canonical_field": "revenue",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_end": date(2024, 12, 31),
                "filed_date": date(2025, 1, 25),
                "tag_rank": 0,
                "pit_available_date": date(2025, 1, 26),
                "value": 150.0,
            },
            {
                "canonical_field": "revenue",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "period_end": date(2024, 12, 31),
                "filed_date": date(2025, 2, 15),
                "tag_rank": 0,
                "pit_available_date": date(2025, 2, 16),
                "value": 160.0,
            },
        ]
    )

    research = build_research_view(facts)
    assert len(research) == 1
    assert float(research.iloc[0]["value"]) == 160.0


def test_coverage_fail_closed_outputs_insufficient():
    specs = get_question_specs("non_financial")
    engine = FirstPrinciplesQuestionEngine(specs)

    # Missing most required fields by design.
    annual = pd.DataFrame({"revenue": [100, 105, 110]}, index=[2022, 2023, 2024])
    results = engine.evaluate(annual, annual, {}, _macro_bundle(), latest_price=None)

    assert all(r.signal == "Insufficient" for r in results)


def test_threshold_stability_small_revision_keeps_signal():
    specs = get_question_specs("non_financial")
    engine = FirstPrinciplesQuestionEngine(specs)

    annual = _base_panel()
    q = {r.question_id: r for r in engine.evaluate(annual, annual, {}, _macro_bundle(), latest_price=100.0)}
    base_signal = q["P1"].signal

    revised = annual.copy()
    revised["operating_cash_flow"] = revised["operating_cash_flow"] * 1.01
    q2 = {r.question_id: r for r in engine.evaluate(revised, revised, {}, _macro_bundle(), latest_price=100.0)}

    assert base_signal in {"Strong", "Watch"}
    assert q2["P1"].signal == base_signal


def test_sanity_deterioration_triggers_falsification():
    specs = get_question_specs("non_financial")
    engine = FirstPrinciplesQuestionEngine(specs)

    annual = _base_panel()
    annual["operating_income"] = [14, 12, 10, 7, 4, 2]
    annual["interest_expense"] = [2, 2, 2, 2, 2, 2]

    q = {r.question_id: r for r in engine.evaluate(annual, annual, {}, _macro_bundle(), latest_price=90.0)}
    assert q["P6"].falsified is True
    assert q["P6"].signal in {"Weak", "Insufficient"}
