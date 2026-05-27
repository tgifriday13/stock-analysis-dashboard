"""Typed schemas for the first-principles past/future signal engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional


Signal = str


@dataclass
class FilingMetadata:
    """Maps an accession number to SEC filing timing metadata."""

    accession: str
    form: str
    filing_date: Optional[date]
    acceptance_datetime: Optional[datetime]
    pit_available_date: Optional[date]


@dataclass
class RawFact:
    """A canonicalized fact derived from SEC companyfacts JSON."""

    cik: str
    ticker: str
    canonical_field: str
    taxonomy: str
    tag: str
    unit: str
    value: float
    fiscal_year: Optional[int]
    fiscal_period: Optional[str]
    form: Optional[str]
    period_end: date
    period_start: Optional[date]
    filed_date: Optional[date]
    accession: Optional[str]
    acceptance_datetime: Optional[datetime]
    pit_available_date: Optional[date]


@dataclass
class PanelCoverage:
    """Coverage summary for a question or panel."""

    required_fields: List[str]
    available_fields: List[str]
    missing_fields: List[str]
    coverage_ratio: float


@dataclass
class ChartSeries:
    """A typed chart series for one first-principles question."""

    name: str
    x: List[str]
    y: List[float]


@dataclass
class ChartReferenceLine:
    """Optional horizontal/vertical reference line metadata."""

    label: str
    value: float


@dataclass
class ChartPayload:
    """Chart-ready payload so rendering does not parse ad-hoc metric dicts."""

    chart_type: str
    title: str
    x_label: str
    y_label: str
    unit: str
    series: List[ChartSeries]
    reference_lines: List[ChartReferenceLine] = field(default_factory=list)


@dataclass
class QuestionResult:
    """Final result artifact for one first-principles question."""

    question_id: str
    question: str
    metric_family: List[str]
    signal: Signal
    falsified: bool
    falsification_reason: Optional[str]
    decision_reason: str
    thresholds: Dict[str, Any]
    metrics: Dict[str, Any]
    percentile_overlay: Dict[str, Any]
    coverage: PanelCoverage
    audit: Dict[str, Any] = field(default_factory=dict)
    takeaway: str = ""
    chart_payload: Optional[ChartPayload] = None


@dataclass
class SignalSynthesis:
    """Portfolio-level synthesis across all question outcomes."""

    decision: str
    hard_fail: bool
    reasons: List[str]
    strong_count: int
    watch_count: int
    weak_count: int
    insufficient_count: int


@dataclass
class EngineMeta:
    """Top-level metadata for reproducibility and PIT auditability."""

    ticker: str
    cik: str
    run_timestamp: datetime
    as_of_date: date
    accepted_cutoff_rule: str
    sector_template: str
    data_freshness: Dict[str, Any]


@dataclass
class FirstPrinciplesReport:
    """Top-level artifact exported by the first-principles engine."""

    meta: EngineMeta
    questions: List[QuestionResult]
    synthesis: SignalSynthesis

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def question_by_id(self, question_id: str) -> Optional[QuestionResult]:
        """Lookup helper."""
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None
