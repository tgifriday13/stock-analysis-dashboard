"""CLI entrypoint for running first-principles analysis."""

from __future__ import annotations

import argparse
from datetime import datetime

from .engine import FirstPrinciplesEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-principles past/future signal engine.")
    parser.add_argument("ticker", help="US listed ticker symbol (SEC filer)")
    parser.add_argument("--as-of", help="PIT as-of date YYYY-MM-DD", default=None)
    parser.add_argument("--force-refresh", action="store_true", help="Ignore local cache")
    parser.add_argument("--sector", help="Override sector template detection", default=None)
    parser.add_argument(
        "--view",
        help="Question view filter: all, past, or future",
        choices=["all", "past", "future"],
        default="all",
    )
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None

    engine = FirstPrinciplesEngine(force_refresh=args.force_refresh)
    report = engine.run(args.ticker, as_of_date=as_of, sector_override=args.sector)
    paths = engine.save_report(report, time_view=args.view)

    print(f"Decision: {report.synthesis.decision}")
    print(f"Hard fail: {report.synthesis.hard_fail}")
    print(f"JSON: {paths['json']}")
    print(f"HTML: {paths['html']}")


if __name__ == "__main__":
    main()
