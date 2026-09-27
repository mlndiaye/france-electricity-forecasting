"""Command-line entry points for the ingestion pipeline."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from felec.ingestion import calendar as calendar_connector
from felec.ingestion import rte as rte_connector
from felec.ingestion import weather as weather_connector
from felec.ingestion.storage import save_processed
from felec.modeling.backtest import walk_forward_backtest
from felec.processing.build_dataset import build_dataset

# First date confirmed to have real AROME France D-1 forecast data via the
# Previous Runs API (verified by binary search against the live API,
# 2026-09-25) -- see docs/decisions/0001-problem-definition-scope-and-weather-window.md
STUDY_WINDOW_START = date(2024, 2, 1)


def backfill() -> None:
    today = datetime.now(UTC).date()
    rte_connector.backfill(STUDY_WINDOW_START, today)
    weather_connector.backfill(STUDY_WINDOW_START, today)
    calendar_connector.backfill(STUDY_WINDOW_START.year, today.year)


def refresh() -> None:
    rte_connector.refresh()
    weather_connector.refresh()
    calendar_connector.refresh()


def backtest() -> None:
    df = pd.read_parquet("data/processed/dataset.parquet")
    df = df.set_index("date_heure").sort_index()
    last_date = df.index.tz_convert("Europe/Paris").normalize().unique().max().date()
    backtest_start = last_date - timedelta(days=365)  # ADR 0004: ~11-12 month backtest window
    results = walk_forward_backtest(df, backtest_start=backtest_start)
    save_processed(results, "backtest_results.parquet")
    print(
        f"Backtest: {len(results)} hourly rows, "
        f"{results['date_heure'].dt.date.nunique()} days, "
        f"saved to data/processed/backtest_results.parquet"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("backfill")
    subparsers.add_parser("refresh")
    subparsers.add_parser("build-dataset")
    subparsers.add_parser("backtest")

    args = parser.parse_args()

    if args.command == "backfill":
        backfill()
    elif args.command == "refresh":
        refresh()
    elif args.command == "build-dataset":
        build_dataset()
    elif args.command == "backtest":
        backtest()


if __name__ == "__main__":
    main()
