"""Read-only FastAPI service exposing the pipeline's already-computed results.
Never calls predict_next_day() or any other modeling function directly --
Airflow already produces these results daily (see ADR 0007/0008); this API
only serves them. See ADR 0010.
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException

from felec.ingestion.storage import load_processed

app = FastAPI(title="France Electricity Forecasting API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast/latest")
def forecast_latest() -> list[dict]:
    try:
        df = load_processed("forecast_latest.parquet")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="No forecast available yet -- has `ingest predict` been run?",
        ) from exc
    return df.to_dict(orient="records")


@app.get("/forecast/history")
def forecast_history(start: str | None = None, end: str | None = None) -> list[dict]:
    start_date = _parse_date_param("start", start)
    end_date = _parse_date_param("end", end)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=400, detail="start must not be after end")

    try:
        backtest = load_processed("quantile_backtest_results.parquet")
        dataset = load_processed("dataset.parquet")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="No backtest available yet -- has `ingest quantile-backtest` been run?",
        ) from exc

    merged = backtest.merge(
        dataset[["date_heure", "consommation", "prevision_j1"]], on="date_heure", how="inner"
    )
    merged = merged.rename(columns={"prevision_j1": "rte_pred"})

    dates = merged["date_heure"].dt.date
    if start_date is not None:
        merged = merged[dates >= start_date]
        dates = merged["date_heure"].dt.date
    if end_date is not None:
        merged = merged[dates <= end_date]

    return merged.to_dict(orient="records")


def _parse_date_param(name: str, value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"{name} must be an ISO date (YYYY-MM-DD)"
        ) from exc
