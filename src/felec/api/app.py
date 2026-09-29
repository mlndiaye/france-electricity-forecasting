"""Read-only FastAPI service exposing the pipeline's already-computed results.
Never calls predict_next_day() or any other modeling function directly --
Airflow already produces these results daily (see ADR 0007/0008); this API
only serves them. See ADR 0010.
"""

from __future__ import annotations

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
