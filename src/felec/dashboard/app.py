"""Streamlit dashboard consuming the serving API -- see ADR 0010 (the API)
and ADR 0011 (this dashboard). Never calls predict_next_day() or any modeling
function directly; only renders what the API already serves.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st

from felec.dashboard.charts import history_chart, latest_forecast_chart
from felec.dashboard.data import (
    ApiUnavailableError,
    ForecastNotReadyError,
    compute_interval_coverage,
    compute_rte_mae,
    fetch_history,
    fetch_latest_forecast,
)

BASE_URL = os.environ.get("FELEC_API_BASE_URL", "http://localhost:8000")
API_HINT = (
    "Can't reach the API at {url} -- is it running? `uv run fastapi dev src/felec/api/app.py`"
)

st.set_page_config(page_title="France Electricity Forecasting", layout="wide")
st.title("France Electricity Forecasting")

st.header("Tomorrow's forecast")
try:
    latest = fetch_latest_forecast(BASE_URL)
    st.plotly_chart(latest_forecast_chart(latest), use_container_width=True)
except ApiUnavailableError:
    st.error(API_HINT.format(url=BASE_URL))
except ForecastNotReadyError:
    st.warning("No forecast yet -- has `ingest predict` been run?")

st.header("Historical performance")
paris_today = datetime.now(UTC).astimezone(ZoneInfo("Europe/Paris")).date()
col1, col2 = st.columns(2)
start = col1.date_input("Start", value=paris_today - timedelta(days=30))
end = col2.date_input("End", value=paris_today)

try:
    history = fetch_history(BASE_URL, start, end)
    if history.empty:
        st.info("No backtest data in this date range.")
    else:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("80% interval coverage", f"{compute_interval_coverage(history):.1%}")
        metric_col2.metric("RTE MAE", f"{compute_rte_mae(history):.0f} MW")
        st.plotly_chart(history_chart(history), use_container_width=True)
except ApiUnavailableError:
    st.error(API_HINT.format(url=BASE_URL))
except ForecastNotReadyError:
    st.warning("No backtest yet -- has `ingest quantile-backtest` been run?")
