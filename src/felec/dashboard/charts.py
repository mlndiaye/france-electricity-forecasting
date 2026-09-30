"""Plotly figure builders for the dashboard -- pure functions (DataFrame in,
Figure out), unit-testable without Streamlit. See ADR 0011.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BAND_FILL_COLOR = "rgba(0, 100, 255, 0.15)"


def _add_interval_band(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.add_trace(
        go.Scatter(x=df["date_heure"], y=df["q90_pred"], line={"width": 0}, showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=df["date_heure"],
            y=df["q10_pred"],
            fill="tonexty",
            fillcolor=BAND_FILL_COLOR,
            line={"width": 0},
            name="80% interval (q10-q90)",
        )
    )


def latest_forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_interval_band(fig, df)
    fig.add_trace(go.Scatter(x=df["date_heure"], y=df["q50_pred"], name="Model (median)"))
    fig.add_trace(go.Scatter(x=df["date_heure"], y=df["rte_pred"], name="RTE forecast"))
    fig.update_layout(xaxis_title="Hour", yaxis_title="Consumption (MW)")
    return fig


def history_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _add_interval_band(fig, df)
    fig.add_trace(go.Scatter(x=df["date_heure"], y=df["consommation"], name="Actual"))
    fig.add_trace(go.Scatter(x=df["date_heure"], y=df["rte_pred"], name="RTE forecast"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Consumption (MW)")
    return fig
