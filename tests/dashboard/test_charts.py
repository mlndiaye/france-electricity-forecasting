import pandas as pd

from felec.dashboard.charts import history_chart, latest_forecast_chart


def test_latest_forecast_chart_has_one_trace_per_series():
    df = pd.DataFrame(
        {
            "date_heure": pd.to_datetime(["2026-09-30T00:00:00Z", "2026-09-30T01:00:00Z"]),
            "q10_pred": [100.0, 105.0],
            "q50_pred": [110.0, 115.0],
            "q90_pred": [120.0, 125.0],
            "naive_pred": [108.0, 112.0],
            "rte_pred": [111.0, 116.0],
        }
    )

    fig = latest_forecast_chart(df)

    # q90 (band top), q10 (band bottom, filled), q50, rte_pred = 4 traces
    assert len(fig.data) == 4
    trace_names = {trace.name for trace in fig.data}
    assert "Model (median)" in trace_names
    assert "RTE forecast" in trace_names


def test_history_chart_has_one_trace_per_series():
    df = pd.DataFrame(
        {
            "date_heure": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]),
            "consommation": [100.0, 105.0],
            "rte_pred": [102.0, 107.0],
            "q10_pred": [90.0, 95.0],
            "q90_pred": [110.0, 115.0],
        }
    )

    fig = history_chart(df)

    # q90 (band top), q10 (band bottom, filled), consommation, rte_pred = 4 traces
    assert len(fig.data) == 4
    trace_names = {trace.name for trace in fig.data}
    assert "Actual" in trace_names
    assert "RTE forecast" in trace_names
