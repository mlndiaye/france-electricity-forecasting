from datetime import date
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from felec.modeling.features import cutoff_instant_for_date


def _synthetic_dataset(n_days: int = 20, start: str = "2024-02-01") -> pd.DataFrame:
    """Small synthetic hourly dataset with a deterministic, learnable demand pattern:
    a diurnal cycle (afternoon peak) plus a weekend dip, expressed in Paris local
    time (matching build_calendar_features), then converted to a UTC index -- this
    is the same "date_heure" contract felec.processing.build_dataset produces.

    Built from a local date_range so every local day has exactly 24 hours (start is
    in February, well outside France's DST transition window), which keeps the
    walk-forward backtest's per-day row counts exact and easy to reason about.

    Because the diurnal+weekend pattern repeats exactly every 7 days (same weekday,
    same hour), lag_168h and recent_trend carry real, checkable signal -- not just
    noise -- so a LightGBM model actually has something to learn.
    """
    local_index = pd.date_range(start, periods=n_days * 24, freq="h", tz="Europe/Paris")
    index = local_index.tz_convert("UTC")
    index.name = "date_heure"

    hour = local_index.hour
    day_of_week = local_index.dayofweek

    diurnal = 8_000 * np.sin(2 * np.pi * (hour - 6) / 24)
    weekend_dip = np.where(day_of_week >= 5, -3_000, 0)
    consommation = 50_000 + diurnal + weekend_dip

    return pd.DataFrame(
        {
            "consommation": consommation,
            "prevision_j1": consommation + 1_000,  # a fixed, RTE-like bias
            "temperature_nationale": 10 + 5 * np.sin(2 * np.pi * hour / 24),
            "est_ferie": False,
            "vacances_zone_a": False,
            "vacances_zone_b": False,
            "vacances_zone_c": False,
        },
        index=index,
    )


def test_walk_forward_backtest_respects_the_cutoff_boundary():
    from felec.modeling.backtest import walk_forward_backtest

    df = _synthetic_dataset(n_days=20)
    backtest_start = date(2024, 2, 19)  # last 2 days of the dataset

    captured_indices = []
    original_fit = lgb.LGBMRegressor.fit

    def fit_spy(self, X, y, *args, **kwargs):
        captured_indices.append(X.index)
        return original_fit(self, X, y, *args, **kwargs)

    with patch.object(lgb.LGBMRegressor, "fit", fit_spy):
        walk_forward_backtest(df, backtest_start, cutoff_hour=12)

    assert len(captured_indices) == 2  # one retrain per backtest day (Feb 19, Feb 20)

    expected_days = [date(2024, 2, 19), date(2024, 2, 20)]
    for day, train_index in zip(expected_days, captured_indices, strict=True):
        cutoff = cutoff_instant_for_date(day, cutoff_hour=12)
        assert (train_index < cutoff).all()


def test_walk_forward_backtest_returns_one_row_per_backtest_hour():
    from felec.modeling.backtest import walk_forward_backtest

    df = _synthetic_dataset(n_days=20)
    backtest_start = date(2024, 2, 19)  # last 2 days of the dataset

    result = walk_forward_backtest(df, backtest_start, cutoff_hour=12)

    assert len(result) == 2 * 24
    assert set(result.columns) == {
        "date_heure",
        "consommation",
        "model_pred",
        "naive_pred",
        "rte_pred",
    }


def test_walk_forward_backtest_passes_lgbm_params_through_to_the_model():
    from felec.modeling.backtest import walk_forward_backtest

    df = _synthetic_dataset(n_days=20)
    backtest_start = date(2024, 2, 19)  # last 2 days of the dataset
    custom_params = {
        "num_leaves": 7,
        "learning_rate": 0.2,
        "n_estimators": 5,
        "min_child_samples": 3,
    }

    captured_params = []
    original_fit = lgb.LGBMRegressor.fit

    def fit_spy(self, X, y, *args, **kwargs):
        captured_params.append(self.get_params())
        return original_fit(self, X, y, *args, **kwargs)

    with patch.object(lgb.LGBMRegressor, "fit", fit_spy):
        walk_forward_backtest(df, backtest_start, cutoff_hour=12, lgbm_params=custom_params)

    assert len(captured_params) == 2  # one retrain per backtest day
    for params in captured_params:
        for key, value in custom_params.items():
            assert params[key] == value


def test_walk_forward_backtest_raises_if_backtest_start_before_enough_history():
    from felec.modeling.backtest import walk_forward_backtest

    df = _synthetic_dataset(n_days=20)
    backtest_start = date(2024, 2, 1)  # the dataset's very first day -- no prior history

    with pytest.raises(ValueError, match="not enough training history"):
        walk_forward_backtest(df, backtest_start, cutoff_hour=12)
