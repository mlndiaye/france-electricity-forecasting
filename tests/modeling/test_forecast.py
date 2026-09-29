from datetime import date
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from felec.modeling.features import cutoff_instant_for_date
from felec.modeling.forecast import predict_next_day


def _synthetic_dataset(n_days: int = 20, start: str = "2024-02-01") -> pd.DataFrame:
    """Same synthetic dataset shape as tests/modeling/test_backtest.py's
    helper -- a deterministic, learnable demand pattern with all the columns
    build_features/build_calendar_features need.
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


def test_predict_next_day_respects_the_cutoff_boundary():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 2, 20)  # the dataset's last day

    captured_indices = []
    original_fit = lgb.LGBMRegressor.fit

    def fit_spy(self, X, y, *args, **kwargs):
        captured_indices.append(X.index)
        return original_fit(self, X, y, *args, **kwargs)

    with patch.object(lgb.LGBMRegressor, "fit", fit_spy):
        predict_next_day(df, target_date, cutoff_hour=12, quantiles=(0.1, 0.5, 0.9))

    assert len(captured_indices) == 3  # one retrain per quantile
    cutoff = cutoff_instant_for_date(target_date, cutoff_hour=12)
    for train_index in captured_indices:
        assert (train_index < cutoff).all()


def test_predict_next_day_returns_one_row_per_target_day_hour_with_quantile_columns():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 2, 20)

    result = predict_next_day(df, target_date, cutoff_hour=12, quantiles=(0.1, 0.5, 0.9))

    assert len(result.predictions) == 24
    assert set(result.predictions.columns) == {
        "date_heure",
        "q10_pred",
        "q50_pred",
        "q90_pred",
        "naive_pred",
        "rte_pred",
    }


def test_predict_next_day_sorts_crossed_quantile_predictions():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 2, 20)

    original_predict = lgb.LGBMRegressor.predict
    call_count = {"n": 0}

    def predict_spy(self, X, *args, **kwargs):
        result = original_predict(self, X, *args, **kwargs)
        call_count["n"] += 1
        # Force the q0.1 model's predictions artificially far above the
        # others', simulating real-world quantile crossing regardless of what
        # these particular models would naturally have predicted.
        if call_count["n"] == 1:  # sorted_quantiles processes 0.1 first
            return result + 100_000
        return result

    with patch.object(lgb.LGBMRegressor, "predict", predict_spy):
        result = predict_next_day(df, target_date, cutoff_hour=12, quantiles=(0.1, 0.5, 0.9))

    assert call_count["n"] == 3
    assert (result.predictions["q10_pred"] <= result.predictions["q50_pred"]).all()
    assert (result.predictions["q50_pred"] <= result.predictions["q90_pred"]).all()


def test_predict_next_day_passes_lgbm_params_through_to_the_model():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 2, 20)  # the dataset's last day
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
        predict_next_day(
            df, target_date, cutoff_hour=12, quantiles=(0.1, 0.5, 0.9), lgbm_params=custom_params
        )

    assert len(captured_params) == 3  # one per quantile
    for params in captured_params:
        for key, value in custom_params.items():
            assert params[key] == value


def test_predict_next_day_raises_if_not_enough_history():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 2, 1)  # the dataset's very first day -- no prior history

    with pytest.raises(ValueError, match="not enough training history"):
        predict_next_day(df, target_date, cutoff_hour=12)


def test_predict_next_day_raises_if_target_date_not_in_dataset():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 3, 1)  # well past the 20-day synthetic dataset

    with pytest.raises(ValueError, match="no rows found"):
        predict_next_day(df, target_date, cutoff_hour=12)


def test_predict_next_day_returns_the_trained_models_and_actual_params_used():
    df = _synthetic_dataset(n_days=20)
    target_date = date(2024, 2, 20)
    custom_params = {
        "num_leaves": 7,
        "learning_rate": 0.2,
        "n_estimators": 5,
        "min_child_samples": 3,
    }

    result = predict_next_day(
        df, target_date, cutoff_hour=12, quantiles=(0.1, 0.5, 0.9), lgbm_params=custom_params
    )

    assert set(result.models.keys()) == {0.1, 0.5, 0.9}
    for model in result.models.values():
        assert isinstance(model, lgb.LGBMRegressor)
    assert result.params == custom_params
    assert result.cutoff_hour == 12
    assert result.n_train_rows > 0
