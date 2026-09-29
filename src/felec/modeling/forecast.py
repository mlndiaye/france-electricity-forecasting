"""Predict a single future day. Unlike backtest.py's evaluation-focused
functions, there is no ground truth to compare against -- see ADR 0007.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from felec.modeling.backtest import DEFAULT_LGBM_PARAMS, sorted_quantile_columns
from felec.modeling.baselines import seasonal_naive
from felec.modeling.features import build_features, cutoff_instant_for_date


@dataclass
class ForecastResult:
    """Everything predict_next_day() produced -- not just the predictions
    themselves, but also what it takes to audit them later (the actual
    hyperparameters and cutoff used, the trained models, how much data went
    in). See ADR 0009.
    """

    predictions: pd.DataFrame
    models: dict[float, lgb.LGBMRegressor]
    params: dict
    cutoff_hour: int
    n_train_rows: int


def predict_next_day(
    df: pd.DataFrame,
    target_date: date,
    cutoff_hour: int = 12,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    lgbm_params: dict | None = None,
) -> ForecastResult:
    """Train one model per quantile on every row strictly before target_date's
    D-1 cutoff, predict target_date's 24 hours. predictions has date_heure, one
    column per quantile (e.g. q10_pred, q50_pred, q90_pred), naive_pred, and
    rte_pred -- no consommation column, since target_date hasn't happened yet.
    """
    params = lgbm_params if lgbm_params is not None else DEFAULT_LGBM_PARAMS
    features = build_features(df, cutoff_hour=cutoff_hour)
    naive_pred_all = seasonal_naive(features)
    sorted_quantiles = sorted(quantiles)

    cutoff = cutoff_instant_for_date(target_date, cutoff_hour)
    if cutoff <= df.index.min():
        raise ValueError(
            f"not enough training history before {target_date}: "
            f"cutoff {cutoff} is at or before the dataset's earliest "
            f"timestamp {df.index.min()}"
        )
    train_mask = (df.index < cutoff) & df["consommation"].notna()
    train_features = features.loc[train_mask]
    train_target = df.loc[train_mask, "consommation"]

    local_dates = df.index.tz_convert("Europe/Paris").normalize()
    day_mask = local_dates == pd.Timestamp(target_date, tz="Europe/Paris")
    if not day_mask.any():
        raise ValueError(
            f"no rows found for {target_date} in the dataset -- "
            "has build-dataset been run with weather/RTE data through this date?"
        )
    day_features = features.loc[day_mask]

    predictions: dict[float, np.ndarray] = {}
    models: dict[float, lgb.LGBMRegressor] = {}
    for q in sorted_quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, random_state=0, verbosity=-1, **params
        )
        model.fit(train_features, train_target)
        predictions[q] = model.predict(day_features)
        models[q] = model

    result: dict[str, object] = {"date_heure": df.index[day_mask]}
    result.update(sorted_quantile_columns(predictions, sorted_quantiles))
    result["naive_pred"] = naive_pred_all.loc[day_mask].to_numpy()
    result["rte_pred"] = df.loc[day_mask, "prevision_j1"].to_numpy()

    return ForecastResult(
        predictions=pd.DataFrame(result),
        models=models,
        params=params,
        cutoff_hour=cutoff_hour,
        n_train_rows=int(train_mask.sum()),
    )
