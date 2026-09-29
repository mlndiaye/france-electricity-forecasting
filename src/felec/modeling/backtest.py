"""Daily walk-forward backtest: retrain on an expanding window, predict one day
ahead, repeat. Mirrors the v2 production design (see ADR 0004) rather than a
single train/test split.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from felec.modeling.baselines import seasonal_naive
from felec.modeling.features import build_features, cutoff_instant_for_date

# Tuned via grid search against a held-out validation slice (2025-08-16 to
# 2025-10-15), see notebooks/model_selection.ipynb and ADR 0004.
DEFAULT_LGBM_PARAMS = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 10,
}


def _iter_backtest_days(
    df: pd.DataFrame,
    features: pd.DataFrame,
    backtest_start: date,
    cutoff_hour: int,
) -> Iterator[tuple[date, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Yield, for each backtest day, the training data strictly before that
    day's D-1 cutoff and that day's own feature rows. Shared by
    walk_forward_backtest and quantile_backtest so both retrain on identical
    per-day splits -- see ADR 0006.
    """
    local_dates = df.index.tz_convert("Europe/Paris").normalize()
    last_date = local_dates.unique().max().date()

    first_cutoff = cutoff_instant_for_date(backtest_start, cutoff_hour)
    if first_cutoff <= df.index.min():
        raise ValueError(
            f"not enough training history before {backtest_start}: "
            f"first cutoff {first_cutoff} is at or before the dataset's earliest "
            f"timestamp {df.index.min()}"
        )

    current = backtest_start
    while current <= last_date:
        cutoff = cutoff_instant_for_date(current, cutoff_hour)
        train_mask = (df.index < cutoff) & df["consommation"].notna()
        train_features = features.loc[train_mask]
        train_target = df.loc[train_mask, "consommation"]

        day_mask = local_dates == pd.Timestamp(current, tz="Europe/Paris")
        day_features = features.loc[day_mask]

        yield current, train_features, train_target, day_features, day_mask
        current = current + timedelta(days=1)


def walk_forward_backtest(
    df: pd.DataFrame,
    backtest_start: date,
    cutoff_hour: int = 12,
    lgbm_params: dict | None = None,
) -> pd.DataFrame:
    """Retrain daily on every row strictly before each day's D-1 cutoff, predict
    that day's 24 hours, and record model/naive/RTE predictions alongside the
    actual consumption. Returns one row per backtest hour.
    """
    params = lgbm_params if lgbm_params is not None else DEFAULT_LGBM_PARAMS
    features = build_features(df, cutoff_hour=cutoff_hour)
    naive_pred_all = seasonal_naive(features)

    rows = []
    for current, train_features, train_target, day_features, day_mask in _iter_backtest_days(
        df, features, backtest_start, cutoff_hour
    ):
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, random_state=0, verbosity=-1, **params
        )
        model.fit(train_features, train_target)
        model_pred = model.predict(day_features)

        rows.append(
            pd.DataFrame(
                {
                    "date_heure": df.index[day_mask],
                    "consommation": df.loc[day_mask, "consommation"].to_numpy(),
                    "model_pred": model_pred,
                    "naive_pred": naive_pred_all.loc[day_mask].to_numpy(),
                    "rte_pred": df.loc[day_mask, "prevision_j1"].to_numpy(),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def sorted_quantile_columns(
    predictions: dict[float, np.ndarray], sorted_quantiles: list[float]
) -> dict[str, np.ndarray]:
    """Sort each row's predictions across quantile levels before assigning
    them to their named columns (e.g. q10_pred, q90_pred), so q10 <= q50 <=
    q90 holds by construction even though each quantile was trained as an
    independent model -- the standard fix for "quantile crossing"
    (Chernozhukov et al. 2010). Shared by quantile_backtest and
    forecast.predict_next_day so the fix lives in one place -- see ADR 0006
    and ADR 0007.
    """
    stacked = np.sort(np.column_stack([predictions[q] for q in sorted_quantiles]), axis=1)
    return {f"q{round(q * 100)}_pred": stacked[:, i] for i, q in enumerate(sorted_quantiles)}


def quantile_backtest(
    df: pd.DataFrame,
    backtest_start: date,
    quantiles: tuple[float, ...] = (0.1, 0.9),
    cutoff_hour: int = 12,
    lgbm_params: dict | None = None,
) -> pd.DataFrame:
    """Retrain daily, once per requested quantile, predicting that day's 24
    hours at each quantile level. Returns one row per backtest hour, with one
    column per quantile (e.g. q10_pred, q90_pred). Shares the same
    walk-forward mechanics as walk_forward_backtest -- see ADR 0006.

    Each quantile is trained as an independent model, so nothing guarantees
    their raw predictions come out in order -- see sorted_quantile_columns.
    """
    params = lgbm_params if lgbm_params is not None else DEFAULT_LGBM_PARAMS
    features = build_features(df, cutoff_hour=cutoff_hour)
    sorted_quantiles = sorted(quantiles)

    rows = []
    for current, train_features, train_target, day_features, day_mask in _iter_backtest_days(
        df, features, backtest_start, cutoff_hour
    ):
        predictions: dict[float, np.ndarray] = {}
        for q in sorted_quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q, random_state=0, verbosity=-1, **params
            )
            model.fit(train_features, train_target)
            predictions[q] = model.predict(day_features)

        day_row: dict[str, object] = {"date_heure": df.index[day_mask]}
        day_row.update(sorted_quantile_columns(predictions, sorted_quantiles))

        rows.append(pd.DataFrame(day_row))

    return pd.concat(rows, ignore_index=True)
