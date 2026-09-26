"""Baseline forecasts, no model, for comparison against the real model."""

from __future__ import annotations

import pandas as pd


def seasonal_naive(features: pd.DataFrame) -> pd.Series:
    """The naive seasonal baseline: predict the value from exactly one week ago.

    Any real model must beat this to be worth using at all.
    """
    return features["lag_168h"]
