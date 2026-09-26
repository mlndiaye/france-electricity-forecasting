import pandas as pd

from felec.modeling.baselines import seasonal_naive


def test_seasonal_naive_returns_the_lag_168h_column_unchanged():
    features = pd.DataFrame({"lag_168h": [100.0, 200.0, None], "hour": [0, 1, 2]})

    result = seasonal_naive(features)

    assert result.tolist()[:2] == [100.0, 200.0]
    assert pd.isna(result.iloc[2])
