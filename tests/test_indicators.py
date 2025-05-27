import pandas as pd
from my_trend_app.indicators import compute_indicators


def test_compute_indicators_basic():
    data = {
        'date': pd.date_range('2024-01-01', periods=120, freq='D'),
        'open': pd.Series(range(120), dtype=float),
        'high': pd.Series(range(1, 121), dtype=float),
        'low': pd.Series(range(120), dtype=float),
        'close': pd.Series(range(1, 121), dtype=float),
    }
    df = pd.DataFrame(data)
    result = compute_indicators(df, lookback=100)
    assert set(result.keys()) == {'ADX', '%Above20MA', 'R2'}
