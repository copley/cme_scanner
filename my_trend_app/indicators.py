from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange


def compute_indicators(df: pd.DataFrame, lookback: int = 100) -> dict | None:
    """Compute ADX, percent above 20MA, and R² of returns vs ATR."""
    if len(df) < lookback:
        return None

    df = df.sort_values("date").reset_index(drop=True)

    adx_indicator = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        fillna=False,
    )
    df["ADX_14"] = adx_indicator.adx()

    df["MA_20"] = df["close"].rolling(20).mean()
    recent_df_20 = df.tail(20)
    pct_above_ma = 100.0 * np.sum(recent_df_20["close"] > recent_df_20["MA_20"]) / 20.0

    df["returns"] = df["close"].pct_change()
    atr_indicator = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        fillna=False,
    )
    df["ATR_14"] = atr_indicator.average_true_range()

    lookback_df = df.tail(lookback).dropna(subset=["returns", "ATR_14"])
    if len(lookback_df) < 2:
        r2_value = np.nan
    else:
        x = lookback_df["ATR_14"].values.reshape(-1, 1)
        y = lookback_df["returns"].values
        lr = LinearRegression()
        lr.fit(x, y)
        y_pred = lr.predict(x)
        r2_value = r2_score(y, y_pred)

    adx_latest = df["ADX_14"].iloc[-1]
    return {
        "ADX": float(adx_latest),
        "%Above20MA": float(pct_above_ma),
        "R2": float(r2_value),
    }
