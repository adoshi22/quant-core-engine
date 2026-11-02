"""
quant_risk/drawdowns.py
Vectorized drawdown, peak tracking, and recovery duration analysis.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def compute_drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """Computes the wealth index, previous peaks, and drawdown series.

    Parameters:
    -----------
    returns : pd.Series
        Time-series of periodic percentage returns (e.g. daily returns).

    Returns:
    --------
    pd.DataFrame containing:
        - 'wealth_index': Compounded portfolio growth starting at 1000.
        - 'previous_peaks': Running historical high of the wealth index.
        - 'drawdown': Percentage decline from the previous peak (values <= 0).
    """
    clean_returns = returns.dropna()
    wealth_index = 1000.0 * (1.0 + clean_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame(
        {
            "wealth_index": wealth_index,
            "previous_peaks": previous_peaks,
            "drawdown": drawdown,
        }
    )


def max_drawdown(returns: pd.Series) -> float:
    """Calculates the absolute maximum drawdown over the entire period."""
    dd_df = compute_drawdown_series(returns)
    return float(dd_df["drawdown"].min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Calculates the maximum number of periods the asset spent below its peak

    (drawdown duration / recovery lag in days).
    """
    dd = compute_drawdown_series(returns)["drawdown"]
    is_zero = dd == 0

    # Group continuous periods where wealth is below previous peak
    periods = (~is_zero).astype(int)
    streaks = periods.groupby(
        (is_zero != is_zero.shift()).cumsum()
    ).cumsum() * periods

    return int(streaks.max()) if not streaks.empty else 0