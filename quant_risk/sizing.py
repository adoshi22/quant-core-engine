"""
quant_risk/sizing.py
Institutional volatility-budgeted position sizing and Average True Range (ATR) calculations.
"""
from typing import Dict, Union
import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Computes Wilder-smoothed Average True Range (ATR) in points.

    True Range (TR) = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))

    Parameters:
    -----------
    df : pd.DataFrame
        Historical price dataframe containing 'High', 'Low', and 'Close' columns.
    period : int
        ATR smoothing period (default: 14 trading days).

    Returns:
    --------
    float: The latest 14-day ATR value in price points (₹).
    """
    if len(df) < period + 1:
        raise ValueError(f"Dataframe must have at least {period + 1} rows to compute a {period}-day ATR.")

    clean_df = df.dropna(subset=["High", "Low", "Close"])
    high = clean_df["High"]
    low = clean_df["Low"]
    prev_close = clean_df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's exponential smoothing (alpha = 1 / period)
    atr_series = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return float(atr_series.iloc[-1])


def calculate_atr_position_size(
    account_capital: float,
    risk_budget_pct: float,
    entry_price: float,
    stop_distance_points: float
) -> Dict[str, Union[int, float]]:
    """
    Computes exact share sizing and capital allocation under a strict 1R risk budget.

    Formula:
    --------
    1R Cash Risk (₹) = account_capital * (risk_budget_pct / 100)
    Shares = floor(1R Cash Risk / stop_distance_points)

    Parameters:
    -----------
    account_capital : float
        Total portfolio capital in ₹.
    risk_budget_pct : float
        Percentage of total capital risked per trade (e.g., 0.75 for 0.75%).
    entry_price : float
        Current market price or limit order execution price in ₹.
    stop_distance_points : float
        Absolute price distance from entry to stop loss in ₹.

    Returns:
    --------
    dict with shares, position_value, portfolio_allocation_pct, and max_loss_rupees.
    """
    if account_capital <= 0 or entry_price <= 0:
        return {
            "shares": 0,
            "position_value": 0.0,
            "portfolio_allocation_pct": 0.0,
            "max_loss_rupees": 0.0,
        }

    risk_budget_rupees = account_capital * (risk_budget_pct / 100.0)

    if stop_distance_points <= 0:
        return {
            "shares": 0,
            "position_value": 0.0,
            "portfolio_allocation_pct": 0.0,
            "max_loss_rupees": 0.0,
        }

    shares = int(np.floor(risk_budget_rupees / stop_distance_points))
    position_value = float(shares * entry_price)
    portfolio_allocation_pct = float((position_value / account_capital) * 100.0)
    max_loss = float(shares * stop_distance_points)

    return {
        "shares": shares,
        "position_value": position_value,
        "portfolio_allocation_pct": portfolio_allocation_pct,
        "max_loss_rupees": max_loss,
    }