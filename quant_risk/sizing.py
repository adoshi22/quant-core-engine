"""
quant_risk/sizing.py
Systematic volatility-budgeted and ATR-based position sizing.
"""

from typing import Dict
import numpy as np
import pandas as pd


def calculate_atr_position_size(
    account_capital: float,
    risk_budget_pct: float,
    entry_price: float,
    stop_distance_points: float,
) -> Dict[str, float]:
    """Calculates position size using a fixed-risk fraction ($1R$).

    Parameters:
    -----------
    account_capital : float
        Total liquid equity (e.g., 10,00,000 INR).
    risk_budget_pct : float
        Max percentage of account risked on a stop-out (e.g., 0.75 for 0.75%).
    entry_price : float
        Execution entry price per share.
    stop_distance_points : float
        Difference between entry price and stop loss (or ATR multiple).

    Returns:
    --------
    dict with share count, total deployed value, and total portfolio risk %.
    """
    if stop_distance_points <= 0 or entry_price <= 0:
        return {
            "shares": 0,
            "position_value": 0.0,
            "portfolio_allocation_pct": 0.0,
            "max_loss_rupees": 0.0,
        }

    max_loss_rupees = account_capital * (risk_budget_pct / 100.0)
    shares = int(max_loss_rupees // stop_distance_points)
    position_value = float(shares * entry_price)
    portfolio_allocation_pct = (
        float((position_value / account_capital) * 100.0)
        if account_capital > 0
        else 0.0
    )

    return {
        "shares": shares,
        "position_value": position_value,
        "portfolio_allocation_pct": portfolio_allocation_pct,
        "max_loss_rupees": float(shares * stop_distance_points),
    }