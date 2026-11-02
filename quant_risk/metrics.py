"""
quant_risk/metrics.py
Annualized returns, volatility, and risk-adjusted ratios (Sharpe, Sortino, Calmar).
"""

import numpy as np
import pandas as pd
from quant_risk.drawdowns import max_drawdown
from quant_risk.tail_risk import semi_deviation


def annualized_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Computes Compound Annual Growth Rate (CAGR)."""
    clean = returns.dropna()
    compounded_growth = (1.0 + clean).prod()
    n_periods = len(clean)
    if n_periods == 0:
        return 0.0
    return float(compounded_growth ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Computes annualized standard deviation."""
    clean = returns.dropna()
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """Computes annualized Sharpe Ratio against a benchmark risk-free rate."""
    cagr = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float((cagr - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """Computes Sortino Ratio, penalizing only downside volatility."""
    cagr = annualized_return(returns, periods_per_year)
    downside_vol = semi_deviation(returns) * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    return float((cagr - risk_free_rate) / downside_vol)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes Calmar Ratio: Annualized Return divided by absolute Max Drawdown."""
    cagr = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)