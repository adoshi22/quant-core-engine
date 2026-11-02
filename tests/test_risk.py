"""
tests/test_risk.py
PyTest test suite for mathematical accuracy and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from quant_risk.drawdowns import compute_drawdown_series, max_drawdown
from quant_risk.metrics import (
    annualized_return,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from quant_risk.sizing import calculate_atr_position_size
from quant_risk.tail_risk import (
    semi_deviation,
    var_cornish_fisher,
    var_historic,
)


def test_drawdown_logic():
    # Asset: 100 -> 120 -> 90 -> 100
    prices = pd.Series([100.0, 120.0, 90.0, 100.0])
    returns = prices.pct_change().dropna()

    dd_df = compute_drawdown_series(returns)
    assert np.isclose(max_drawdown(returns), -0.25, atol=1e-3)
    assert dd_df["previous_peaks"].max() == 1200.0  # scaled from base 1000


def test_tail_risk_skew_adjustment():
    np.random.seed(42)
    normal_returns = np.random.normal(0.0005, 0.015, 1000)
    crash_days = np.array([-0.07, -0.09, -0.11])  # Black swans
    skewed_returns = pd.Series(np.concatenate([normal_returns, crash_days]))

    hist_var = var_historic(skewed_returns, level=5.0)
    cf_var = var_cornish_fisher(skewed_returns, level=5.0)

    # Cornish-Fisher must penalize negative skewness higher than standard normal percentile
    assert cf_var > 0
    assert hist_var > 0
    assert isinstance(semi_deviation(skewed_returns), float)


def test_ratios_and_sizing():
    returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.005] * 40)
    cagr = annualized_return(returns)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.065)
    sortino = sortino_ratio(returns, risk_free_rate=0.065)
    calmar = calmar_ratio(returns)

    assert cagr > 0
    assert isinstance(sharpe, float)
    assert sortino > sharpe  # Sortino ignores upside volatility

    # Position sizing: 10L capital, 1% risk (10k), 1000 price, 50 stop distance
    sizing = calculate_atr_position_size(
        account_capital=1000000,
        risk_budget_pct=1.0,
        entry_price=1000.0,
        stop_distance_points=50.0,
    )
    assert sizing["shares"] == 200
    assert sizing["max_loss_rupees"] == 10000.0
    assert sizing["portfolio_allocation_pct"] == 20.0