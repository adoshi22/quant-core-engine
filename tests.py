"""
tests/test_risk.py
Comprehensive PyTest suite validating quantitative risk formulas, moments, and ATR sizing.
"""
import numpy as np
import pandas as pd
import pytest

from quant_risk.drawdowns import compute_drawdown_series, max_drawdown_duration
from quant_risk.tail_risk import (
    var_historic,
    var_cornish_fisher,
    cvar_historic,
    semi_deviation,
    distribution_skewness,
    distribution_excess_kurtosis,
)
from quant_risk.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)
from quant_risk.sizing import calculate_atr_position_size, calculate_atr


def test_drawdown_logic():
    """Validates peak tracking, maximum drawdown depth, and recovery durations."""
    returns = pd.Series([0.10, -0.20, 0.05, 0.15])
    wealth, peaks, dd = compute_drawdown_series(returns, initial_capital=1000.0)

    assert wealth.iloc[0] == pytest.approx(1100.0, rel=1e-3)
    assert wealth.iloc[1] == pytest.approx(880.0, rel=1e-3)
    assert peaks.iloc[1] == pytest.approx(1100.0, rel=1e-3)
    assert dd.iloc[1] == pytest.approx(-0.20, rel=1e-3)

    dd_duration = max_drawdown_duration(returns)
    assert dd_duration >= 2


def test_tail_risk_skew_adjustment():
    """Validates Cornish-Fisher expansion vs historical percentile under negative skew."""
    np.random.seed(42)
    # Generate negatively skewed return distribution (sudden crash events)
    base = np.random.normal(0.001, 0.015, 1000)
    crashes = np.random.choice([0.0, -0.06], size=1000, p=[0.95, 0.05])
    skewed_returns = pd.Series(base + crashes)

    hist_var = var_historic(skewed_returns, level=5.0)
    cf_var = var_cornish_fisher(skewed_returns, level=5.0)
    cvar = cvar_historic(skewed_returns, level=5.0)

    assert cf_var > hist_var, "Cornish-Fisher VaR must exceed Gaussian VaR on fat-tailed crash series"
    assert cvar >= hist_var, "CVaR must be greater than or equal to VaR"


def test_distribution_moments_gaussian():
    """Validates that a Gaussian distribution has Skewness ≈ 0.0 and Excess Kurtosis ≈ 0.0."""
    np.random.seed(42)
    normal_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.015, size=25000))

    skew_val = distribution_skewness(normal_returns)
    kurt_val = distribution_excess_kurtosis(normal_returns)

    assert abs(skew_val) < 0.1, f"Expected normal skewness near 0.0, got {skew_val}"
    assert abs(kurt_val) < 0.1, f"Expected normal excess kurtosis near 0.0, got {kurt_val}"


def test_distribution_skewness_asymmetry():
    """Validates directional sensitivity of skewness on known asymmetric distributions."""
    np.random.seed(42)
    right_skew = pd.Series(np.random.exponential(scale=0.02, size=20000))
    left_skew = -right_skew

    assert distribution_skewness(right_skew) > 1.5
    assert distribution_skewness(left_skew) < -1.5


def test_distribution_excess_kurtosis_tails():
    """Validates tail-heaviness: Student-t (leptokurtic > 0) vs. Uniform (platykurtic < 0)."""
    np.random.seed(42)
    fat_tailed = pd.Series(np.random.standard_t(df=5, size=25000))
    thin_tailed = pd.Series(np.random.uniform(low=-0.05, high=0.05, size=25000))

    assert distribution_excess_kurtosis(fat_tailed) > 2.0
    assert distribution_excess_kurtosis(thin_tailed) < -1.0


def test_distribution_moments_edge_cases():
    """Ensures fallback to 0.0 when series is empty or has insufficient sample size."""
    empty_series = pd.Series([], dtype=float)
    short_series = pd.Series([0.01, -0.02])

    assert distribution_skewness(empty_series) == 0.0
    assert distribution_skewness(short_series) == 0.0
    assert distribution_excess_kurtosis(empty_series) == 0.0
    assert distribution_excess_kurtosis(short_series) == 0.0


def test_calculate_atr_known_values():
    """Validates 14-day Wilder ATR on known static price range."""
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({
        "High": [105.0] * 30,
        "Low": [95.0] * 30,
        "Close": [100.0] * 30,
    }, index=dates)

    atr = calculate_atr(mock_df, period=14)
    assert abs(atr - 10.0) < 1e-4


def test_ratios_and_sizing():
    """Validates performance ratios and volatility-budgeted 1R capital allocation."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    cagr = annualized_return(returns)
    vol = annualized_volatility(returns)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.065)
    sortino = sortino_ratio(returns, risk_free_rate=0.065)

    assert isinstance(cagr, float)
    assert isinstance(vol, float)
    assert isinstance(sharpe, float)
    assert isinstance(sortino, float)

    # 1R Position sizing test: ₹10L capital, 1% risk (₹10,000 risk), ₹50 stop distance
    sizing = calculate_atr_position_size(
        account_capital=1000000.0,
        risk_budget_pct=1.0,
        entry_price=1000.0,
        stop_distance_points=50.0,
    )
    assert sizing["shares"] == 200
    assert sizing["position_value"] == 200000.0
    assert sizing["portfolio_allocation_pct"] == pytest.approx(20.0, rel=1e-2)
    assert sizing["max_loss_rupees"] == 10000.0