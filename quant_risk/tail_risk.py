"""
quant_risk/tail_risk.py
Non-Gaussian tail risk modeling, Cornish-Fisher expansion, semi-deviation, and distribution moments.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def var_historic(returns: pd.Series, level: float = 5.0) -> float:
    """
    Computes Historical (Empirical) Value at Risk at a given confidence level.
    Returns a positive float representing percentage loss (e.g. 0.05 for 5% loss).
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(-np.percentile(clean, level))


def var_gaussian(returns: pd.Series, level: float = 5.0) -> float:
    """
    Computes Parametric Gaussian (Normal) Value at Risk.
    Assumes returns follow an idealized bell curve (Skew = 0, Kurt = 0).
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    z = norm.ppf(level / 100.0)
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    return float(-(mu + z * sigma))


def var_cornish_fisher(returns: pd.Series, level: float = 5.0) -> float:
    """
    Computes Modified Value at Risk using the Cornish-Fisher expansion.
    Adjusts standard Gaussian quantiles using empirical skewness and excess kurtosis.
    """
    clean = returns.dropna()
    if len(clean) < 4:
        return var_gaussian(clean, level=level)

    z = norm.ppf(level / 100.0)
    s = distribution_skewness(clean)
    k = distribution_excess_kurtosis(clean)

    # Cornish-Fisher modified quantile expansion
    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    )

    mu = clean.mean()
    sigma = clean.std(ddof=1)
    return float(-(mu + z_cf * sigma))


def cvar_historic(returns: pd.Series, level: float = 5.0) -> float:
    """
    Computes Historical Conditional Value at Risk (CVaR / Expected Shortfall).
    Measures the average loss beyond the VaR threshold.
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    cutoff = -var_historic(clean, level=level)
    tail_losses = clean[clean <= cutoff]
    if tail_losses.empty:
        return var_historic(clean, level=level)
    return float(-tail_losses.mean())


def semi_deviation(returns: pd.Series) -> float:
    """
    Computes downside semi-deviation (volatility of negative returns only).
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    negative_returns = clean[clean < 0.0]
    if negative_returns.empty:
        return 0.0
    return float(negative_returns.std(ddof=1))


def distribution_skewness(returns: pd.Series) -> float:
    """
    Computes Fisher sample skewness.
    Values < 0 indicate a heavy left tail (crash-prone).
    """
    clean = returns.dropna()
    if len(clean) < 3:
        return 0.0
    return float(skew(clean, bias=False))


def distribution_excess_kurtosis(returns: pd.Series) -> float:
    """
    Computes Fisher excess kurtosis (standard normal = 0.0).
    Values > 0 denote fat tails (leptokurtic).
    """
    clean = returns.dropna()
    if len(clean) < 4:
        return 0.0
    return float(kurtosis(clean, fisher=True, bias=False))