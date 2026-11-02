"""
quant_risk/tail_risk.py
Advanced downside and tail-risk analytics: Semi-deviation, Cornish-Fisher VaR, and CVaR.
"""

from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

def distribution_skewness(returns: pd.Series) -> float:
    """Calculates Fisher sample skewness.

    Negative values denote a left tail with crash vulnerability.
    """
    clean = returns.dropna()
    if len(clean) < 3:
        return 0.0
    return float(skew(clean, bias=False))

def distribution_excess_kurtosis(returns: pd.Series) -> float:
    """Calculates Fisher excess kurtosis (normal distribution = 0.0).

    Values > 0 denote leptokurtic / fat-tailed distributions.
    """
    clean = returns.dropna()
    if len(clean) < 4:
        return 0.0
    return float(kurtosis(clean, fisher=True, bias=False))

def semi_deviation(returns: pd.Series) -> float:
    """Calculates downside volatility (semi-deviation / downside deviation)

    by replacing positive excess returns with zero and averaging over all N
    periods.
    """
    clean = returns.dropna()
    excess = clean - clean.mean()
    # Replace positive excess with 0, then take the root-mean-square over all observations
    downside_excess = np.minimum(excess, 0.0)
    return float(np.sqrt((downside_excess**2).mean()))


def var_historic(returns: pd.Series, level: float = 5.0) -> float:
    """Computes Historical Value-at-Risk at a given percentile (default: 5% / 95% confidence).

    Returns a positive number representing the percentage loss.
    """
    clean = returns.dropna()
    return float(-np.percentile(clean, level))


def var_cornish_fisher(returns: pd.Series, level: float = 5.0) -> float:
    """Computes Modified Value-at-Risk using the Cornish-Fisher expansion,

    correcting for empirical skewness and excess kurtosis.
    """
    clean = returns.dropna()
    z = norm.ppf(level / 100.0)

    # Higher moments
    skew = clean.skew()
    kurt = clean.kurtosis()  # Excess kurtosis (Fisher definition)

    # Cornish-Fisher adjusted Z-score
    z_cf = (
        z
        + (z**2 - 1) * skew / 6.0
        + (z**3 - 3 * z) * kurt / 24.0
        - (2 * z**3 - 5 * z) * (skew**2) / 36.0
    )

    modified_var = -(clean.mean() + z_cf * clean.std(ddof=1))
    return float(modified_var)


def cvar_historic(returns: pd.Series, level: float = 5.0) -> float:
    """Calculates Conditional Value-at-Risk (CVaR / Expected Shortfall),

    measuring the expected average loss beyond the VaR threshold.
    """
    clean = returns.dropna()
    cutoff = np.percentile(clean, level)
    tail_losses = clean[clean <= cutoff]
    return float(-tail_losses.mean())