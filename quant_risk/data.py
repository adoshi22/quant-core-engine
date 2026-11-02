"""
quant_risk/data.py
Automated market data ingestion and returns preprocessing.
"""
from typing import Optional
import pandas as pd
import yfinance as yf


def fetch_historical_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetches adjusted historical OHLCV data for an equity or ETF.
    Appends '.NS' automatically if no exchange suffix is provided for Indian stocks.

    Parameters:
    -----------
    symbol : str
        Ticker symbol (e.g., 'NIFTYBEES', 'BAJFINANCE.NS', 'SPY').
    period : str
        Historical timeframe (default: '2y').
    interval : str
        Data frequency (default: '1d').

    Returns:
    --------
    pd.DataFrame with DatetimeIndex and clean Adjusted Close series.
    """
    clean_symbol = symbol.strip().upper()
    if not (clean_symbol.endswith(".NS") or clean_symbol.endswith(".BO") or "^" in clean_symbol):
        clean_symbol = f"{clean_symbol}.NS"

    ticker = yf.Ticker(clean_symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No price data found for symbol '{symbol}'. Verify the ticker symbol.")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_daily_returns(symbol: str, period: str = "2y") -> pd.Series:
    """
    Fetches historical data and returns a clean, daily percentage return series.
    """
    df = fetch_historical_data(symbol, period=period)
    returns = df["Close"].pct_change().dropna()
    returns.name = f"{symbol}_returns"
    return returns