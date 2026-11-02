"""
app.py
Production Streamlit Risk & Portfolio Engine Dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from quant_risk.data import fetch_historical_data
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

st.set_page_config(
    page_title="Quant Core Risk Engine",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Institutional Risk & Tail Analytics Engine")
st.markdown("""
*Vectorized risk analytics, non-Gaussian tail modeling (Cornish-Fisher), underwater duration profiling, and 14-day ATR volatility-budgeted execution.*
""")

# --- Sidebar Configuration ---
st.sidebar.header("🕹️ Engine Controls")
data_mode = st.sidebar.radio("Data Source", ["Live NSE / Global Equity", "Synthetic Test Generator"])

if data_mode == "Live NSE / Global Equity":
    ticker_input = st.sidebar.text_input("Enter NSE/BSE Ticker (e.g., NIFTYBEES, GOLDBEES, KMEW, BAJFINANCE)", value="NIFTYBEES")
    time_frame = st.sidebar.selectbox("History Period", ["1y", "2y", "3y", "5y"], index=1)

    try:
        with st.spinner(f"Ingesting live historical data for {ticker_input}..."):
            hist_df = fetch_historical_data(ticker_input, period=time_frame)
            returns_series = hist_df["Close"].pct_change().dropna()
            current_price = float(hist_df["Close"].iloc[-1])
            asset_name = ticker_input.upper()
    except Exception as e:
        st.sidebar.error(f"Error fetching {ticker_input}: {str(e)}")
        st.stop()
else:
    sim_days = st.sidebar.slider("Simulation Length (Trading Days)", 252, 1260, 504)
    annual_mu = st.sidebar.slider("Annual Mean Drift (%)", -10.0, 40.0, 18.0) / 100.0
    annual_sig = st.sidebar.slider("Annual Volatility (%)", 5.0, 50.0, 22.0) / 100.0
    crash_prob = st.sidebar.slider("Fat-Tail Crash Frequency (%)", 0.0, 5.0, 1.5) / 100.0

    np.random.seed(42)
    dt = 1.0 / 252.0
    base_ret = np.random.normal((annual_mu - 0.5 * annual_sig**2) * dt, annual_sig * np.sqrt(dt), sim_days)
    shocks = np.random.choice([0.0, -0.05], size=sim_days, p=[1.0 - crash_prob, crash_prob])
    sim_returns = base_ret + shocks

    dates = pd.date_range(end=datetime.today(), periods=sim_days, freq="B")
    returns_series = pd.Series(sim_returns, index=dates)
    price_levels = 1000.0 * (1.0 + returns_series).cumprod()
    hist_df = pd.DataFrame({
        "High": price_levels * 1.01,
        "Low": price_levels * 0.99,
        "Close": price_levels,
    }, index=dates)
    current_price = float(price_levels.iloc[-1])
    asset_name = "SYNTHETIC_ASSET"

# --- Main Analytical Dashboard ---
wealth, peaks, drawdowns = compute_drawdown_series(returns_series)

# Top Performance Metrics
st.subheader(f"📊 Performance Overview: {asset_name}")
col1, col2, col3, col4, col5 = st.columns(5)

cagr_val = annualized_return(returns_series) * 100.0
ann_vol = annualized_volatility(returns_series) * 100.0
mdd_val = drawdowns.min() * 100.0
sharpe_val = sharpe_ratio(returns_series, risk_free_rate=0.065)
sortino_val = sortino_ratio(returns_series, risk_free_rate=0.065)

col1.metric("CAGR", f"{cagr_val:.2f}%")
col2.metric("Annualized Volatility", f"{ann_vol:.2f}%")
col3.metric("Max Drawdown", f"{mdd_val:.2f}%")
col4.metric("Sharpe Ratio (Rf=6.5%)", f"{sharpe_val:.2f}")
col5.metric(
    "Sortino Ratio",
    f"{sortino_val:.2f}",
    help="Uses downside semi-deviation. Sortino > Sharpe proves volatility is upside expansion rather than crash shocks."
)

# Interactive Charts
fig = go.Figure()
fig.add_trace(go.Scatter(x=wealth.index, y=wealth, mode="lines", name="Wealth Index (₹1,000 Base)", line=dict(color="#00CC96", width=2)))
fig.add_trace(go.Scatter(x=peaks.index, y=peaks, mode="lines", name="High-Water Mark (Peak)", line=dict(color="#AB63FA", dash="dot", width=1.5)))
fig.update_layout(title="Running Wealth Index vs. Peak Progression", xaxis_title="Date", yaxis_title="Portfolio Capital Value (₹)", height=400, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns * 100.0, mode="lines", fill="tozeroy", name="Drawdown %", line=dict(color="#EF553B", width=1.5)))
fig_dd.update_layout(title="Underwater Drawdown Profile", xaxis_title="Date", yaxis_title="Drawdown (%)", height=300, template="plotly_white")
st.plotly_chart(fig_dd, use_container_width=True)

# Tail Risk & Non-Gaussian Moments
st.subheader("⚠️ Tail Risk & Non-Gaussian Distribution Moments")
st.caption("Distribution asymmetry: Normal distribution has Skewness = 0.0 and Excess Kurtosis = 0.0.")

tcol1, tcol2, tcol3, tcol4, tcol5, tcol6 = st.columns(6)

cf_var_val = var_cornish_fisher(returns_series, level=5.0) * 100.0
hist_var_val = var_historic(returns_series, level=5.0) * 100.0
cvar_val = cvar_historic(returns_series, level=5.0) * 100.0
longest_dd = max_drawdown_duration(returns_series)
skew_val = distribution_skewness(returns_series)
kurt_val = distribution_excess_kurtosis(returns_series)

tcol1.metric("95% Cornish-Fisher VaR", f"{cf_var_val:.2f}%")
tcol2.metric("95% Historical VaR", f"{hist_var_val:.2f}%")
tcol3.metric("95% CVaR (Expected Loss)", f"{cvar_val:.2f}%")
tcol4.metric("Skewness", f"{skew_val:.2f}", help="Negative skew indicates vulnerability to sudden gap-down selloffs.")
tcol5.metric("Excess Kurtosis", f"{kurt_val:.2f}", help="Values > 0 denote fat tails (leptokurtic). Higher probability of extreme 3+ sigma shocks.")
tcol6.metric("Max DD Duration", f"{longest_dd} Days")

# Volatility-Budgeted Position Sizing
st.subheader("🎯 Volatility-Budgeted Execution Calculator (14-Day ATR)")
st.caption("Automatically sets stop-loss buffer based on market volatility rather than arbitrary guessing.")

try:
    auto_atr = calculate_atr(hist_df, period=14)
except Exception:
    auto_atr = float(current_price * 0.025)

p1, p2, p3, p4 = st.columns(4)

total_capital = p1.number_input("Portfolio Capital (₹)", value=1000000, step=50000)
risk_pct = p2.slider("Risk Budget (1R %)", min_value=0.25, max_value=2.00, value=0.75, step=0.05)
stock_price = p3.number_input("Execution Price (₹)", value=float(round(current_price, 2)), step=1.0)

atr_multiplier = p4.selectbox(
    "Stop Distance Buffer",
    options=[1.0, 1.5, 2.0, 2.5, 3.0],
    index=1,
    format_func=lambda x: f"{x}x ATR (₹{x * auto_atr:,.2f})"
)

stop_points = float(round(atr_multiplier * auto_atr, 2))
stop_loss_price = max(0.0, stock_price - stop_points)

sizing = calculate_atr_position_size(
    account_capital=total_capital,
    risk_budget_pct=risk_pct,
    entry_price=stock_price,
    stop_distance_points=stop_points
)

st.info(f"""
**Execution Summary for {asset_name}:**  
• **14-Day ATR:** **₹{auto_atr:,.2f}** per share | Selected Stop Buffer: **{atr_multiplier}x ATR (₹{stop_points:,.2f})**  
• **Stop-Loss Level:** **₹{stop_loss_price:,.2f}** ({((stop_points / stock_price) * 100):.2f}% below entry)  
• **Order Sizing:** Buy **{sizing['shares']} shares** at **₹{stock_price:,.2f}**  
• **Total Capital Committed:** **₹{sizing['position_value']:,.2f}** (**{sizing['portfolio_allocation_pct']:.1f}%** of account)  
• **Worst-Case 1R Capital Risk:** Capped at **₹{sizing['max_loss_rupees']:,.2f}** ({risk_pct}% of total capital).
""")