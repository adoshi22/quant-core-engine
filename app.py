"""
app.py
Production Streamlit Web Dashboard for quant-core-engine with real market feeds.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quant_risk.drawdowns import compute_drawdown_series, max_drawdown, max_drawdown_duration
from quant_risk.tail_risk import var_historic, var_cornish_fisher, cvar_historic, semi_deviation, distribution_skewness, distribution_excess_kurtosis
from quant_risk.metrics import annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio, calmar_ratio
from quant_risk.sizing import calculate_atr_position_size
from quant_risk.data import get_daily_returns, fetch_historical_data

st.set_page_config(
    page_title="Quantitative Risk & Portfolio Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Institutional Risk & Quantitative Sizing Engine")
st.markdown("Automated downside risk modeling, Cornish-Fisher tail analytics, and volatility-budgeted execution on live NSE/BSE feeds.")

# Sidebar Configuration
st.sidebar.header("Asset Selection & Ingestion")
data_mode = st.sidebar.radio("Data Source", ["Live NSE / Global Equity", "Preset Simulation Profile"])

returns_series = None
current_price = 1000.0
asset_name = ""

if data_mode == "Live NSE / Global Equity":
    ticker_input = st.sidebar.text_input("Enter NSE/BSE Ticker (e.g., NIFTYBEES, BAJFINANCE, GOLDBEES)", value="NIFTYBEES")
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
    scenario = st.sidebar.selectbox(
        "Select Return Profile",
        [
            "Pillar 2: Momentum Satellite (High Beta)",
            "Pillar 1: Core Compounder (Low Volatility)",
            "Stressed Asset: Bearish Drawdown Tape"
        ]
    )
    asset_name = scenario
    np.random.seed(101)
    n_days = 504
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    
    if "Momentum" in scenario:
        daily_ret = np.random.normal(0.0010, 0.018, n_days)
        daily_ret[::35] += 0.035
    elif "Core" in scenario:
        daily_ret = np.random.normal(0.0006, 0.010, n_days)
    else:
        daily_ret = np.random.normal(-0.0003, 0.020, n_days)
        daily_ret[::25] -= 0.045
        
    returns_series = pd.Series(daily_ret, index=dates)
    current_price = 1500.0

dd_data = compute_drawdown_series(returns_series)

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
cagr_val = annualized_return(returns_series) * 100.0
vol_val = annualized_volatility(returns_series) * 100.0
mdd_val = max_drawdown(returns_series) * 100.0
sharpe_val = sharpe_ratio(returns_series, risk_free_rate=0.065)
sortino_val = sortino_ratio(returns_series, risk_free_rate=0.065)

col1.metric("CAGR", f"{cagr_val:.1f}%")
col2.metric("Annual Volatility", f"{vol_val:.1f}%")
col3.metric("Max Drawdown", f"{mdd_val:.2f}%")
col4.metric("Sharpe Ratio (Rf=6.5%)", f"{sharpe_val:.2f}")
col5.metric("Sortino Ratio", f"{sortino_val:.2f}")

st.divider()

# Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader(f"Equity Curve & High-Water Mark: {asset_name}")
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=dd_data.index, y=dd_data["wealth_index"], name="Wealth Index (Base 1000)", line=dict(color="#00CC96", width=2)))
    fig_w.add_trace(go.Scatter(x=dd_data.index, y=dd_data["previous_peaks"], name="Peak High-Water Mark", line=dict(color="#AB63FA", dash="dot")))
    fig_w.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_w, use_container_width=True)

with col_chart2:
    st.subheader("Underwater Drawdown Profile")
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=dd_data.index, y=dd_data["drawdown"] * 100.0, fill="tozeroy", name="Drawdown %", line=dict(color="#EF553B", width=1.5)))
    fig_d.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Decline from Peak (%)")
    st.plotly_chart(fig_d, use_container_width=True)

st.divider()

# Tail Risk Metrics
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
tcol4.metric(
    "Skewness", 
    f"{skew_val:.2f}",
    help="Negative skew indicates vulnerability to sudden gap-down selloffs."
)
tcol5.metric(
    "Excess Kurtosis", 
    f"{kurt_val:.2f}",
    help="Values > 0 denote fat tails (leptokurtic). Risk of extreme 3+ sigma shocks."
)
tcol6.metric("Max DD Duration", f"{longest_dd} Days")


# Volatility Sizing Calculator
st.subheader("🎯 Volatility-Budgeted Execution Calculator")
st.markdown("Calculates the precise share quantity based on a strict 1R risk budget and stop-loss buffer.")

p1, p2, p3, p4 = st.columns(4)
total_capital = p1.number_input("Portfolio Capital (₹)", value=1000000, step=50000)
risk_pct = p2.slider("Risk Budget (1R %)", min_value=0.25, max_value=2.00, value=0.75, step=0.05)
stock_price = p3.number_input("Execution Price (₹)", value=float(round(current_price, 2)), step=10.0)
stop_points = p4.number_input("Stop Distance (₹ / ATR Buffer)", value=float(round(current_price * 0.05, 2)), step=5.0)

sizing = calculate_atr_position_size(
    account_capital=total_capital,
    risk_budget_pct=risk_pct,
    entry_price=stock_price,
    stop_distance_points=stop_points
)

st.info(f"""
**Execution Summary for {asset_name}:**  
Buy **{sizing['shares']} shares** at **₹{stock_price:,.2f}** | Total Position Size: **₹{sizing['position_value']:,.2f}** (**{sizing['portfolio_allocation_pct']:.1f}%** of portfolio)  
Worst-Case Loss at Stop: strictly capped at **₹{sizing['max_loss_rupees']:,.2f}** ({risk_pct}% of account).
""")