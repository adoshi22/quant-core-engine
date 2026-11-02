# 🛡️ Quant Core Engine: Institutional Risk & Sizing Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-PyTest%20Passing-brightgreen.svg)]()
[![Framework](https://img.shields.io/badge/UI-Streamlit%20%7C%20Plotly-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

A modular quantitative equity risk modeling, non-Gaussian tail-risk analytics, and volatility-budgeted execution pipeline built with vectorized Python (`NumPy`, `pandas`, `SciPy`) and an interactive `Streamlit` dashboard.

---

## 📌 Problem Solved
Traditional retail dashboards rely on simplistic Gaussian assumptions (standard deviation) and focus purely on nominal returns, obscuring crash risk and drawdown duration. 

**Quant Core Engine** provides an institutional risk assessment framework:
1. **True Downside Isolation:** Evaluates **Sortino** and **Semi-Deviation** over standard Sharpe to distinguish upward momentum expansions from downside distribution shocks.
2. **Fat-Tail Modeling:** Implements the **Cornish-Fisher expansion** to adjust Value-at-Risk (VaR) and Conditional VaR (CVaR) for real-world negative skewness and excess kurtosis.
3. **Underwater Equity Analytics:** Tracks running high-water marks, absolute peak-to-trough drawdowns, and recovery duration lag in trading days.
4. **Volatility-Budgeted Execution ($1R$):** Dynamically scales position sizing and share quantity to guarantee no single execution exceeds a pre-defined portfolio risk budget.

---

## 🏛️ System Architecture

```text
quant-core-engine/
├── quant_risk/                 # Pure Quantitative Math Core
│   ├── drawdowns.py            # High-water marks, drawdown series, duration streaks
│   ├── tail_risk.py            # Cornish-Fisher Modified VaR, Semi-deviation, CVaR
│   ├── metrics.py              # CAGR, Annualized Volatility, Sharpe, Sortino, Calmar
│   ├── sizing.py               # Volatility-budgeted 1R capital allocation
│   └── data.py                 # Live market data ETL (NSE / Global feeds via yfinance)
├── tests/
│   └── test_risk.py            # PyTest numerical accuracy & edge-case suite
├── app.py                      # Interactive Streamlit & Plotly analytics dashboard
├── requirements.txt            # Project dependencies
└── .gitignore