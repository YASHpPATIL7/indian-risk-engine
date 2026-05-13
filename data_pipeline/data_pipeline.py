# ============================================================
# GARCH RISK ENGINE — BLOCK 4A: DATA PIPELINE
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import logging
logger = logging.getLogger(__name__)

# ============================================================
# STEP 1: DEFINE UNIVERSE
# ============================================================
# 15 Nifty 50 stocks — spread across sectors
# NSE tickers on Yahoo Finance end with .NS
tickers = [
    "RELIANCE.NS",   # Energy
    "TCS.NS",        # IT
    "INFY.NS",       # IT
    "HDFCBANK.NS",   # Banking
    "ICICIBANK.NS",  # Banking
    "AXISBANK.NS",   # Banking
    "HINDUNILVR.NS", # FMCG
    "ITC.NS",        # FMCG
    "WIPRO.NS",      # IT
    "MARUTI.NS",     # Auto
    "SUNPHARMA.NS",  # Pharma
    "DRREDDY.NS",    # Pharma
    "ONGC.NS",       # Energy
    "BAJFINANCE.NS",  # NBFC
    "LTIM.NS"
]

# ============================================================
# STEP 2: FETCH 5 YEARS OF ADJUSTED CLOSE
# ============================================================
raw = yf.download(
    tickers,
    start="2019-01-01",
    end="2024-12-31",
    auto_adjust=True,   # adjusts for splits + dividends automatically
    progress=False
)

close = raw["Close"]
# Shape: ~1500 rows × 15 columns
# Each cell = adjusted closing price for that stock on that date
# NaN = market holiday or stock suspended that day

logger.info(f"Raw price shape: {close.shape}")
logger.info(f"Date range: {close.index[0].date()} → {close.index[-1].date()}")
logger.info(f"NaN count per stock:\n{close.isna().sum()}")

# ============================================================
# STEP 3: COMPUTE LOG RETURNS
# ============================================================
log_returns = np.log(close / close.shift(1))
# log(P_t / P_{t-1}) = log return for day t
# Why log returns and not % returns?
# 1. Additive over time: 5-day log return = sum of 5 daily log returns
#    % returns compound (multiply) — harder to work with mathematically
# 2. Symmetric: +10% and -10% are equally sized in log space
#    % returns are asymmetric: +100% to double, -50% to halve
# 3. GARCH and DCC theory assumes log returns — using % breaks assumptions
# First row = NaN for all stocks (no previous day to compare)

log_returns = log_returns.dropna(how="all")
# Drop rows where EVERY stock is NaN (first row)
# Keep rows where SOME stocks have NaN — handle per stock in GARCH step

logger.info(f"\nLog returns shape: {log_returns.shape}")
logger.info(f"\nFirst 3 rows:\n{log_returns.head(3).round(4)}")
logger.info(f"\nDescriptive stats:\n{log_returns.describe().round(4)}")
# Check: mean returns should be near 0, std around 0.01-0.03 (1-3% daily vol)
# If you see 0.5 or 50 somewhere — data error

# ============================================================
# STEP 4: HANDLE MISSING DATA
# ============================================================
# NSE holidays create NaN gaps — two strategies:
# (a) Forward fill: carry last known price → return = 0 that day
# (b) Drop the row: lose that day for ALL stocks
# Strategy: forward fill prices FIRST, then recompute returns
# This gives 0 return on holidays (correct — no trading = no change)

close_filled = close.ffill()
# ffill = forward fill: NaN gets the value from the previous row
# ONGC suspended for 2 days? Those days get the last known price
# Return for those days = 0 (no change) — correct behavior

log_returns_clean = np.log(close_filled / close_filled.shift(1)).dropna(how="all")
logger.info(f"\nAfter forward fill, NaN count:\n{log_returns_clean.isna().sum()}")
# Should be 0 for all stocks after ffill

# ============================================================
# STEP 5: ADF TEST — CONFIRM STATIONARITY
# ============================================================
# ADF (Augmented Dickey-Fuller) tests for unit root
# H0 (null hypothesis): series has a unit root → NON-stationary
# H1 (alternative): series is stationary
# p-value < 0.05 → reject H0 → series IS stationary → safe for GARCH
# p-value > 0.05 → fail to reject H0 → series is NON-stationary → problem

# ============================================================
# STEP 5: ADF TEST — CONFIRM STATIONARITY
# ============================================================
# Drop columns that are entirely NaN before ADF
log_returns_clean = log_returns_clean.dropna(axis=1, how='all')
logger.info(f"Stocks after dropping empty columns: {list(log_returns_clean.columns)}")

logger.info("\n--- ADF STATIONARITY TEST ON LOG RETURNS ---")
adf_results = {}

for col in log_returns_clean.columns:
    series = log_returns_clean[col].dropna()

    if len(series) < 50:
        logger.info(f"{col:20s} | SKIPPED — insufficient data ({len(series)} rows)")
        continue

    adf_stat, p_value, _, _, critical_values, _ = adfuller(series, autolag="AIC")
    result = "STATIONARY ✓" if p_value < 0.05 else "NON-STATIONARY ✗"
    adf_results[col] = {"ADF Stat": round(adf_stat, 4), "p-value": round(p_value, 4), "Result": result}
    logger.info(f"{col:20s} | ADF: {adf_stat:8.4f} | p: {p_value:.4f} | {result}")

# Expected: ALL stocks should show STATIONARY
# Log returns are almost always stationary — prices are not
# If any stock shows non-stationary p > 0.05:
#   → check for data errors (missing periods, wrong ticker)
#   → try differencing: log_returns.diff().dropna() (rare edge case)

# ============================================================
# STEP 6: FINAL CLEAN MATRIX
# ============================================================
# Remove .NS suffix for clean column names
log_returns_clean.columns = [c.replace(".NS", "") for c in log_returns_clean.columns]

logger.info(f"\n--- FINAL RETURNS MATRIX ---")
logger.info(f"Shape: {log_returns_clean.shape}")
# Should be ~1490 rows × 15 columns
# 1490 trading days × 15 stocks

logger.info(f"\nVol per stock (annualised):")
annual_vol = log_returns_clean.std() * np.sqrt(252)
logger.info(annual_vol.round(4))
# std of daily returns × sqrt(252) = annualised volatility
# Normal range for NSE large-caps: 0.20 to 0.50 (20% to 50% annual vol)
# BAJFINANCE and TATAMOTORS typically higher (~0.35-0.45)
# HINDUNILVR and ITC typically lower (~0.20-0.25)

# Save for Block 4B
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
log_returns_clean.to_csv(os.path.join(BASE_DIR, "data", "vajra_returns.csv"))

logger.info("\nSaved: vajra_returns.csv → ready for GARCH in Block 4B")