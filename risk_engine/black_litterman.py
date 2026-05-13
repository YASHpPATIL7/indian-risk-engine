"""
Black-Litterman Optimization
GARCH Risk Engine

Goal:
- derive market-implied prior returns from latest covariance
- inject analyst views
- compute Black-Litterman posterior returns
- optimize posterior portfolio weights

Inputs:
    data/vajra_dcc_cov.pkl
    data/vajra_returns.csv

Outputs:
    data/black_litterman_summary.csv
    data/black_litterman_weights.csv
    assets/black_litterman.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import logging
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. LOAD DATA ──────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "data", "vajra_dcc_cov.pkl"), "rb") as f:
    cov_obj = pickle.load(f)

# Support both dict-style and raw cube-style saves
if isinstance(cov_obj, dict):
    cov_cube = np.asarray(cov_obj["cov"], dtype=np.float64)
    dates = pd.Index(pd.to_datetime(cov_obj["dates"]))
    stocks = list(cov_obj["stocks"])
else:
    cov_cube = np.asarray(cov_obj, dtype=np.float64)

    returns_df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "vajra_returns.csv"),
        index_col=0,
        parse_dates=True
    )
    dates = returns_df.index[-len(cov_cube):]
    stocks = list(returns_df.columns)

latest_cov_pct = pd.DataFrame(cov_cube[-1], index=stocks, columns=stocks)

# The DCC engine stores covariance using percentage returns/vols.
# PyPortfolioOpt expects covariance in decimal-return units.
# So:
#   1. convert %² -> decimal² by dividing by 100^2
#   2. annualize daily covariance by multiplying by 252
latest_cov = latest_cov_pct / 10000.0
latest_cov = latest_cov * 252

logger.info("\n--- COVARIANCE SCALING CHECK ---")
logger.info(f"Raw daily variance max (%²): {np.diag(latest_cov_pct).max():.6f}")
logger.info(f"Scaled annual variance max  : {np.diag(latest_cov).max():.6f}")
logger.info(f"Scaled annual vol max (%)   : {100*np.sqrt(np.diag(latest_cov)).max():.2f}")

returns_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_returns.csv"),
    index_col=0,
    parse_dates=True
)

logger.info(f"Returns loaded: {returns_df.shape}")
logger.info(f"Covariance cube loaded: {cov_cube.shape}")
logger.info(f"Evaluation date: {dates[-1].date()}")
logger.info(f"Stocks: {stocks}")


# ── 2. MARKET CAPS → MARKET WEIGHTS ───────────────────────────
# Approximate market caps in INR (replace later with exact current values if you want)
market_caps = pd.Series({
    "AXISBANK":    3.6e12,
    "BAJFINANCE":  4.5e12,
    "DRREDDY":     1.1e12,
    "HDFCBANK":   13.5e12,
    "HINDUNILVR":  5.7e12,
    "ICICIBANK":   8.9e12,
    "INFY":        6.3e12,
    "ITC":         5.4e12,
    "MARUTI":      3.9e12,
    "ONGC":        3.5e12,
    "RELIANCE":   20.0e12,
    "SUNPHARMA":   4.1e12,
    "TCS":        14.5e12,
    "WIPRO":       3.0e12,
}).reindex(stocks)

market_weights = market_caps / market_caps.sum()

logger.info("\n--- MARKET WEIGHTS ---")
for s, w in market_weights.sort_values(ascending=False).items():
    logger.info(f"{s:<12} {100*w:>7.2f}%")


# ── 3. IMPLIED PRIOR RETURNS (MARKET EQUILIBRIUM) ─────────────
# delta = risk aversion. 2 to 3 is common as a practical choice.
delta = 1.0

prior = black_litterman.market_implied_prior_returns(
    market_caps=market_caps,
    risk_aversion=delta,
    cov_matrix=latest_cov
)

logger.info("\n--- IMPLIED PRIOR RETURNS (%) ---")
for s, r in (prior * 100).sort_values(ascending=False).items():
    logger.info(f"{s:<12} {r:>8.3f}%")


# ── 4. ANALYST VIEW ────────────────────────────────────────────
# View:
# "IT sector outperforms broad basket by 3%
#  because RBI pause + USD strength"

# Positive basket
it_names = ["INFY", "TCS", "WIPRO"]

# Negative comparison basket
rest_names = [
    "AXISBANK", "HDFCBANK", "ICICIBANK",
    "RELIANCE", "ITC", "HINDUNILVR", "ONGC"
]

P = pd.DataFrame(0.0, index=["IT_outperforms"], columns=stocks)

for s in it_names:
    P.loc["IT_outperforms", s] = 1.0 / len(it_names)

for s in rest_names:
    P.loc["IT_outperforms", s] = -1.0 / len(rest_names)

Q = pd.Series({"IT_outperforms": 0.03})   # +3% relative outperformance

# Confidence:
# smaller omega = more confidence in view
omega = np.array([[0.0004]])

logger.info("\n--- VIEW MATRIX P ---")
logger.info(P.round(4).to_string())

logger.info("\n--- VIEW VECTOR Q ---")
logger.info(Q.to_string())


# ── 5. BLACK-LITTERMAN POSTERIOR ──────────────────────────────
bl = BlackLittermanModel(
    cov_matrix=latest_cov,
    pi=prior,
    P=P.values,
    Q=Q.values,
    omega=omega,
    tau=0.05
)

posterior_returns = bl.bl_returns()
posterior_cov = bl.bl_cov()

logger.info("\n--- POSTERIOR RETURNS (%) ---")
for s, r in (posterior_returns * 100).sort_values(ascending=False).items():
    logger.info(f"{s:<12} {r:>8.3f}%")


# ── 6. OPTIMIZE POSTERIOR PORTFOLIO ───────────────────────────
# Keep realistic caps to avoid optimizer insanity
# use a low risk-free rate because posterior returns are modest
rf = 0.00

ef = EfficientFrontier(
    expected_returns=posterior_returns,
    cov_matrix=posterior_cov,
    weight_bounds=(0.0, 0.20)
)

weights = ef.max_sharpe(risk_free_rate=rf)
cleaned_weights = ef.clean_weights()

exp_ret, vol, sharpe = ef.portfolio_performance(
    verbose=False,
    risk_free_rate=rf
)

weights_df = pd.DataFrame({
    "stock": stocks,
    "market_weight": market_weights.values,
    "prior_return": prior.reindex(stocks).values,
    "posterior_return": posterior_returns.reindex(stocks).values,
    "bl_weight": [cleaned_weights.get(s, 0.0) for s in stocks],
})

weights_df["active_weight"] = weights_df["bl_weight"] - weights_df["market_weight"]
weights_df = weights_df.sort_values("bl_weight", ascending=False)

logger.info("\n--- BLACK-LITTERMAN POSTERIOR WEIGHTS ---")
for _, row in weights_df.iterrows():
    logger.info(
        f"{row['stock']:<12} "
        f"BL={100*row['bl_weight']:>6.2f}%   "
        f"MKT={100*row['market_weight']:>6.2f}%   "
        f"ACTIVE={100*row['active_weight']:>7.2f}%"
    )

logger.info("\n--- PORTFOLIO PERFORMANCE ---")
logger.info(f"Expected return : {100*exp_ret:.2f}%")
logger.info(f"Volatility      : {100*vol:.2f}%")
logger.info(f"Sharpe ratio    : {sharpe:.4f}")


# ── 7. SAVE RESULTS ───────────────────────────────────────────
summary_df = pd.DataFrame([{
    "date": dates[-1],
    "view": "IT sector outperforms broad basket by 3%",
    "risk_aversion_delta": delta,
    "tau": 0.05,
    "expected_return": exp_ret,
    "volatility": vol,
    "sharpe": sharpe,
    "risk_free_rate": rf,
}])

summary_df.to_csv(
    os.path.join(BASE_DIR, "data", "black_litterman_summary.csv"),
    index=False
)

weights_df.to_csv(
    os.path.join(BASE_DIR, "data", "black_litterman_weights.csv"),
    index=False
)


# ── 8. PLOT ───────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)

plot_df = weights_df.sort_values("bl_weight", ascending=True)

fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0f0f0f")
ax.set_facecolor("#0f0f0f")

ax.barh(
    plot_df["stock"],
    plot_df["market_weight"] * 100,
    color="#4a4a4a",
    alpha=0.75,
    label="Market Weight"
)

ax.barh(
    plot_df["stock"],
    plot_df["bl_weight"] * 100,
    color="#00d4aa",
    alpha=0.85,
    label="BL Posterior Weight"
)

ax.set_title(
    "Black-Litterman Posterior vs Market Weights\n"
    "View: IT Sector Outperforms by 3%",
    color="white",
    fontsize=13
)
ax.set_xlabel("Weight (%)", color="white")
ax.set_ylabel("Stock", color="white")
ax.tick_params(colors="white")
ax.spines[["top", "right", "left", "bottom"]].set_color("#333333")
ax.grid(axis="x", alpha=0.12, color="white")
ax.legend(facecolor="#1a1a1a", labelcolor="white")

plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "assets", "black_litterman.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="#0f0f0f"
)
plt.close()

logger.info("\nSaved → data/black_litterman_summary.csv")
logger.info("Saved → data/black_litterman_weights.csv")
logger.info("Saved → assets/black_litterman.png")
