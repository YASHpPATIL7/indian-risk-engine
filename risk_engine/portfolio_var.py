"""
Portfolio VaR + CVaR Calculator
Inputs:  data/vajra_dcc_cov.pkl     (T x N x N covariance cube, pct units)
         data/vajra_returns.csv     (T x N log returns, decimal units)
Outputs: data/vajra_var_history.csv (daily VaR, CVaR, portfolio vol)
         assets/var_history.png     (VaR through time plot)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. LOAD INPUTS
with open(os.path.join(BASE_DIR, "data", "vajra_dcc_cov.pkl"), "rb") as f:
    cov_data = pickle.load(f)

cov_cube = cov_data["cov"]
dates    = pd.Index(cov_data["dates"])
stocks   = cov_data["stocks"]

returns_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_returns.csv"),
    index_col=0, parse_dates=True
)
returns_df = returns_df[stocks].reindex(dates).fillna(0.0).astype(np.float64)
returns_df = returns_df[stocks]
returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
returns_df = returns_df.fillna(0.0)
returns_df = returns_df.astype(np.float64)

logger.info(f"NaN check: {returns_df.isna().sum().sum()} NaNs remaining")
logger.info(f"Inf check: {np.isinf(returns_df.values).sum()} Infs remaining")

T, N, _ = cov_cube.shape
logger.info(f"Covariance cube loaded: {cov_cube.shape}")
logger.info(f"Stocks: {stocks}\n")

# 2. PORTFOLIO WEIGHTS
weights         = np.full(N, 1.0 / N, dtype=np.float64)
PORTFOLIO_VALUE = 10_000_000

logger.info(f"Portfolio value : Rs{PORTFOLIO_VALUE:,.0f}")
logger.info(f"Weights         : Equal ({round(1/N*100, 2)}% per stock)")
logger.info(f"N stocks        : {N}\n")

Z_95    = 1.6449
Z_99    = 2.3263
CVAR_95 = 2.0627
CVAR_99 = 2.6652

# 3. COMPUTE DAILY VaR AND CVaR
logger.info("--- COMPUTING DAILY VaR ---")

sigma_p_list = []
var95_list   = []
var99_list   = []
cvar95_list  = []
cvar99_list  = []
cholesky_failures = 0

for t in range(T):
    Sigma_t = cov_cube[t]
    try:
        L       = np.linalg.cholesky(Sigma_t)
        Lw      = L.T @ weights
        sigma_p = np.sqrt(float(np.dot(Lw, Lw)))
    except np.linalg.LinAlgError:
        cholesky_failures += 1
        var_p   = float(weights @ Sigma_t @ weights)
        sigma_p = np.sqrt(max(var_p, 0.0))

    sigma_dec = sigma_p / 100.0
    sigma_p_list.append(sigma_p)
    var95_list.append(Z_95    * sigma_dec * PORTFOLIO_VALUE)
    var99_list.append(Z_99    * sigma_dec * PORTFOLIO_VALUE)
    cvar95_list.append(CVAR_95 * sigma_dec * PORTFOLIO_VALUE)
    cvar99_list.append(CVAR_99 * sigma_dec * PORTFOLIO_VALUE)

logger.info(f"Cholesky failures: {cholesky_failures} / {T} days\n")

logger.info("--- UNITS DIAGNOSIS ---")
for i in range(5):
    logger.info(f"  Day {i}: sigma_p = {sigma_p_list[i]:.6f}")
logger.info(f"\nreturns_df sample (first row):\n{returns_df.iloc[0]}")
logger.info(f"\nreturns_df describe:\n{returns_df.describe()}\n")

# 4. BUILD RESULTS DATAFRAME
var_df = pd.DataFrame({
    "sigma_p_pct" : sigma_p_list,
    "VaR_95"      : var95_list,
    "VaR_99"      : var99_list,
    "CVaR_95"     : cvar95_list,
    "CVaR_99"     : cvar99_list,
}, index=dates)
weights = weights.astype(np.float64)
logger.info(f"weights dtype: {weights.dtype}, shape: {weights.shape}")
# Debug: check state right before matmul
arr = returns_df.values
logger.info(f"Pre-matmul NaN: {np.isnan(arr).sum()}, Inf: {np.isinf(arr).sum()}")
logger.info(f"Pre-matmul dtype: {arr.dtype}")
logger.info(f"Pre-matmul min: {arr.min():.6f}, max: {arr.max():.6f}")
var_df["actual_return_decimal"] = np.einsum('ij,j->i', returns_df.values, weights)
var_df["actual_pnl"]            = var_df["actual_return_decimal"] * PORTFOLIO_VALUE
var_df["actual_loss"]           = -var_df["actual_pnl"]

# 5. SUMMARY STATISTICS
logger.info("--- VaR SUMMARY STATISTICS ---\n")
logger.info(f"{'Metric':<30} {'Min':>12} {'Mean':>12} {'Max':>12}")
logger.info("-" * 68)
for label, col in [("Portfolio Vol (% daily)", "sigma_p_pct"),
                   ("VaR 95% (Rs)",            "VaR_95"),
                   ("VaR 99% (Rs)",            "VaR_99"),
                   ("CVaR 95% (Rs)",           "CVaR_95")]:
    if "Vol" in label:
        logger.info(f"{label:<30} {var_df[col].min():>12.4f} {var_df[col].mean():>12.4f} {var_df[col].max():>12.4f}")
    else:
        logger.info(f"{label:<30} {var_df[col].min():>12,.0f} {var_df[col].mean():>12,.0f} {var_df[col].max():>12,.0f}")

logger.info("\n--- TOP 5 HIGHEST RISK DAYS ---")
top5 = var_df.nlargest(5, "VaR_95")[["sigma_p_pct", "VaR_95", "CVaR_95"]]
for date, row in top5.iterrows():
    logger.info(f"  {str(date)[:10]}  sigma_p={row['sigma_p_pct']:.3f}%  VaR95=Rs{row['VaR_95']:,.0f}  CVaR95=Rs{row['CVaR_95']:,.0f}")

# 6. BREACH ANALYSIS
logger.info("\n--- BREACH ANALYSIS (95% VaR) ---")
breaches_95    = var_df[var_df["actual_loss"] > var_df["VaR_95"]]
breaches_99    = var_df[var_df["actual_loss"] > var_df["VaR_99"]]
breach_rate_95 = len(breaches_95) / T
breach_rate_99 = len(breaches_99) / T

logger.info(f"Total days          : {T}")
logger.info(f"95% VaR breaches    : {len(breaches_95)} ({breach_rate_95:.2%})  [expected: ~5.00%]")
logger.info(f"99% VaR breaches    : {len(breaches_99)} ({breach_rate_99:.2%})  [expected: ~1.00%]")

if breach_rate_95 < 0.04:
    logger.info("WARNING 95% VaR: breach rate too low")
elif breach_rate_95 > 0.06:
    logger.info("WARNING 95% VaR: breach rate too high")
else:
    logger.info("OK 95% VaR: breach rate within acceptable range")

logger.info("\nTop 5 breach days (95%):")
for date, row in breaches_95.nlargest(5, "actual_loss").iterrows():
    excess = row["actual_loss"] - row["VaR_95"]
    logger.info(f"  {str(date)[:10]}  Loss=Rs{row['actual_loss']:,.0f}  VaR=Rs{row['VaR_95']:,.0f}  Excess=Rs{excess:,.0f}")

# 7. PLOT
logger.info("\n--- PLOTTING ---")
fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0f0f0f", sharex=True)
fig.suptitle("GARCH Risk Engine — Dynamic VaR (DCC-GARCH)\nEqual Weight Portfolio | Rs1 Crore | NSE Large-Caps",
             color="white", fontsize=13, y=0.98)

c95 = "#00d4aa"; c99 = "#ff6b6b"; cc95 = "#ffd93d"; cl = "#ffffff"; cb = "#ff3333"

ax1 = axes[0]
ax1.fill_between(dates, var_df["sigma_p_pct"], alpha=0.4, color=c95)
ax1.plot(dates, var_df["sigma_p_pct"], color=c95, linewidth=0.8, label="Portfolio sigma (%)")
ax1.set_facecolor("#0f0f0f"); ax1.set_ylabel("Daily Vol (%)", color="white", fontsize=10)
ax1.tick_params(colors="white"); ax1.legend(loc="upper right", facecolor="#1a1a1a", labelcolor="white", fontsize=9)
ax1.spines[["top","right","left","bottom"]].set_color("#333333"); ax1.grid(alpha=0.15, color="white")

ax2 = axes[1]
ax2.plot(dates, var_df["VaR_95"]  / 1000, color=c95,  linewidth=0.8, label="VaR 95% (RsK)")
ax2.plot(dates, var_df["VaR_99"]  / 1000, color=c99,  linewidth=0.8, label="VaR 99% (RsK)")
ax2.plot(dates, var_df["CVaR_95"] / 1000, color=cc95, linewidth=0.6, linestyle="--", label="CVaR 95% (RsK)", alpha=0.8)
ax2.set_facecolor("#0f0f0f"); ax2.set_ylabel("VaR / CVaR (Rs Thousands)", color="white", fontsize=10)
ax2.tick_params(colors="white"); ax2.legend(loc="upper right", facecolor="#1a1a1a", labelcolor="white", fontsize=9)
ax2.spines[["top","right","left","bottom"]].set_color("#333333"); ax2.grid(alpha=0.15, color="white")

ax3 = axes[2]
ax3.plot(dates, var_df["actual_loss"] / 1000, color=cl, linewidth=0.6, alpha=0.7, label="Actual Loss (RsK)")
ax3.plot(dates, var_df["VaR_95"]      / 1000, color=c95, linewidth=0.8, label="VaR 95% (RsK)")
if len(breaches_95) > 0:
    ax3.scatter(breaches_95.index, breaches_95["actual_loss"] / 1000,
                color=cb, s=15, zorder=5, label=f"Breach ({len(breaches_95)} days)", alpha=0.8)
ax3.set_facecolor("#0f0f0f"); ax3.set_ylabel("Loss vs VaR (Rs Thousands)", color="white", fontsize=10)
ax3.tick_params(colors="white"); ax3.legend(loc="upper right", facecolor="#1a1a1a", labelcolor="white", fontsize=9)
ax3.spines[["top","right","left","bottom"]].set_color("#333333"); ax3.grid(alpha=0.15, color="white")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", color="white", fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "assets", "var_history.png"), dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
logger.info("Saved assets/var_history.png")

# 8. SAVE CSV
var_df.to_csv(os.path.join(BASE_DIR, "data", "vajra_var_history.csv"))
logger.info("Saved data/vajra_var_history.csv")
