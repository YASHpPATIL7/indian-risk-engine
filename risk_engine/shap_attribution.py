"""
SHAP Risk Attribution
Closed-form Shapley decomposition of daily portfolio variance and VaR
Input:  data/vajra_returns.csv
        data/vajra_var_history.csv
Output: data/shap_variance.csv
        data/shap_var_rupees.csv
        assets/shap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
logger = logging.getLogger(__name__)
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
returns_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_returns.csv"),
    index_col=0, parse_dates=True
)
var_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_var_history.csv"),
    index_col=0, parse_dates=True
)

stocks          = list(returns_df.columns)
N               = len(stocks)
weights         = np.full(N, 1.0 / N)
PORTFOLIO_VALUE = 10_000_000
Z_95            = 1.6449
Z_99            = 2.3263

common_idx = returns_df.index.intersection(var_df.index)
returns_df = returns_df.loc[common_idx]
var_df     = var_df.loc[common_idx]
T          = len(common_idx)

logger.info(f"Stocks: {N}, Days: {T}")
logger.info(f"Date range: {common_idx[0].date()} → {common_idx[-1].date()}\n")

# ── 2. LOAD DCC COVARIANCE OR FALLBACK ────────────────────────────────────────
dcc_cov_path = os.path.join(BASE_DIR, "data", "dcc_covariance.npy")
if os.path.exists(dcc_cov_path):
    H_all   = np.load(dcc_cov_path)
    use_dcc = True
    logger.info(f"Loaded DCC covariance: {H_all.shape}")
else:
    use_dcc = False
    logger.info("DCC covariance not found — using rolling 60-day empirical cov")

# ── 3. CLOSED-FORM SHAP — VARIANCE ATTRIBUTION ────────────────────────────────
shap_variance = np.full((T, N), np.nan)
returns_arr   = returns_df[stocks].values
ROLL          = 60

if use_dcc:
    for t in range(T):
        Ht               = H_all[t]
        Hw               = Ht @ weights
        shap_variance[t] = weights * Hw
else:
    for t in range(ROLL, T):
        window           = returns_arr[t - ROLL:t]
        Ht               = np.cov(window.T)
        Hw               = Ht @ weights
        shap_variance[t] = weights * Hw

# ── 4. CLEAN NaNs ─────────────────────────────────────────────────────────────
shap_variance_clean = np.nan_to_num(shap_variance, nan=0.0)

# ── 5. EFFICIENCY CHECK ───────────────────────────────────────────────────────
sigma_p_sq_check = shap_variance_clean.sum(axis=1)
sigma_p_check    = np.sqrt(np.abs(sigma_p_sq_check))
sigma_p_actual   = var_df["VaR_95"].values / (Z_95 * PORTFOLIO_VALUE)

corr = np.corrcoef(sigma_p_check[ROLL:], sigma_p_actual[ROLL:])[0, 1]
logger.info(f"Efficiency check — correlation σ_p (SHAP) vs σ_p (DCC): {corr:.4f}")
logger.info(f"  ~0.99 if DCC used | ~0.7-0.9 for rolling fallback\n")

# ── 6. SHAP PERCENTAGE SHARES ─────────────────────────────────────────────────
total_var = shap_variance_clean.sum(axis=1, keepdims=True)
total_var = np.where(total_var == 0, 1e-12, total_var)
shap_pct  = shap_variance_clean / total_var * 100       # (T × N)

# ── 7. VAR SHAP — RUPEE ATTRIBUTION ───────────────────────────────────────────
var95_series      = var_df["VaR_95"].values.reshape(-1, 1)
var99_series      = var_df["VaR_99"].values.reshape(-1, 1)
shap_var95_rupees = (shap_pct / 100) * var95_series
shap_var99_rupees = (shap_pct / 100) * var99_series

# ── 8. SUMMARY STATISTICS ─────────────────────────────────────────────────────
valid_mask        = sigma_p_sq_check > 0
mean_shap_pct     = pd.Series(
    np.nanmean(shap_pct[valid_mask], axis=0), index=stocks
)
mean_var95_rupees = pd.Series(
    np.nanmean(shap_var95_rupees[valid_mask], axis=0), index=stocks
)

mean_shap_pct     = mean_shap_pct.sort_values(ascending=False)
mean_var95_rupees = mean_var95_rupees.reindex(mean_shap_pct.index)

logger.info("Average SHAP Variance Attribution (% of portfolio variance):")
logger.info(f"{'Stock':<14} {'Var%':>8}  {'VaR95 ₹':>12}")
logger.info("─" * 42)
for stock in mean_shap_pct.index:
    pct   = mean_shap_pct[stock]
    rupee = mean_var95_rupees[stock]
    pct_safe = pct if not np.isnan(pct) else 0.0
    bar   = "█" * int(pct_safe / 2)
    logger.info(f"{stock:<14} {pct_safe:>7.2f}%  ₹{rupee/1000:>8.1f}K  {bar}")

logger.info(f"\nTotal check: {mean_shap_pct.sum():.2f}%  (should = 100%)")

# ── 9. REGIME ANALYSIS ────────────────────────────────────────────────────────
covid_mask = (common_idx >= "2020-02-15") & (common_idx <= "2020-04-30")
calm_mask  = (common_idx >= "2021-06-01") & (common_idx <= "2022-01-01")
bull_mask  = (common_idx >= "2023-01-01") & (common_idx <= "2024-01-01")

covid_shap = np.nanmean(shap_pct[covid_mask], axis=0)
calm_shap  = np.nanmean(shap_pct[calm_mask],  axis=0)
bull_shap  = np.nanmean(shap_pct[bull_mask],  axis=0)

logger.info("\nRegime SHAP % — All Stocks:")
logger.info(f"{'Stock':<14} {'COVID':>8} {'Calm':>8} {'Bull23':>8}")
logger.info("─" * 42)
covid_order = np.argsort(covid_shap)[::-1]
for i in covid_order:
    s = stocks[i]
    logger.info(f"{s:<14} {covid_shap[i]:>7.2f}%  "
          f"{calm_shap[i]:>7.2f}%  {bull_shap[i]:>7.2f}%")

# ── 10. PEAK DAY — 2020-03-23 ─────────────────────────────────────────────────
peak_date  = pd.Timestamp("2020-03-23")
peak_exists = peak_date in common_idx
if peak_exists:
    peak_idx  = common_idx.get_loc(peak_date)
    peak_shap = shap_pct[peak_idx]
    peak_var  = var95_series[peak_idx, 0]
    logger.info(f"\nPeak COVID Day Attribution — 2020-03-23")
    logger.info(f"Portfolio VaR 95% = ₹{peak_var/1000:.1f}K")
    logger.info(f"{'Stock':<14} {'Var%':>8} {'₹K':>10}")
    logger.info("─" * 36)
    for i in np.argsort(peak_shap)[::-1]:
        s       = stocks[i]
        rupee_k = peak_shap[i] / 100 * peak_var / 1000
        logger.info(f"{s:<14} {peak_shap[i]:>7.2f}%  ₹{rupee_k:>7.1f}K")

# ── 11. SAVE CSVs ─────────────────────────────────────────────────────────────
pd.DataFrame(shap_pct, index=common_idx,
             columns=stocks).rename_axis("date").to_csv(
    os.path.join(BASE_DIR, "data", "shap_variance.csv")
)
pd.DataFrame(shap_var95_rupees, index=common_idx,
             columns=stocks).rename_axis("date").to_csv(
    os.path.join(BASE_DIR, "data", "shap_var_rupees.csv")
)
logger.info("\nSaved → data/shap_variance.csv")
logger.info("Saved → data/shap_var_rupees.csv")

# ── 12. PLOT ───────────────────────────────────────────────────────────────────
COLORS = [
    "#00d4aa","#ffd93d","#ff6b6b","#a29bfe","#fd79a8",
    "#6c5ce7","#00b894","#e17055","#74b9ff","#55efc4",
    "#fdcb6e","#b2bec3","#d63031","#0984e3"
]

fig = plt.figure(figsize=(20, 16), facecolor="#0f0f0f")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Average SHAP % bar
ax1 = fig.add_subplot(gs[0, 0])
sorted_stocks = mean_shap_pct.index.tolist()
sorted_vals   = mean_shap_pct.values
bar_colors    = [COLORS[stocks.index(s)] for s in sorted_stocks]
bars = ax1.barh(sorted_stocks, sorted_vals, color=bar_colors, alpha=0.85)
for bar, val in zip(bars, sorted_vals):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", color="white", fontsize=7)
ax1.set_facecolor("#0f0f0f")
ax1.set_title("Average Variance SHAP %\n(Full Period 2019–2024)",
              color="white", fontsize=10)
ax1.set_xlabel("% of Portfolio Variance", color="white", fontsize=9)
ax1.tick_params(colors="white", labelsize=8)
ax1.spines[["top","right","left","bottom"]].set_color("#333333")
ax1.grid(alpha=0.1, color="white", axis="x")

# Panel 2 — Average VaR ₹ bar
ax2 = fig.add_subplot(gs[0, 1])
sorted_rupee = mean_var95_rupees.sort_values(ascending=False)
rupee_stocks = sorted_rupee.index.tolist()
rupee_colors = [COLORS[stocks.index(s)] for s in rupee_stocks]
bars2 = ax2.barh(rupee_stocks, sorted_rupee.values / 1000,
                 color=rupee_colors, alpha=0.85)
for bar, val in zip(bars2, sorted_rupee.values / 1000):
    ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
             f"₹{val:.1f}K", va="center", color="white", fontsize=7)
ax2.set_facecolor("#0f0f0f")
ax2.set_title("Average VaR SHAP (₹K)\n95% VaR Attribution per Stock",
              color="white", fontsize=10)
ax2.set_xlabel("₹ Thousands", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[["top","right","left","bottom"]].set_color("#333333")
ax2.grid(alpha=0.1, color="white", axis="x")

# Panel 3 — Stacked area top 6
ax3 = fig.add_subplot(gs[1, :])
top6 = mean_shap_pct.head(6).index.tolist()
shap_smooth = pd.DataFrame(
    shap_pct, index=common_idx, columns=stocks
)[top6].rolling(30).mean().fillna(0)

ax3.stackplot(
    common_idx,
    [shap_smooth[s].values for s in top6],
    labels=top6,
    colors=[COLORS[stocks.index(s)] for s in top6],
    alpha=0.8
)
ax3.set_facecolor("#0f0f0f")
ax3.set_title("Rolling 30-day SHAP % — Top 6 Risk Contributors Over Time",
              color="white", fontsize=10)
ax3.set_ylabel("% of Portfolio Variance", color="white", fontsize=9)
ax3.legend(loc="upper right", facecolor="#1a1a1a",
           labelcolor="white", fontsize=8, ncol=3)
ax3.tick_params(colors="white", labelsize=8)
ax3.spines[["top","right","left","bottom"]].set_color("#333333")
ax3.grid(alpha=0.1, color="white")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax3.xaxis.get_majorticklabels(),
         rotation=45, ha="right", color="white", fontsize=7)

# Panel 4 — Regime grouped bar
ax4 = fig.add_subplot(gs[2, 0])
x     = np.arange(N)
width = 0.25
ax4.bar(x - width, covid_shap, width,
        label="COVID Crash",     color="#ff6b6b", alpha=0.85)
ax4.bar(x,          calm_shap,  width,
        label="Post-COVID Calm", color="#00d4aa", alpha=0.85)
ax4.bar(x + width,  bull_shap,  width,
        label="2023 Bull Run",   color="#ffd93d", alpha=0.85)
ax4.set_facecolor("#0f0f0f")
ax4.set_title("SHAP % by Regime — COVID vs Calm vs Bull",
              color="white", fontsize=10)
ax4.set_ylabel("% of Portfolio Variance", color="white", fontsize=9)
ax4.set_xticks(x)
ax4.set_xticklabels(stocks, rotation=45, ha="right",
                    color="white", fontsize=7)
ax4.tick_params(colors="white", labelsize=8)
ax4.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
ax4.spines[["top","right","left","bottom"]].set_color("#333333")
ax4.grid(alpha=0.1, color="white", axis="y")

# Panel 5 — Peak COVID day pie
ax5 = fig.add_subplot(gs[2, 1])
if peak_exists:
    peak_order = np.argsort(peak_shap)[::-1]
    top5_idx   = peak_order[:5]
    other_pct  = peak_shap[peak_order[5:]].sum()
    pie_labels = [stocks[i] for i in top5_idx] + ["Others"]
    pie_vals   = list(peak_shap[top5_idx]) + [other_pct]
    pie_colors = [COLORS[i] for i in top5_idx] + ["#636e72"]
    ax5.pie(pie_vals, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 8},
            startangle=140)
    ax5.set_title("VaR Attribution — Peak COVID Day\n2020-03-23",
                  color="white", fontsize=10)
ax5.set_facecolor("#0f0f0f")

fig.suptitle(
    "GARCH Risk Engine — SHAP Risk Attribution\n"
    "Variance SHAP + VaR SHAP | 14 NSE Stocks | 2019–2024",
    color="white", fontsize=13
)

os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(
    os.path.join(BASE_DIR, "assets", "shap.png"),
    dpi=150, bbox_inches="tight", facecolor="#0f0f0f"
)
plt.close()
logger.info("Saved → assets/shap.png")
