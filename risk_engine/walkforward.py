"""
Walk-Forward VaR Validation
Expanding window OOS validation — reveals regime-dependent model failures
Input:  data/vajra_returns.csv
        data/vajra_var_history.csv
Output: data/walkforward_results.csv
        assets/walkforward.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
logger = logging.getLogger(__name__)
from scipy import stats

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

stocks  = list(returns_df.columns)
N       = len(stocks)
weights = np.full(N, 1.0 / N)
PORTFOLIO_VALUE = 10_000_000

# Align both dataframes
common_idx  = returns_df.index.intersection(var_df.index)
returns_df  = returns_df.loc[common_idx]
var_df      = var_df.loc[common_idx]

logger.info(f"Returns  : {returns_df.shape}")
logger.info(f"VaR hist : {var_df.shape}")
logger.info(f"Date range: {common_idx[0].date()} → {common_idx[-1].date()}\n")


# ── 2. DAILY PORTFOLIO LOSS SERIES ────────────────────────────────────────────
daily_returns = np.einsum('ij,j->i', returns_df[stocks].values, weights)
daily_pnl     = daily_returns * PORTFOLIO_VALUE
daily_loss    = pd.Series(-daily_pnl, index=common_idx)

var95_series  = var_df["VaR_95"]
var99_series  = var_df["VaR_99"]

# ── 3. HIT SEQUENCES ──────────────────────────────────────────────────────────
hit95 = (daily_loss > var95_series).astype(int)
hit99 = (daily_loss > var99_series).astype(int)

T = len(daily_loss)
logger.info(f"Total days         : {T}")
logger.info(f"Total breaches 95% : {hit95.sum()}  ({hit95.mean():.2%})")
logger.info(f"Total breaches 99% : {hit99.sum()}  ({hit99.mean():.2%})\n")


# ── 4. ROLLING BREACH RATE (60-day window) ────────────────────────────────────
ROLL = 60
rolling_breach95 = hit95.rolling(ROLL).mean()
rolling_breach99 = hit99.rolling(ROLL).mean()


# ── 5. LOCAL KUPIEC (252-day rolling windows) ─────────────────────────────────
def kupiec_pof(N_breaches, T_days, p_expected):
    if T_days == 0:
        return np.nan, np.nan
    p_hat = N_breaches / T_days
    if p_hat == 0:
        p_hat = 1e-10
    if p_hat == 1:
        p_hat = 1 - 1e-10
    try:
        ll_null = N_breaches * np.log(p_expected) + \
                  (T_days - N_breaches) * np.log(1 - p_expected)
        ll_alt  = N_breaches * np.log(p_hat) + \
                  (T_days - N_breaches) * np.log(1 - p_hat)
        lr      = -2 * (ll_null - ll_alt)
        p_value = 1 - stats.chi2.cdf(lr, df=1)
        return lr, p_value
    except:
        return np.nan, np.nan

KUPIEC_WINDOW = 252
kupiec_records = []

for start in range(0, T - KUPIEC_WINDOW + 1, KUPIEC_WINDOW):
    end        = start + KUPIEC_WINDOW
    window_idx = common_idx[start:end]
    w_hit95    = hit95.iloc[start:end]
    w_hit99    = hit99.iloc[start:end]

    lr95, pv95 = kupiec_pof(w_hit95.sum(), KUPIEC_WINDOW, 0.05)
    lr99, pv99 = kupiec_pof(w_hit99.sum(), KUPIEC_WINDOW, 0.01)

    kupiec_records.append({
        "window_start"    : window_idx[0].date(),
        "window_end"      : window_idx[-1].date(),
        "breach_rate_95"  : w_hit95.mean(),
        "breach_rate_99"  : w_hit99.mean(),
        "lr_95"           : lr95,
        "pval_95"         : pv95,
        "pass_95"         : pv95 > 0.05 if not np.isnan(pv95) else False,
        "lr_99"           : lr99,
        "pval_99"         : pv99,
        "pass_99"         : pv99 > 0.05 if not np.isnan(pv99) else False,
    })

kupiec_df = pd.DataFrame(kupiec_records)


# ── 6. UCE — UNCONDITIONAL COVERAGE ERROR ─────────────────────────────────────
uce95 = (rolling_breach95 - 0.05).abs()
uce99 = (rolling_breach99 - 0.01).abs()
days_uce_bad95 = (uce95 > 0.05).sum()
days_uce_bad99 = (uce99 > 0.01).sum()


# ── 7. REGIME DETECTION ───────────────────────────────────────────────────────
# Simple regime: crisis if rolling breach95 > 3x expected (>15%)
crisis_mask    = rolling_breach95 > 0.15
calm_mask      = rolling_breach95.between(0.03, 0.07)
post_crisis    = rolling_breach95 < 0.02

crisis_breach  = rolling_breach95[crisis_mask].mean()
calm_breach    = rolling_breach95[calm_mask].mean()
post_breach    = rolling_breach95[post_crisis].mean()


# ── 8. PRINT RESULTS ──────────────────────────────────────────────────────────
logger.info("=" * 65)
logger.info("WALK-FORWARD VALIDATION RESULTS")
logger.info("=" * 65)

logger.info(f"\nRolling 60-day breach rate (95% VaR):")
logger.info(f"  Overall mean   : {rolling_breach95.mean():.2%}")
logger.info(f"  Max            : {rolling_breach95.max():.2%}  "
      f"on {rolling_breach95.idxmax().date()}")
logger.info(f"  Min            : {rolling_breach95.min():.2%}  "
      f"on {rolling_breach95.idxmin().date()}")
logger.info(f"  Days UCE > 5%  : {days_uce_bad95}")

logger.info(f"\nRegime Analysis (95% VaR):")
logger.info(f"  Calm regime avg breach    : {calm_breach:.2%}  "
      f"({calm_mask.sum()} days)")
logger.info(f"  Crisis regime avg breach  : {crisis_breach:.2%}  "
      f"({crisis_mask.sum()} days)")
logger.info(f"  Post-crisis avg breach    : {post_breach:.2%}  "
      f"({post_crisis.sum()} days)")

logger.info(f"\nLocal Kupiec (252-day windows):")
logger.info(f"{'Window':<25} {'BR95':>6} {'pval95':>8} {'Pass95':>7} "
      f"{'BR99':>6} {'pval99':>8} {'Pass99':>7}")
logger.info("─" * 75)
for _, row in kupiec_df.iterrows():
    logger.info(f"{str(row['window_start'])+' → '+str(row['window_end']):<25} "
          f"{row['breach_rate_95']:>6.2%} "
          f"{row['pval_95']:>8.4f} "
          f"{'PASS' if row['pass_95'] else 'FAIL':>7} "
          f"{row['breach_rate_99']:>6.2%} "
          f"{row['pval_99']:>8.4f} "
          f"{'PASS' if row['pass_99'] else 'FAIL':>7}")

n95 = kupiec_df["pass_95"].sum()
n99 = kupiec_df["pass_99"].sum()
total = len(kupiec_df)
logger.info(f"\n95% VaR windows passing: {n95}/{total}")
logger.info(f"99% VaR windows passing: {n99}/{total}")


# ── 9. SAVE CSV ───────────────────────────────────────────────────────────────
wf_daily = pd.DataFrame({
    "date"             : common_idx,
    "daily_loss"       : daily_loss.values,
    "var95"            : var95_series.values,
    "var99"            : var99_series.values,
    "hit95"            : hit95.values,
    "hit99"            : hit99.values,
    "rolling_breach95" : rolling_breach95.values,
    "rolling_breach99" : rolling_breach99.values,
    "uce95"            : uce95.values,
})
wf_daily.to_csv(
    os.path.join(BASE_DIR, "data", "walkforward_results.csv"), index=False
)
kupiec_df.to_csv(
    os.path.join(BASE_DIR, "data", "walkforward_kupiec_windows.csv"),
    index=False
)
logger.info(f"\nSaved → data/walkforward_results.csv")
logger.info(f"Saved → data/walkforward_kupiec_windows.csv")


# ── 10. PLOT ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor="#0f0f0f")
fig.suptitle(
    "GARCH Risk Engine — Walk-Forward VaR Validation\n"
    "Equal Weight Portfolio | ₹1 Crore | 60-day Rolling Window",
    color="white", fontsize=13, y=0.98
)

dates = common_idx

# Panel 1 — Daily loss vs VaR 95%
ax1 = axes[0]
ax1.fill_between(dates, daily_loss / 1000, 0,
                 where=(daily_loss > 0), alpha=0.4,
                 color="#ff6b6b", label="Daily Loss (₹K)")
ax1.plot(dates, var95_series / 1000, color="#ffd93d",
         linewidth=0.8, label="VaR 95% (₹K)")
ax1.plot(dates, var99_series / 1000, color="#a29bfe",
         linewidth=0.8, label="VaR 99% (₹K)", alpha=0.7)

breach_dates95 = dates[hit95.values == 1]
breach_vals95  = daily_loss[hit95.values == 1] / 1000
ax1.scatter(breach_dates95, breach_vals95,
            color="#ff3333", s=15, zorder=5, label="VaR 95% Breach")

ax1.set_facecolor("#0f0f0f")
ax1.set_title("Daily Portfolio Loss vs VaR", color="white", fontsize=10)
ax1.set_ylabel("₹ Thousands", color="white", fontsize=9)
ax1.tick_params(colors="white", labelsize=8)
ax1.legend(loc="upper right", facecolor="#1a1a1a",
           labelcolor="white", fontsize=8)
ax1.spines[["top","right","left","bottom"]].set_color("#333333")
ax1.grid(alpha=0.1, color="white")

# Panel 2 — Rolling 60-day breach rate
ax2 = axes[1]
ax2.plot(dates, rolling_breach95 * 100, color="#00d4aa",
         linewidth=1.2, label="60-day Breach Rate 95% VaR (%)")
ax2.plot(dates, rolling_breach99 * 100, color="#a29bfe",
         linewidth=1.0, label="60-day Breach Rate 99% VaR (%)", alpha=0.8)
ax2.axhline(5.0, color="#ffd93d", linewidth=1.0,
            linestyle="--", label="Expected 5%", alpha=0.8)
ax2.axhline(1.0, color="#a29bfe", linewidth=0.8,
            linestyle="--", label="Expected 1%", alpha=0.5)

# Shade crisis zone
ax2.fill_between(dates, 0, 100,
                 where=crisis_mask.fillna(False).values,
                 alpha=0.15, color="#ff6b6b", label="Crisis Regime")

ax2.set_facecolor("#0f0f0f")
ax2.set_title("Rolling 60-day Breach Rate — Regime Detection",
              color="white", fontsize=10)
ax2.set_ylabel("Breach Rate (%)", color="white", fontsize=9)
ax2.set_ylim(0, max(rolling_breach95.max() * 100 * 1.1, 10))
ax2.tick_params(colors="white", labelsize=8)
ax2.legend(loc="upper right", facecolor="#1a1a1a",
           labelcolor="white", fontsize=8)
ax2.spines[["top","right","left","bottom"]].set_color("#333333")
ax2.grid(alpha=0.1, color="white")

# Panel 3 — UCE heatmap (bar chart)
ax3 = axes[2]
colors_uce = ["#ff6b6b" if v > 0.05 else "#00d4aa"
              for v in uce95.fillna(0).values]
ax3.bar(dates, uce95.fillna(0) * 100, color=colors_uce,
        width=1.0, alpha=0.8)
ax3.axhline(5.0, color="#ffd93d", linewidth=1.0,
            linestyle="--", label="UCE = 5% threshold")
ax3.set_facecolor("#0f0f0f")
ax3.set_title("Unconditional Coverage Error — UCE (95% VaR)  "
              "|  Red = Model Dangerously Wrong",
              color="white", fontsize=10)
ax3.set_ylabel("UCE (%)", color="white", fontsize=9)
ax3.tick_params(colors="white", labelsize=8)
ax3.legend(loc="upper right", facecolor="#1a1a1a",
           labelcolor="white", fontsize=8)
ax3.spines[["top","right","left","bottom"]].set_color("#333333")
ax3.grid(alpha=0.1, color="white")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=45, ha="right", color="white", fontsize=7)

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(
    os.path.join(BASE_DIR, "assets", "walkforward.png"),
    dpi=150, bbox_inches="tight", facecolor="#0f0f0f"
)
plt.close()
logger.info("Saved → assets/walkforward.png")
