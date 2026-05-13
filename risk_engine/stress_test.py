"""
Historical Stress Testing
Input:  data/vajra_returns.csv
        data/vajra_var_history.csv
Output: data/stress_results.csv
        assets/stress_test.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
logger = logging.getLogger(__name__)

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
weights = np.full(N, 1.0 / N, dtype=np.float64)
PORTFOLIO_VALUE = 10_000_000

logger.info(f"Returns loaded  : {returns_df.shape}")
logger.info(f"VaR history     : {var_df.shape}\n")


# ── 2. SCENARIOS ──────────────────────────────────────────────────────────────
SCENARIOS = {
    "COVID_Crash"       : ("2020-02-20", "2020-03-23"),
    "COVID_Recovery"    : ("2020-03-24", "2020-06-01"),
    "Adani_Hindenburg"  : ("2023-01-24", "2023-02-27"),
    "ILFS_Crisis"       : ("2018-09-21", "2018-10-19"),
    "Fed_Rate_Shock"    : ("2022-06-01", "2022-06-30"),
}


# ── 3. STRESS TEST FUNCTION ───────────────────────────────────────────────────
def run_stress_test(scenario_name, start, end):
    # Slice returns and VaR to scenario window
    r_window   = returns_df.loc[start:end].copy()
    var_window = var_df.loc[start:end].copy()

    if len(r_window) == 0:
        logger.info(f"  ⚠ No data for {scenario_name} ({start} to {end})")
        return None

    # Align — only days present in both
    common_idx = r_window.index.intersection(var_window.index)
    r_window   = r_window.loc[common_idx]
    var_window = var_window.loc[common_idx]

    # Daily portfolio P&L
    daily_returns = np.einsum('ij,j->i', r_window[stocks].values, weights)
    daily_pnl     = daily_returns * PORTFOLIO_VALUE
    daily_loss    = -daily_pnl

    # Cumulative P&L (starts at 0)
    cumulative_pnl = daily_pnl.cumsum()

    # Metrics
    peak_loss_idx      = daily_loss.argmax()
    peak_loss_day      = r_window.index[peak_loss_idx]
    peak_loss_value    = daily_loss[peak_loss_idx]
    cumulative_loss    = -cumulative_pnl[-1]
    total_days         = len(r_window)

    # Breach count vs VaR 95
    if "VaR_95" in var_window.columns:
        breaches_95 = (daily_loss > var_window["VaR_95"].values).sum()
        avg_var95   = var_window["VaR_95"].mean()
    else:
        breaches_95 = np.nan
        avg_var95   = np.nan

    # Max excess over VaR on single day
    if "VaR_95" in var_window.columns:
        excess       = daily_loss - var_window["VaR_95"].values
        max_excess   = excess.max()
        max_excess_day = r_window.index[excess.argmax()]
    else:
        max_excess     = np.nan
        max_excess_day = None

    return {
        "scenario"         : scenario_name,
        "start"            : start,
        "end"              : end,
        "total_days"       : total_days,
        "peak_loss_day"    : str(peak_loss_day)[:10],
        "peak_loss"        : peak_loss_value,
        "cumulative_loss"  : cumulative_loss,
        "avg_var95"        : avg_var95,
        "breaches_95"      : breaches_95,
        "breach_rate"      : breaches_95 / total_days if total_days > 0 else np.nan,
        "max_excess_over_var" : max_excess,
        "max_excess_day"   : str(max_excess_day)[:10] if max_excess_day else None,
        # Store series for plotting
        "_dates"           : r_window.index,
        "_daily_loss"      : daily_loss,
        "_cumulative_pnl"  : cumulative_pnl,
        "_var95"           : var_window["VaR_95"].values if "VaR_95" in var_window.columns else None,
    }


# ── 4. RUN ALL SCENARIOS ──────────────────────────────────────────────────────
logger.info("=" * 65)
logger.info("STRESS TEST RESULTS")
logger.info("=" * 65)

results = []
plot_data = {}

for name, (start, end) in SCENARIOS.items():
    r = run_stress_test(name, start, end)
    if r is None:
        continue

    logger.info(f"\n{'─'*65}")
    logger.info(f"Scenario        : {r['scenario']}")
    logger.info(f"Window          : {r['start']} → {r['end']}  ({r['total_days']} days)")
    logger.info(f"Peak single-day loss : ₹{r['peak_loss']:>12,.0f}  on {r['peak_loss_day']}")
    logger.info(f"Cumulative loss      : ₹{r['cumulative_loss']:>12,.0f}")
    logger.info(f"Avg VaR 95%          : ₹{r['avg_var95']:>12,.0f}")
    logger.info(f"VaR 95% breaches     : {r['breaches_95']} / {r['total_days']}  ({r['breach_rate']:.1%})")
    logger.info(f"Max excess over VaR  : ₹{r['max_excess_over_var']:>12,.0f}  on {r['max_excess_day']}")

    # Store plot data separately
    plot_data[name] = {
        "dates"          : r.pop("_dates"),
        "daily_loss"     : r.pop("_daily_loss"),
        "cumulative_pnl" : r.pop("_cumulative_pnl"),
        "var95"          : r.pop("_var95"),
    }
    results.append(r)


# ── 5. SAVE CSV ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(BASE_DIR, "data", "stress_results.csv"), index=False
)
logger.info(f"\n\nSaved → data/stress_results.csv")


# ── 6. PLOT ───────────────────────────────────────────────────────────────────
n_scenarios = len(plot_data)
fig, axes   = plt.subplots(n_scenarios, 2,
                            figsize=(16, 4 * n_scenarios),
                            facecolor="#0f0f0f")
fig.suptitle("GARCH Risk Engine — Historical Stress Tests\nEqual Weight Portfolio | ₹1 Crore | NSE Large-Caps",
             color="white", fontsize=13, y=1.01)

COLORS = {
    "COVID_Crash"      : "#ff6b6b",
    "COVID_Recovery"   : "#00d4aa",
    "Adani_Hindenburg" : "#ffd93d",
    "ILFS_Crisis"      : "#ff9f43",
    "Fed_Rate_Shock"   : "#a29bfe",
}

for row_idx, (name, pd_) in enumerate(plot_data.items()):
    color  = COLORS.get(name, "#ffffff")
    dates  = pd_["dates"]
    dloss  = pd_["daily_loss"]
    cpnl   = pd_["cumulative_pnl"]
    var95  = pd_["var95"]

    # Left panel — daily loss vs VaR
    ax_l = axes[row_idx, 0]
    ax_l.bar(dates, dloss / 1000, color=color, alpha=0.7, width=0.8, label="Daily Loss (₹K)")
    if var95 is not None:
        ax_l.plot(dates, var95 / 1000, color="#ffffff",
                  linewidth=1.2, linestyle="--", label="VaR 95% (₹K)")
        # Mark breaches
        breach_mask = dloss > var95
        if breach_mask.any():
            ax_l.scatter(dates[breach_mask], dloss[breach_mask] / 1000,
                         color="#ff3333", s=40, zorder=5, label="Breach")

    ax_l.set_facecolor("#0f0f0f")
    ax_l.set_title(f"{name} — Daily Loss", color="white", fontsize=10)
    ax_l.set_ylabel("₹ Thousands", color="white", fontsize=9)
    ax_l.tick_params(colors="white", labelsize=8)
    ax_l.legend(loc="upper right", facecolor="#1a1a1a",
                labelcolor="white", fontsize=8)
    ax_l.spines[["top","right","left","bottom"]].set_color("#333333")
    ax_l.grid(alpha=0.15, color="white")
    ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax_l.xaxis.get_majorticklabels(),
             rotation=45, ha="right", color="white", fontsize=7)

    # Right panel — cumulative P&L
    ax_r = axes[row_idx, 1]
    ax_r.fill_between(dates, cpnl / 1000, 0,
                      where=(cpnl < 0), alpha=0.4, color="#ff6b6b")
    ax_r.fill_between(dates, cpnl / 1000, 0,
                      where=(cpnl >= 0), alpha=0.4, color="#00d4aa")
    ax_r.plot(dates, cpnl / 1000, color=color, linewidth=1.2)
    ax_r.axhline(0, color="#ffffff", linewidth=0.5, linestyle="--", alpha=0.5)

    ax_r.set_facecolor("#0f0f0f")
    ax_r.set_title(f"{name} — Cumulative P&L", color="white", fontsize=10)
    ax_r.set_ylabel("₹ Thousands", color="white", fontsize=9)
    ax_r.tick_params(colors="white", labelsize=8)
    ax_r.spines[["top","right","left","bottom"]].set_color("#333333")
    ax_r.grid(alpha=0.15, color="white")
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax_r.xaxis.get_majorticklabels(),
             rotation=45, ha="right", color="white", fontsize=7)

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "assets", "stress_test.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
logger.info("Saved → assets/stress_test.png")
