"""
Kupiec POF Backtest + Christoffersen Independence Test
Input:  data/vajra_var_history.csv
Output: data/kupiec_results.csv
        assets/kupiec_backtest.png
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
var_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_var_history.csv"),
    index_col=0, parse_dates=True
)

T = len(var_df)
logger.info(f"Loaded {T} days of VaR history\n")


# ── 2. KUPIEC POF TEST ────────────────────────────────────────────────────────
def kupiec_pof(actual_loss, var_series, confidence, T):
    p           = 1 - confidence
    N           = int((actual_loss > var_series).sum())
    breach_rate = N / T

    # Avoid log(0) edge cases
    N_adj   = max(N, 0.5)
    N_adj   = min(N_adj, T - 0.5)
    p_hat   = N_adj / T

    lr_pof  = -2 * (
        np.log((1 - p) ** (T - N_adj) * p ** N_adj) -
        np.log((1 - p_hat) ** (T - N_adj) * p_hat ** N_adj)
    )

    p_value   = 1 - stats.chi2.cdf(lr_pof, df=1)
    critical  = stats.chi2.ppf(0.95, df=1)   # always 3.841
    reject_h0 = lr_pof > critical

    return {
        "breaches"     : N,
        "breach_rate"  : breach_rate,
        "expected_rate": p,
        "LR_pof"       : float(lr_pof),
        "p_value"      : float(p_value),
        "critical_val" : float(critical),
        "reject_H0"    : bool(reject_h0),
        "verdict"      : "FAIL — model miscalibrated" if reject_h0 else "PASS — model acceptable"
    }


# ── 3. CHRISTOFFERSEN INDEPENDENCE TEST ───────────────────────────────────────
def christoffersen_independence(actual_loss, var_series):
    hits = (actual_loss > var_series).astype(int).values

    n00 = sum(1 for i in range(1, len(hits)) if hits[i-1]==0 and hits[i]==0)
    n01 = sum(1 for i in range(1, len(hits)) if hits[i-1]==0 and hits[i]==1)
    n10 = sum(1 for i in range(1, len(hits)) if hits[i-1]==1 and hits[i]==0)
    n11 = sum(1 for i in range(1, len(hits)) if hits[i-1]==1 and hits[i]==1)

    eps   = 1e-10
    pi01  = np.clip(n01 / (n00 + n01) if (n00 + n01) > 0 else 0, eps, 1-eps)
    pi11  = np.clip(n11 / (n10 + n11) if (n10 + n11) > 0 else 0, eps, 1-eps)
    pi    = np.clip((n01 + n11) / (n00 + n01 + n10 + n11), eps, 1-eps)

    lr_ind = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi) -
        n00 * np.log(1 - pi01) - n01 * np.log(pi01) -
        n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )

    p_value   = 1 - stats.chi2.cdf(lr_ind, df=1)
    reject_h0 = lr_ind > stats.chi2.ppf(0.95, df=1)

    return {
        "n00"       : n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01"      : float(pi01),
        "pi11"      : float(pi11),
        "LR_ind"    : float(lr_ind),
        "p_value"   : float(p_value),
        "reject_H0" : bool(reject_h0),
        "verdict"   : "FAIL — breaches cluster" if reject_h0 else "PASS — breaches independent"
    }


# ── 4. RUN TESTS ──────────────────────────────────────────────────────────────
tests = [
    ("VaR_95", 0.95, "95% VaR"),
    ("VaR_99", 0.99, "99% VaR"),
]

all_results = []

logger.info("=" * 60)
logger.info("KUPIEC POF BACKTEST")
logger.info("=" * 60)

for var_col, conf, label in tests:
    r = kupiec_pof(var_df["actual_loss"], var_df[var_col], conf, T)

    logger.info(f"\n{label}")
    logger.info(f"  Observed breaches : {r['breaches']} / {T}  ({r['breach_rate']:.2%})")
    logger.info(f"  Expected rate     : {r['expected_rate']:.2%}")
    logger.info(f"  LR_pof            : {r['LR_pof']:.4f}  (critical = {r['critical_val']:.3f})")
    logger.info(f"  p-value           : {r['p_value']:.4f}")
    logger.info(f"  Result            : {r['verdict']}")

    all_results.append({"test": f"Kupiec_{label}", **r})

logger.info("\n" + "=" * 60)
logger.info("CHRISTOFFERSEN INDEPENDENCE TEST")
logger.info("=" * 60)

for var_col, conf, label in tests:
    r = christoffersen_independence(var_df["actual_loss"], var_df[var_col])

    logger.info(f"\n{label}")
    logger.info(f"  Transition counts : n00={r['n00']}  n01={r['n01']}  n10={r['n10']}  n11={r['n11']}")
    logger.info(f"  P(breach | no breach yesterday) : {r['pi01']:.4f}")
    logger.info(f"  P(breach | breach yesterday)    : {r['pi11']:.4f}")
    logger.info(f"  LR_ind            : {r['LR_ind']:.4f}  (critical = 3.841)")
    logger.info(f"  p-value           : {r['p_value']:.4f}")
    logger.info(f"  Result            : {r['verdict']}")

    all_results.append({"test": f"Christoffersen_{label}", **r})


# ── 5. SAVE CSV ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(BASE_DIR, "data", "kupiec_results.csv"), index=False)
logger.info(f"\nSaved → data/kupiec_results.csv")


# ── 6. PLOT ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                          facecolor="#0f0f0f", sharex=True)
fig.suptitle("Kupiec Backtest — VaR Breach Analysis\nGARCH Risk Engine | NSE Large-Caps",
             color="white", fontsize=13, y=0.98)

plot_tests = [
    ("VaR_95", "95% VaR", "#00d4aa"),
    ("VaR_99", "99% VaR", "#ff6b6b"),
]

for ax, (var_col, label, color) in zip(axes, plot_tests):
    breaches = var_df[var_df["actual_loss"] > var_df[var_col]]

    ax.plot(var_df.index, var_df["actual_loss"] / 1000,
            color="#ffffff", linewidth=0.5, alpha=0.6, label="Actual Loss (₹K)")
    ax.plot(var_df.index, var_df[var_col] / 1000,
            color=color, linewidth=1.0, label=f"{label} (₹K)")
    ax.fill_between(var_df.index, var_df[var_col] / 1000,
                    alpha=0.1, color=color)

    if len(breaches) > 0:
        ax.scatter(breaches.index, breaches["actual_loss"] / 1000,
                   color="#ff3333", s=20, zorder=5, alpha=0.9,
                   label=f"Breaches: {len(breaches)} ({len(breaches)/T:.1%})")

    ax.set_facecolor("#0f0f0f")
    ax.set_ylabel("Loss / VaR (₹ Thousands)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor="#1a1a1a",
              labelcolor="white", fontsize=9)
    ax.spines[["top","right","left","bottom"]].set_color("#333333")
    ax.grid(alpha=0.15, color="white")

axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(axes[1].xaxis.get_majorticklabels(),
         rotation=45, ha="right", color="white", fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "assets", "kupiec_backtest.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
logger.info("Saved → assets/kupiec_backtest.png")
