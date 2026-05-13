"""
GARCH Diagnostics
Ljung-Box Test + ARCH LM Test on standardized residuals
Input:  data/vajra_z.pkl         (1479 x 14) standardized residuals
Output: data/diagnostics_results.csv
        assets/garch_diagnostics.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import logging
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "data", "vajra_z.pkl"), "rb") as f:
    z_df = pickle.load(f)

stocks = list(z_df.columns)
T      = len(z_df)
logger.info(f"Standardized residuals loaded: {z_df.shape}")
logger.info(f"Stocks: {stocks}\n")


# ── 2. RUN DIAGNOSTICS PER STOCK ──────────────────────────────────────────────
LAGS = [5, 10, 20]   # short / medium / long autocorrelation

results = []

logger.info("=" * 70)
logger.info("GARCH DIAGNOSTICS — LJUNG-BOX + ARCH LM")
logger.info("=" * 70)

for stock in stocks:
    z = z_df[stock].dropna().values.astype(np.float64)

    # ── Ljung-Box on z_t (raw standardized residuals) ────────────────────────
    lb = acorr_ljungbox(z, lags=LAGS, return_df=True)
    lb_stats  = lb["lb_stat"].values
    lb_pvals  = lb["lb_pvalue"].values
    lb_pass   = all(lb_pvals > 0.05)   # pass if ALL lags pass

    # ── ARCH LM on z_t² (squared residuals) ──────────────────────────────────
    arch_results = []
    for lag in LAGS:
        lm_stat, lm_pval, f_stat, f_pval = het_arch(z, nlags=lag)
        arch_results.append((lm_stat, lm_pval))

    arch_pvals = [r[1] for r in arch_results]
    arch_pass  = all(p > 0.05 for p in arch_pvals)

    overall = "✅ PASS" if (lb_pass and arch_pass) else "⚠ FAIL"

    logger.info(f"\n{stock}")
    logger.info(f"  Ljung-Box  (lags 5/10/20) p-values: "
          f"{lb_pvals[0]:.4f} / {lb_pvals[1]:.4f} / {lb_pvals[2]:.4f}  "
          f"→ {'PASS' if lb_pass else 'FAIL'}")
    logger.info(f"  ARCH LM    (lags 5/10/20) p-values: "
          f"{arch_pvals[0]:.4f} / {arch_pvals[1]:.4f} / {arch_pvals[2]:.4f}  "
          f"→ {'PASS' if arch_pass else 'FAIL'}")
    logger.info(f"  Overall: {overall}")

    results.append({
        "stock"          : stock,
        "lb_pval_lag5"   : lb_pvals[0],
        "lb_pval_lag10"  : lb_pvals[1],
        "lb_pval_lag20"  : lb_pvals[2],
        "lb_pass"        : lb_pass,
        "arch_pval_lag5" : arch_pvals[0],
        "arch_pval_lag10": arch_pvals[1],
        "arch_pval_lag20": arch_pvals[2],
        "arch_pass"      : arch_pass,
        "overall_pass"   : lb_pass and arch_pass,
    })

results_df = pd.DataFrame(results)

# ── 3. SUMMARY ────────────────────────────────────────────────────────────────
n_pass_lb   = results_df["lb_pass"].sum()
n_pass_arch = results_df["arch_pass"].sum()
n_pass_both = results_df["overall_pass"].sum()
N           = len(stocks)

logger.info(f"\n{'='*70}")
logger.info(f"SUMMARY")
logger.info(f"{'='*70}")
logger.info(f"Ljung-Box PASS  : {n_pass_lb}  / {N} stocks")
logger.info(f"ARCH LM PASS    : {n_pass_arch} / {N} stocks")
logger.info(f"Both tests PASS : {n_pass_both} / {N} stocks")

fails = results_df[~results_df["overall_pass"]]["stock"].tolist()
if fails:
    logger.info(f"Failed stocks   : {fails}")
else:
    logger.info("All stocks pass both diagnostics ✅")


# ── 4. SAVE CSV ───────────────────────────────────────────────────────────────
results_df.to_csv(
    os.path.join(BASE_DIR, "data", "diagnostics_results.csv"), index=False
)
logger.info(f"\nSaved → data/diagnostics_results.csv")


# ── 5. PLOT ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="#0f0f0f")
fig.suptitle("GARCH Diagnostics — Ljung-Box & ARCH LM p-values\nGARCH Risk Engine | 14 NSE Stocks",
             color="white", fontsize=13, y=0.98)

x      = np.arange(N)
width  = 0.35
thresh = 0.05   # significance line

# Panel 1 — Ljung-Box p-values at lag 10
ax1 = axes[0]
bars = ax1.bar(x, results_df["lb_pval_lag10"],
               width=0.6, color=[
                   "#00d4aa" if p else "#ff6b6b"
                   for p in results_df["lb_pass"]
               ], alpha=0.85)
ax1.axhline(thresh, color="#ffd93d", linewidth=1.2,
            linestyle="--", label=f"p = {thresh} threshold")
ax1.set_facecolor("#0f0f0f")
ax1.set_title("Ljung-Box Test (lag 10) — p-values  |  Green = PASS  Red = FAIL",
              color="white", fontsize=10)
ax1.set_ylabel("p-value", color="white", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(stocks, rotation=45, ha="right",
                    color="white", fontsize=9)
ax1.tick_params(colors="white")
ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
ax1.spines[["top","right","left","bottom"]].set_color("#333333")
ax1.grid(alpha=0.15, color="white", axis="y")
ax1.set_ylim(0, 1.05)

# Panel 2 — ARCH LM p-values at lag 10
ax2 = axes[1]
ax2.bar(x, results_df["arch_pval_lag10"],
        width=0.6, color=[
            "#00d4aa" if p else "#ff6b6b"
            for p in results_df["arch_pass"]
        ], alpha=0.85)
ax2.axhline(thresh, color="#ffd93d", linewidth=1.2,
            linestyle="--", label=f"p = {thresh} threshold")
ax2.set_facecolor("#0f0f0f")
ax2.set_title("ARCH LM Test (lag 10) — p-values  |  Green = PASS  Red = FAIL",
              color="white", fontsize=10)
ax2.set_ylabel("p-value", color="white", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(stocks, rotation=45, ha="right",
                    color="white", fontsize=9)
ax2.tick_params(colors="white")
ax2.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
ax2.spines[["top","right","left","bottom"]].set_color("#333333")
ax2.grid(alpha=0.15, color="white", axis="y")
ax2.set_ylim(0, 1.05)

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "assets", "garch_diagnostics.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
logger.info("Saved → assets/garch_diagnostics.png")
