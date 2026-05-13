"""
Monte Carlo Portfolio VaR (Cholesky)

Inputs:
    data/vajra_dcc_cov.pkl
    data/vajra_var_history.csv

Outputs:
    data/mc_var_results.csv
    assets/mc_var.png
"""

import os
import pickle
import numpy as np
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. LOAD INPUTS ─────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "data", "vajra_dcc_cov.pkl"), "rb") as f:
    cov_data = pickle.load(f)

cov_cube = cov_data["cov"]
dates    = pd.Index(cov_data["dates"])
stocks   = cov_data["stocks"]

var_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_var_history.csv"),
    index_col=0,
    parse_dates=True
)

# Align var_df to DCC dates just to be safe
var_df = var_df.reindex(dates)

T, N, _ = cov_cube.shape
weights = np.full(N, 1.0 / N, dtype=np.float64)

PORTFOLIO_VALUE = 10_000_000
N_SIMS = 10_000
SEED = 42

logger.info(f"Covariance cube loaded: {cov_cube.shape}")
logger.info(f"Stocks: {stocks}")
logger.info(f"Portfolio value: Rs{PORTFOLIO_VALUE:,.0f}")
logger.info(f"Simulations: {N_SIMS:,}")


# ── 2. PICK DATE TO EVALUATE ──────────────────────────────────
# Use latest date in the DCC cube
eval_idx  = T - 1
eval_date = dates[eval_idx]
Sigma_t   = np.array(cov_cube[eval_idx], dtype=np.float64)

logger.info(f"\nEvaluation date: {str(eval_date)[:10]}")


# ── 3. CLEAN COVARIANCE MATRIX ────────────────────────────────
# Make symmetric
Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)

# If tiny numerical issue makes it non-PSD, repair it
eigvals = np.linalg.eigvalsh(Sigma_t)
min_eig = eigvals.min()
if min_eig < 0:
    Sigma_t += np.eye(N) * (abs(min_eig) + 1e-10)

logger.info(f"Min eigenvalue after repair check: {np.linalg.eigvalsh(Sigma_t).min():.12f}")


# ── 4. CHOLESKY DECOMPOSITION ─────────────────────────────────
# Sigma_t = L @ L.T
L = np.linalg.cholesky(Sigma_t)
logger.info(f"Cholesky factor shape: {L.shape}")


# ── 5. GENERATE INDEPENDENT NORMAL SHOCKS ─────────────────────
rng = np.random.default_rng(SEED)

# Shape: (N, N_SIMS)
# Each column = one scenario of N independent shocks
Z = rng.standard_normal((N, N_SIMS))


# ── 6. TRANSFORM TO CORRELATED SHOCKS ─────────────────────────
# If eps ~ N(0, I), then L @ eps ~ N(0, Sigma_t)
L = np.asarray(L, dtype=np.float64, order="C")
Z = np.asarray(Z, dtype=np.float64, order="C")
weights = np.asarray(weights, dtype=np.float64, order="C")

with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    sim_asset_returns_pct = np.dot(L, Z)
    sim_port_returns_pct = np.dot(weights, sim_asset_returns_pct)

logger.info("sim_asset finite:", np.isfinite(sim_asset_returns_pct).all())
logger.info("sim_port finite :", np.isfinite(sim_port_returns_pct).all())
logger.info(np.isfinite(L).all(), np.isfinite(Z).all())
sim_asset_returns_pct = L @ Z   # still in % return units

# Portfolio return for each scenario
# weights: (N,), sim_asset_returns_pct: (N, N_SIMS)
sim_port_returns_pct = weights @ sim_asset_returns_pct

# Convert to decimal return
sim_port_returns_dec = sim_port_returns_pct / 100.0

# Portfolio P&L and loss
sim_pnl  = sim_port_returns_dec * PORTFOLIO_VALUE
sim_loss = -sim_pnl

logger.info("\n--- SIMULATION DIAGNOSTICS ---")
logger.info(f"Simulated portfolio mean return (%): {sim_port_returns_pct.mean():.6f}")
logger.info(f"Simulated portfolio vol (%):         {sim_port_returns_pct.std(ddof=1):.6f}")
logger.info(f"Simulated worst loss (Rs):           {sim_loss.max():,.0f}")
logger.info(f"Simulated best gain  (Rs):           {sim_pnl.max():,.0f}")


# ── 7. MONTE CARLO VaR / CVaR ─────────────────────────────────
mc_var95  = np.percentile(sim_loss, 95)
mc_var99  = np.percentile(sim_loss, 99)

mc_cvar95 = sim_loss[sim_loss >= mc_var95].mean()
mc_cvar99 = sim_loss[sim_loss >= mc_var99].mean()


# ── 8. PARAMETRIC VaR COMPARISON ──────────────────────────────
param_row = var_df.loc[eval_date]

param_var95  = float(param_row["VaR_95"])
param_var99  = float(param_row["VaR_99"])
param_cvar95 = float(param_row["CVaR_95"])
param_cvar99 = float(param_row["CVaR_99"])

comparison_df = pd.DataFrame({
    "metric": ["VaR_95", "VaR_99", "CVaR_95", "CVaR_99"],
    "parametric_rs": [param_var95, param_var99, param_cvar95, param_cvar99],
    "mc_rs": [mc_var95, mc_var99, mc_cvar95, mc_cvar99]
})

comparison_df["diff_rs"] = comparison_df["mc_rs"] - comparison_df["parametric_rs"]
comparison_df["diff_pct"] = 100 * comparison_df["diff_rs"] / comparison_df["parametric_rs"]

logger.info("\n--- PARAMETRIC VS MONTE CARLO ---")
for _, row in comparison_df.iterrows():
    logger.info(
        f"{row['metric']:<8}  "
        f"Param=Rs{row['parametric_rs']:>10,.0f}  "
        f"MC=Rs{row['mc_rs']:>10,.0f}  "
        f"Diff={row['diff_pct']:>8.2f}%"
    )


# ── 9. SAVE RESULTS CSV ───────────────────────────────────────
results_df = pd.DataFrame([{
    "date": eval_date,
    "n_sims": N_SIMS,
    "portfolio_value": PORTFOLIO_VALUE,
    "mc_var95": mc_var95,
    "mc_var99": mc_var99,
    "mc_cvar95": mc_cvar95,
    "mc_cvar99": mc_cvar99,
    "param_var95": param_var95,
    "param_var99": param_var99,
    "param_cvar95": param_cvar95,
    "param_cvar99": param_cvar99,
    "mc_mean_return_pct": sim_port_returns_pct.mean(),
    "mc_vol_pct": sim_port_returns_pct.std(ddof=1),
}])

comparison_save = comparison_df.copy()
comparison_save["date"] = eval_date

results_df.to_csv(
    os.path.join(BASE_DIR, "data", "mc_var_results.csv"),
    index=False
)

comparison_save.to_csv(
    os.path.join(BASE_DIR, "data", "mc_var_comparison.csv"),
    index=False
)


# ── 10. PLOT LOSS DISTRIBUTION ────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0f0f0f")
ax.set_facecolor("#0f0f0f")

ax.hist(
    sim_loss / 1000,
    bins=80,
    color="#5dade2",
    alpha=0.75,
    edgecolor="none"
)

ax.axvline(mc_var95 / 1000, color="#ffd93d", lw=2, ls="--",
           label=f"MC VaR 95% = Rs{mc_var95:,.0f}")
ax.axvline(mc_var99 / 1000, color="#ff6b6b", lw=2, ls="--",
           label=f"MC VaR 99% = Rs{mc_var99:,.0f}")
ax.axvline(param_var95 / 1000, color="#00d4aa", lw=2,
           label=f"Parametric VaR 95% = Rs{param_var95:,.0f}")
ax.axvline(param_var99 / 1000, color="#a29bfe", lw=2,
           label=f"Parametric VaR 99% = Rs{param_var99:,.0f}")

ax.set_title(
    f"Monte Carlo Portfolio Loss Distribution — {str(eval_date)[:10]}\n"
    f"Equal Weight Portfolio | 10,000 Scenarios | Rs1 Crore",
    color="white",
    fontsize=13
)
ax.set_xlabel("Simulated One-Day Loss (Rs Thousands)", color="white")
ax.set_ylabel("Frequency", color="white")
ax.tick_params(colors="white")
ax.spines[["top", "right", "left", "bottom"]].set_color("#333333")
ax.grid(alpha=0.12, color="white")
ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "assets", "mc_var.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="#0f0f0f"
)
plt.close()


logger.info("\nSaved → data/mc_var_results.csv")
logger.info("Saved → data/mc_var_comparison.csv")
logger.info("Saved → assets/mc_var.png")
