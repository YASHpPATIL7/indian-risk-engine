"""
DCC Engine
Inputs:  data/vajra_z.pkl       (T × N standardised shocks from GARCH)
         data/vajra_sigma.pkl   (T × N conditional vols from GARCH)
Outputs: data/vajra_dcc_rho.pkl     (T × N × N correlation cubes)
         data/vajra_dcc_cov.pkl     (T × N × N covariance cubes)
         data/vajra_dcc_params.csv  (a, b, log-likelihood)
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
import logging
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. LOAD GARCH OUTPUTS ─────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "data", "vajra_z.pkl"), "rb") as f:
    z_df = pickle.load(f)

with open(os.path.join(BASE_DIR, "data", "vajra_sigma.pkl"), "rb") as f:
    sigma_df = pickle.load(f)

Z = z_df.values          # shape (T, N)
SIGMA = sigma_df.values  # shape (T, N)
T, N = Z.shape
stocks = z_df.columns.tolist()
dates  = z_df.index

logger.info(f"Z matrix loaded:     {Z.shape}")
logger.info(f"Sigma matrix loaded: {SIGMA.shape}")
logger.info(f"Stocks: {stocks}\n")

# ── 2. COMPUTE Q_BAR (unconditional correlation) ──────────────────────────────
# Q_bar = (1/T) * sum of z_t z_t' for all t
# This is just the sample correlation matrix of standardised shocks
Q_bar = np.cov(Z.T)   # (N × N)
# np.cov already divides by T-1 and demeans — correct for Q_bar estimation

logger.info("Q_bar (unconditional correlation matrix):")
logger.info(np.round(Q_bar, 4))
logger.info()

# ── 3. DCC LOG-LIKELIHOOD FUNCTION ───────────────────────────────────────────
def dcc_loglikelihood(params, Z, Q_bar):
    """
    Compute negative log-likelihood for DCC(1,1).
    params = [a, b]
    Constraint: a > 0, b > 0, a + b < 1
    """
    a, b = params

    if a <= 0 or b <= 0 or a + b >= 1:
        return 1e10  # infeasible — return large penalty

    T, N = Z.shape
    Q_t = Q_bar.copy()
    ll  = 0.0

    for t in range(1, T):
        z_prev  = Z[t-1, :].reshape(-1, 1)          # (N × 1)
        zt_ztT  = z_prev @ z_prev.T                  # (N × N) outer product

        # DCC recursion
        Q_t = Q_bar * (1 - a - b) + a * zt_ztT + b * Q_t

        # Normalise Q_t → R_t (correlation matrix)
        diag_sqrt = np.sqrt(np.diag(Q_t))            # (N,)
        denom     = np.outer(diag_sqrt, diag_sqrt)   # (N × N)
        R_t       = Q_t / denom

        # DCC log-likelihood contribution at time t
        z_t    = Z[t, :].reshape(-1, 1)
        sign, logdet = np.linalg.slogdet(R_t)
        if sign <= 0:
            return 1e10  # R_t not positive definite — skip

        R_inv  = np.linalg.inv(R_t)
        ll    += -0.5 * (logdet + float(z_t.T @ R_inv @ z_t))

    return -ll  # return NEGATIVE log-likelihood for minimisation


# ── 4. FIT DCC PARAMETERS (a, b) ─────────────────────────────────────────────
logger.info("--- FITTING DCC PARAMETERS ---")
logger.info("Optimising a and b via maximum likelihood...\n")

# Starting values — typical for equity markets
a0, b0 = 0.05, 0.93

result = minimize(
    dcc_loglikelihood,
    x0     = [a0, b0],
    args   = (Z, Q_bar),
    method = 'L-BFGS-B',
    bounds = [(1e-6, 0.5), (1e-6, 0.9999)],
    options= {'maxiter': 500, 'ftol': 1e-12}
)

a_hat, b_hat = result.x
ll_val       = -result.fun   # convert back to positive LL

logger.info(f"  a (shock sensitivity) : {a_hat:.6f}")
logger.info(f"  b (correlation decay) : {b_hat:.6f}")
logger.info(f"  a + b (persistence)   : {a_hat + b_hat:.6f}")
logger.info(f"  Log-likelihood        : {ll_val:.4f}")
logger.info(f"  Converged             : {result.success}")
logger.info(f"  Message               : {result.message}\n")

if not result.success:
    logger.info("⚠ WARNING: Optimiser did not converge. Results may be unreliable.")
    logger.info("  Trying with different starting values...\n")
    # Fallback starting values
    result2 = minimize(
        dcc_loglikelihood,
        x0     = [0.02, 0.95],
        args   = (Z, Q_bar),
        method = 'L-BFGS-B',
        bounds = [(1e-6, 0.5), (1e-6, 0.9999)],
        options= {'maxiter': 1000}
    )
    if -result2.fun > ll_val:
        result  = result2
        a_hat, b_hat = result.x
        ll_val       = -result.fun
        logger.info(f"  Better solution found: a={a_hat:.6f}, b={b_hat:.6f}")


# ── 5. GENERATE FULL DCC MATRICES ─────────────────────────────────────────────
logger.info("--- GENERATING CORRELATION + COVARIANCE CUBES ---")

# Normalise Q_bar to proper correlation matrix
std_qbar = np.sqrt(np.diag(Q_bar))
Q_bar_normalised = Q_bar / np.outer(std_qbar, std_qbar)
Q_bar = Q_bar_normalised

rho_cube = np.zeros((T, N, N))   # T × N × N correlation matrices
cov_cube = np.zeros((T, N, N))   # T × N × N covariance matrices
Q_t      = Q_bar.copy()


for t in range(T):

    if t == 0:
        # Day 0: use unconditional correlation
        rho_cube[0] = np.corrcoef(Z.T)
    else:
        z_prev = Z[t-1, :].reshape(-1, 1)
        zt_ztT = z_prev @ z_prev.T

        # DCC recursion
        Q_t = Q_bar * (1 - a_hat - b_hat) + a_hat * zt_ztT + b_hat * Q_t

        # Normalise → ρ_t
        diag_sqrt   = np.sqrt(np.diag(Q_t))
        denom       = np.outer(diag_sqrt, diag_sqrt)
        rho_cube[t] = np.clip(Q_t / denom, -1.0, 1.0)
        np.fill_diagonal(rho_cube[t], 1.0)

    # Σ_t = D_t × ρ_t × D_t
    sigma_t        = SIGMA[t, :]                   # (N,) vol vector
    D_t            = np.diag(sigma_t)              # (N × N) diagonal
    cov_cube[t]    = D_t @ rho_cube[t] @ D_t       # (N × N) covariance

logger.info(f"rho_cube shape: {rho_cube.shape}   (T × N × N correlation)")
logger.info(f"cov_cube shape: {cov_cube.shape}   (T × N × N covariance)\n")



# ── 6. VALIDATION ─────────────────────────────────────────────────────────────
logger.info("--- DCC VALIDATION ---\n")

# 6a. Check correlation matrices are valid (diagonal=1, values in [-1,1])
sample_days = [0, 260, 500, 1000, T-1]   # spot check 5 days
for t in sample_days:
    R = rho_cube[t]
    diag_ok   = np.allclose(np.diag(R), 1.0, atol=1e-6)
    bounds_ok = (R.min() >= -1.0) and (R.max() <= 1.0)
    sym_ok    = np.allclose(R, R.T, atol=1e-8)
    date_str  = str(dates[t])[:10]
    logger.info(f"  Day {t:4d} ({date_str}) | diag=1: {diag_ok} | "
          f"bounds[-1,1]: {bounds_ok} | symmetric: {sym_ok}")

# 6b. Mean correlation over time per pair
logger.info("\n--- AVERAGE PAIRWISE CORRELATIONS (top 10 pairs) ---")
mean_corr = rho_cube.mean(axis=0)   # average across T → (N × N)
pairs = []
for i in range(N):
    for j in range(i+1, N):
        pairs.append((stocks[i], stocks[j], mean_corr[i, j]))

pairs.sort(key=lambda x: -x[2])
logger.info(f"{'Pair':<30} {'Mean ρ':>8}")
logger.info("-" * 40)
for s1, s2, rho in pairs[:10]:
    logger.info(f"  {s1:<12} — {s2:<12}  {rho:>8.4f}")

# 6c. Correlation on COVID crash day vs normal day
covid_idx  = z_df.index.get_loc('2020-03-23')
normal_idx = z_df.index.get_loc('2020-01-15')

logger.info(f"\n--- CORRELATION: COVID DAY ({str(dates[covid_idx])[:10]}) vs "
      f"NORMAL DAY ({str(dates[normal_idx])[:10]}) ---")
logger.info(f"{'Pair':<30} {'Normal ρ':>10} {'COVID ρ':>10} {'Δρ':>8}")
logger.info("-" * 60)
for i in range(N):
    for j in range(i+1, N):
        rho_n = rho_cube[normal_idx, i, j]
        rho_c = rho_cube[covid_idx,  i, j]
        delta = rho_c - rho_n
        if abs(delta) > 0.10:   # only show pairs where COVID changed ρ by >10%
            logger.info(f"  {stocks[i]:<12} — {stocks[j]:<12}  "
                  f"{rho_n:>10.4f} {rho_c:>10.4f} {delta:>+8.4f}")

# 6d. Persistence interpretation
logger.info(f"\n--- DCC PERSISTENCE ---")
half_life = np.log(0.5) / np.log(a_hat + b_hat)
logger.info(f"  a + b = {a_hat + b_hat:.4f}")
logger.info(f"  Correlation shock half-life: {half_life:.1f} observations")
logger.info(f"  (At daily frequency = {half_life:.1f} trading days)")

# ── 7. SAVE OUTPUTS ───────────────────────────────────────────────────────────
logger.info("\n--- SAVING ---")

# Save DCC covariance cube for SHAP attribution
np.save(
    os.path.join(BASE_DIR, "data", "dcc_covariance.npy"),
    cov_cube
)
logger.info(f"  data/dcc_covariance.npy   → Σ_t cube  {cov_cube.shape}")

with open(os.path.join(BASE_DIR, "data", "vajra_dcc_rho.pkl"), "wb") as f:
    pickle.dump({"rho": rho_cube, "dates": dates, "stocks": stocks}, f)
# ... rest unchanged


# ── 7. SAVE OUTPUTS ───────────────────────────────────────────────────────────
logger.info("\n--- SAVING ---")

with open(os.path.join(BASE_DIR, "data", "vajra_dcc_rho.pkl"), "wb") as f:
    pickle.dump({"rho": rho_cube, "dates": dates, "stocks": stocks}, f)

with open(os.path.join(BASE_DIR, "data", "vajra_dcc_cov.pkl"), "wb") as f:
    pickle.dump({"cov": cov_cube, "dates": dates, "stocks": stocks}, f)

params_df = pd.DataFrame({
    "a":           [a_hat],
    "b":           [b_hat],
    "persistence": [a_hat + b_hat],
    "half_life":   [half_life],
    "log_lik":     [ll_val],
    "converged":   [result.success]
})
params_df.to_csv(
    os.path.join(BASE_DIR, "data", "vajra_dcc_params.csv"), index=False
)

logger.info(f"  data/vajra_dcc_rho.pkl    → ρ_t cube  {rho_cube.shape}")
logger.info(f"  data/vajra_dcc_cov.pkl    → Σ_t cube  {cov_cube.shape}")
logger.info(f"  data/vajra_dcc_params.csv → a, b, LL")
