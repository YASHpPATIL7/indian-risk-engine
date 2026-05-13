"""
Monte Carlo Options Pricing
GARCH-MC vs Black-Scholes comparison + Volatility Smile
Inputs:  data/vajra_garch_params.csv   (ω, α, γ, β — fitted on % returns)
         data/vajra_sigma.pkl          (last GARCH sigma — in % units e.g. 1.407)
Outputs: data/mc_results.csv
         assets/mc_options.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
logger = logging.getLogger(__name__)
from scipy.stats import norm
from scipy.optimize import brentq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. BLACK-SCHOLES PRICER ───────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    """sigma = annualised decimal e.g. 0.2234"""
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    """sigma = annualised decimal"""
    if sigma <= 0 or T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ── 2. IMPLIED VOL SOLVER ──────────────────────────────────────────────────────
def implied_vol(market_price, S, K, T, r, option_type="call", tol=1e-6):
    fn       = bs_call if option_type == "call" else bs_put
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price <= intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(
            lambda sigma: fn(S, K, T, r, sigma) - market_price,
            1e-6, 10.0, xtol=tol
        )
    except (ValueError, RuntimeError):
        return np.nan

# ── 3. GARCH-MC PRICER ────────────────────────────────────────────────────────
def garch_mc_price(S, K, T, r, sigma0_pct, omega, alpha, gamma, beta,
                   M=10000, option_type="call", seed=42):
    """
    Units
    -----
    sigma0_pct  : daily vol in PERCENTAGE  e.g. 1.407  (not 0.01407)
    omega       : GARCH ω in %² units  (fitted on % returns)
    alpha/gamma/beta : dimensionless

    Key insight — NO sqrt(dt) inside loop
    --------------------------------------
    sigma0_pct is already a DAILY vol.
    Each loop iteration = exactly 1 day.
    Scaling by sqrt(dt) = sqrt(1/252) would make vol 16× too small.
    drift uses r/252 (daily risk-free), diffusion uses sigma_daily directly.
    """
    np.random.seed(seed)
    n_days  = max(1, round(T * 252))
    M_half  = M // 2

    # Antithetic variates: draw M/2, mirror for second half
    eps_raw = np.random.standard_normal((M_half, n_days))
    eps     = np.concatenate([eps_raw, -eps_raw], axis=0)   # (M_total, n_days)
    M_total = M_half * 2

    # ── GARCH-MC paths ────────────────────────────────────────────────────────
    S_t   = np.full(M_total, float(S))
    var_t = np.full(M_total, sigma0_pct**2)     # variance in %² units

    for t in range(n_days):
        eps_t     = eps[:, t]
        sigma_dec = np.sqrt(var_t) / 100.0      # daily vol → decimal

        # Daily log return: drift = r/252, diffusion = sigma_daily × ε
        ret   = (r / 252 - 0.5 * sigma_dec**2) + sigma_dec * eps_t
        S_t   = S_t * np.exp(ret)

        # GJR-GARCH variance update in %² space
        shock = eps_t * np.sqrt(var_t)           # shock in % units
        ind   = (eps_t < 0).astype(float)        # leverage indicator
        var_t = (omega
                 + (alpha + gamma * ind) * shock**2
                 + beta * var_t)
        var_t = np.clip(var_t, 1e-6, 100.0)     # floor/ceil in %² units

    # Payoffs
    if option_type == "call":
        payoffs       = np.maximum(S_t - K, 0)
        bs_analytical = bs_call(S, K, T, r, sigma0_pct / 100.0 * np.sqrt(252))
    else:
        payoffs       = np.maximum(K - S_t, 0)
        bs_analytical = bs_put(S, K, T, r, sigma0_pct / 100.0 * np.sqrt(252))

    discount     = np.exp(-r * T)
    mc_price_raw = discount * payoffs.mean()

    # ── Control Variate: flat-vol MC with same shocks ─────────────────────────
    # Measures pure sampling noise → subtract from GARCH MC
    S_flat         = np.full(M_total, float(S))
    sigma_dec_flat = sigma0_pct / 100.0          # constant daily decimal vol

    for t in range(n_days):
        eps_t    = eps[:, t]
        ret_flat = (r / 252 - 0.5 * sigma_dec_flat**2) + sigma_dec_flat * eps_t
        S_flat   = S_flat * np.exp(ret_flat)

    if option_type == "call":
        payoffs_flat = np.maximum(S_flat - K, 0)
    else:
        payoffs_flat = np.maximum(K - S_flat, 0)

    mc_bs_price = discount * payoffs_flat.mean()
    noise       = mc_bs_price - bs_analytical    # sampling error estimate
    mc_price_cv = mc_price_raw - noise           # corrected GARCH MC price

    se = discount * payoffs.std() / np.sqrt(M_total)

    return {
        "mc_raw"  : mc_price_raw,
        "mc_cv"   : mc_price_cv,
        "bs_price": bs_analytical,
        "noise"   : noise,
        "se"      : se,
        "paths"   : M_total,
    }

# ── 4. LOAD DATA ──────────────────────────────────────────────────────────────
params_path = os.path.join(BASE_DIR, "data", "vajra_garch_params.csv")
sigma_path  = os.path.join(BASE_DIR, "data", "vajra_sigma.pkl")

garch_params = pd.read_csv(params_path, index_col=0)

with open(sigma_path, "rb") as f:
    sigma_df = pickle.load(f)

# Keep sigma in % units as stored (e.g. 1.407 not 0.01407)
last_sigma_pct = sigma_df.iloc[-1]

logger.info("Loaded GARCH parameters:")
logger.info(garch_params[["omega", "alpha", "gamma", "beta"]].round(6))
logger.info("\nLast GARCH daily sigma (% units):")
for s, v in last_sigma_pct.items():
    logger.info(f"  {s:<14} {v:.4f}%/day  →  {v * np.sqrt(252):.2f}% annualised")

# ── 5. MARKET SETUP ───────────────────────────────────────────────────────────
spot_prices  = {"RELIANCE": 1255.0, "HDFCBANK": 1770.0, "INFY": 1930.0}
R_FREE       = 0.065
EXPIRIES     = {"30d": 30/252, "60d": 60/252, "90d": 90/252}
STRIKE_MULT  = {"ATM": 1.00, "OTM_5pct": 1.05, "OTM_10pct": 1.10}
M_PATHS      = 10000

# ── 6. MAIN LOOP ──────────────────────────────────────────────────────────────
logger.info("\n" + "="*70)
logger.info("GARCH MONTE CARLO vs BLACK-SCHOLES — CALL OPTIONS")
logger.info("="*70)

records = []

for stock in spot_prices:
    S      = spot_prices[stock]
    s0_pct = last_sigma_pct[stock]
    p      = garch_params.loc[stock]
    omega  = float(p["omega"])
    alpha  = float(p["alpha"])
    gamma  = float(p["gamma"])
    beta   = float(p["beta"])

    logger.info(f"\n{'─'*60}")
    logger.info(f"  {stock}  Spot=₹{S:.0f}  "
          f"σ_daily={s0_pct:.3f}%  "
          f"σ_annual={s0_pct * np.sqrt(252):.2f}%")
    logger.info(f"  GARCH: ω={omega:.2e}  α={alpha:.4f}  γ={gamma:.4f}  β={beta:.4f}")
    logger.info(f"{'─'*60}")
    logger.info(f"  {'Expiry':<8} {'Strike':<12} {'BS₹':>8} {'MC_raw₹':>9} "
          f"{'MC_cv₹':>9} {'Diff₹':>8} {'Diff%':>7} {'SE₹':>7}")
    logger.info(f"  {'─'*68}")

    for exp_label, T in EXPIRIES.items():
        for sk_label, mult in STRIKE_MULT.items():
            K   = S * mult
            res = garch_mc_price(
                S, K, T, R_FREE,
                s0_pct, omega, alpha, gamma, beta,
                M=M_PATHS, option_type="call"
            )

            diff     = res["mc_cv"] - res["bs_price"]
            diff_pct = (diff / res["bs_price"] * 100
                        if res["bs_price"] > 0.01 else 0.0)

            iv_mc = implied_vol(
                max(res["mc_cv"], 0.01), S, K, T, R_FREE, "call"
            )

            records.append({
                "stock"      : stock,
                "expiry"     : exp_label,
                "strike_type": sk_label,
                "spot"       : S,
                "strike"     : K,
                "T"          : T,
                "bs_price"   : res["bs_price"],
                "mc_raw"     : res["mc_raw"],
                "mc_cv"      : res["mc_cv"],
                "diff_rupee" : diff,
                "diff_pct"   : diff_pct,
                "se"         : res["se"],
                "iv_bs"      : s0_pct / 100.0 * np.sqrt(252) * 100,
                "iv_mc"      : (iv_mc * 100
                                if iv_mc and not np.isnan(iv_mc)
                                else np.nan),
            })

            logger.info(f"  {exp_label:<8} {sk_label:<12} "
                  f"₹{res['bs_price']:>6.1f} "
                  f"₹{res['mc_raw']:>7.1f} "
                  f"₹{res['mc_cv']:>7.1f} "
                  f"₹{diff:>+6.1f} "
                  f"{diff_pct:>+6.1f}% "
                  f"₹{res['se']:>5.2f}")

results_df = pd.DataFrame(records)

# ── 7. VOLATILITY SMILE TABLE ─────────────────────────────────────────────────
logger.info("\n" + "="*70)
logger.info("IMPLIED VOLATILITY SMILE — MC vs BS (30-day expiry)")
logger.info("="*70)
logger.info(f"{'Stock':<12} {'Strike':<12} {'IV_BS%':>8} {'IV_MC%':>8} {'ΔIV%':>8}")
logger.info("─"*52)

smile_df = results_df[results_df["expiry"] == "30d"]
for _, r in smile_df.iterrows():
    delta_iv = (r["iv_mc"] - r["iv_bs"]
                if not np.isnan(r["iv_mc"]) else np.nan)
    logger.info(f"{r['stock']:<12} {r['strike_type']:<12} "
          f"{r['iv_bs']:>7.2f}% "
          f"{r['iv_mc']:>7.2f}%  "
          f"{delta_iv:>+6.2f}%")

# ── 8. CONVERGENCE STUDY — RELIANCE ATM 30d ───────────────────────────────────
logger.info("\nConvergence Study — RELIANCE ATM 30d Call")
logger.info(f"{'M paths':<10} {'BS₹':>8} {'MC price₹':>11} {'SE₹':>8}")
logger.info("─"*42)

conv_records = []
S0   = spot_prices["RELIANCE"]
s0p  = last_sigma_pct["RELIANCE"]
p_r  = garch_params.loc["RELIANCE"]

for M in [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
    res = garch_mc_price(
        S0, S0, 30/252, R_FREE,
        s0p,
        float(p_r["omega"]), float(p_r["alpha"]),
        float(p_r["gamma"]), float(p_r["beta"]),
        M=M, option_type="call", seed=42
    )
    conv_records.append({
        "M"    : M,
        "price": res["mc_cv"],
        "se"   : res["se"],
        "bs"   : res["bs_price"]
    })
    logger.info(f"{M:<10} ₹{res['bs_price']:>6.1f} "
          f"₹{res['mc_cv']:>9.2f}  "
          f"₹{res['se']:>6.3f}")

conv_df = pd.DataFrame(conv_records)

# ── 9. SAVE CSV ───────────────────────────────────────────────────────────────
results_df.to_csv(
    os.path.join(BASE_DIR, "data", "mc_results.csv"), index=False
)
logger.info(f"\nSaved → data/mc_results.csv")

# ── 10. PLOTS ─────────────────────────────────────────────────────────────────
COLORS        = {"RELIANCE": "#00d4aa", "HDFCBANK": "#ffd93d", "INFY": "#ff6b6b"}
DARK          = "#0f0f0f"
STRIKE_ORDER  = ["ATM", "OTM_5pct", "OTM_10pct"]
STRIKE_LABELS = ["ATM", "5% OTM", "10% OTM"]
EXPIRY_ORDER  = ["30d", "60d", "90d"]

fig = plt.figure(figsize=(20, 16), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

# Panel 1 — Volatility Smirk
ax1 = fig.add_subplot(gs[0, :2])
x_pos = [0, 1, 2]
for stock in spot_prices:
    sub     = smile_df[smile_df["stock"] == stock]
    iv_bs_v = [sub[sub["strike_type"]==s]["iv_bs"].values[0] for s in STRIKE_ORDER]
    iv_mc_v = [sub[sub["strike_type"]==s]["iv_mc"].values[0] for s in STRIKE_ORDER]
    ax1.plot(x_pos, iv_bs_v, "--",
             color=COLORS[stock], alpha=0.4, linewidth=1.2,
             label=f"{stock} BS (flat)")
    ax1.plot(x_pos, iv_mc_v, "-o",
             color=COLORS[stock], linewidth=2, markersize=7,
             label=f"{stock} GARCH MC")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(STRIKE_LABELS, color="white")
ax1.set_facecolor(DARK)
ax1.set_title(
    "Volatility Smirk — GARCH MC vs BS Flat Vol\n"
    "Dashed = BS (flat) · Solid = GARCH MC · Smile emerges OTM",
    color="white", fontsize=10)
ax1.set_ylabel("Implied Volatility %", color="white", fontsize=9)
ax1.set_xlabel("Strike", color="white", fontsize=9)
ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=7, ncol=3)
ax1.tick_params(colors="white", labelsize=8)
ax1.spines[["top","right","left","bottom"]].set_color("#333333")
ax1.grid(alpha=0.15, color="white")

# Panel 2 — % Premium
ax2 = fig.add_subplot(gs[0, 2])
x = np.arange(3)
w = 0.28
for i, stock in enumerate(spot_prices):
    sub  = smile_df[smile_df["stock"] == stock]
    vals = [sub[sub["strike_type"]==s]["diff_pct"].values[0] for s in STRIKE_ORDER]
    ax2.bar(x + i*w, vals, w, label=stock, color=COLORS[stock], alpha=0.85)
ax2.axhline(0, color="white", linewidth=0.5, alpha=0.5)
ax2.set_facecolor(DARK)
ax2.set_title("GARCH MC Premium Over BS\n(% difference, 30-day)",
              color="white", fontsize=10)
ax2.set_ylabel("% Premium", color="white", fontsize=9)
ax2.set_xticks(x + w)
ax2.set_xticklabels(STRIKE_LABELS, color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=8)
ax2.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=7)
ax2.spines[["top","right","left","bottom"]].set_color("#333333")
ax2.grid(alpha=0.15, color="white", axis="y")

# Panel 3 — Convergence
ax3 = fig.add_subplot(gs[1, :2])
ax3.semilogx(conv_df["M"], conv_df["price"], "-o",
             color="#00d4aa", linewidth=2, markersize=6,
             label="GARCH MC (control variate)")
ax3.axhline(conv_df["bs"].iloc[0], color="#ffd93d",
            linewidth=1.5, linestyle="--", label="BS Analytical")
ax3.fill_between(
    conv_df["M"],
    conv_df["price"] - 2*conv_df["se"],
    conv_df["price"] + 2*conv_df["se"],
    alpha=0.2, color="#00d4aa", label="±2 SE band"
)
ax3.set_facecolor(DARK)
ax3.set_title(
    "Monte Carlo Convergence — RELIANCE ATM 30d Call\n"
    "Price stabilises ~M=5,000 · SE shrinks as 1/√M",
    color="white", fontsize=10)
ax3.set_xlabel("Number of Paths (log scale)", color="white", fontsize=9)
ax3.set_ylabel("Option Price ₹", color="white", fontsize=9)
ax3.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
ax3.tick_params(colors="white", labelsize=8)
ax3.spines[["top","right","left","bottom"]].set_color("#333333")
ax3.grid(alpha=0.15, color="white")

# Panel 4 — SE decay
ax4 = fig.add_subplot(gs[1, 2])
ax4.loglog(conv_df["M"], conv_df["se"], "-o",
           color="#ff6b6b", linewidth=2, markersize=6,
           label="Observed SE")
m_theory  = np.array(conv_df["M"])
se_theory = conv_df["se"].iloc[0] * np.sqrt(conv_df["M"].iloc[0] / m_theory)
ax4.loglog(m_theory, se_theory, "--",
           color="white", alpha=0.4, linewidth=1, label="1/√M theory")
ax4.set_facecolor(DARK)
ax4.set_title("Standard Error Decay\n1/√M confirmed",
              color="white", fontsize=10)
ax4.set_xlabel("M paths (log)", color="white", fontsize=9)
ax4.set_ylabel("Standard Error ₹ (log)", color="white", fontsize=9)
ax4.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
ax4.tick_params(colors="white", labelsize=8)
ax4.spines[["top","right","left","bottom"]].set_color("#333333")
ax4.grid(alpha=0.15, color="white")

# Panel 5 — BS vs MC across expiries, RELIANCE
ax5 = fig.add_subplot(gs[2, :2])
rel_df  = results_df[results_df["stock"] == "RELIANCE"]
x2      = np.arange(len(EXPIRY_ORDER))
w2      = 0.25
sk_cols = {"ATM": "#00d4aa", "OTM_5pct": "#ffd93d", "OTM_10pct": "#ff6b6b"}
for i, sk in enumerate(STRIKE_ORDER):
    sub   = rel_df[rel_df["strike_type"] == sk]
    bs_v  = [sub[sub["expiry"]==e]["bs_price"].values[0] for e in EXPIRY_ORDER]
    mc_v  = [sub[sub["expiry"]==e]["mc_cv"].values[0]    for e in EXPIRY_ORDER]
    offset = x2 + (i - 1) * w2
    ax5.bar(offset, bs_v, w2*0.9, color=sk_cols[sk], alpha=0.35,
            label=f"BS {STRIKE_LABELS[i]}")
    ax5.bar(offset, mc_v, w2*0.9, color=sk_cols[sk], alpha=0.9,
            label=f"MC {STRIKE_LABELS[i]}")
ax5.set_facecolor(DARK)
ax5.set_title(
    "RELIANCE — BS vs GARCH MC Call Prices Across Expiries\n"
    "Faded = BS · Solid = GARCH MC · Gap grows OTM",
    color="white", fontsize=10)
ax5.set_xticks(x2)
ax5.set_xticklabels(EXPIRY_ORDER, color="white")
ax5.set_ylabel("Option Price ₹", color="white", fontsize=9)
ax5.tick_params(colors="white", labelsize=8)
ax5.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=7, ncol=3)
ax5.spines[["top","right","left","bottom"]].set_color("#333333")
ax5.grid(alpha=0.15, color="white", axis="y")

# Panel 6 — Payoff distribution RELIANCE ATM 30d
ax6 = fig.add_subplot(gs[2, 2])
np.random.seed(42)
s0p_rel = last_sigma_pct["RELIANCE"]
p_rel   = garch_params.loc["RELIANCE"]
M_hist  = 5000
eps_h   = np.random.standard_normal((M_hist, 30))
eps_h   = np.concatenate([eps_h, -eps_h], axis=0)
S_h     = np.full(M_hist * 2, float(S0))
var_h   = np.full(M_hist * 2, s0p_rel**2)

for t in range(30):
    ep        = eps_h[:, t]
    sig_dec   = np.sqrt(var_h) / 100.0
    ret       = (R_FREE / 252 - 0.5 * sig_dec**2) + sig_dec * ep
    S_h       = S_h * np.exp(ret)
    shock_pct = ep * np.sqrt(var_h)
    ind       = (ep < 0).astype(float)
    var_h     = (float(p_rel["omega"])
                 + (float(p_rel["alpha"]) + float(p_rel["gamma"]) * ind) * shock_pct**2
                 + float(p_rel["beta"]) * var_h)
    var_h     = np.clip(var_h, 1e-6, 100.0)

payoffs_h    = np.maximum(S_h - S0, 0)
nonzero_mask = payoffs_h > 0
pct_itm      = nonzero_mask.sum() / len(payoffs_h) * 100

ax6.hist(payoffs_h[nonzero_mask], bins=60,
         color="#a29bfe", alpha=0.8, edgecolor="none")
ax6.axvline(payoffs_h.mean(), color="#ffd93d",
            linewidth=2, linestyle="--",
            label=f"Mean=₹{payoffs_h.mean():.1f}")
ax6.set_facecolor(DARK)
ax6.set_title(
    f"Payoff Distribution — RELIANCE ATM 30d\n"
    f"{pct_itm:.1f}% of paths expire ITM",
    color="white", fontsize=10)
ax6.set_xlabel("Payoff ₹", color="white", fontsize=9)
ax6.set_ylabel("Frequency", color="white", fontsize=9)
ax6.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
ax6.tick_params(colors="white", labelsize=8)
ax6.spines[["top","right","left","bottom"]].set_color("#333333")
ax6.grid(alpha=0.15, color="white")

fig.suptitle(
    "GARCH Risk Engine — Monte Carlo Options Pricing\n"
    "GARCH-MC vs Black-Scholes  |  Volatility Smirk  |  "
    "Convergence  |  RELIANCE · HDFCBANK · INFY",
    color="white", fontsize=12
)

os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(
    os.path.join(BASE_DIR, "assets", "mc_options.png"),
    dpi=150, bbox_inches="tight", facecolor=DARK
)
plt.close()
logger.info("Saved → assets/mc_options.png")
