import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger(__name__)

# ── 1. Fetch Nifty 50 data ─────────────────────────────────────
ticker = "^NSEI"
data = yf.download(ticker, start="2018-01-01", end="2026-04-29", auto_adjust=True)
prices = data['Close'].dropna().squeeze()
returns = np.log(prices / prices.shift(1)).dropna() * 100  # % scale

logger.info(f"Rows: {len(returns)} | Mean: {returns.mean():.4f}% | Std: {returns.std():.4f}%")

# ── 2. Fit GARCH(1,1) ──────────────────────────────────────────
model  = arch_model(returns, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
result = model.fit(disp='off')
logger.info(result.summary())

# ── 3. Extract parameters ─────────────────────────────────────
omega = result.params['omega']
alpha = result.params['alpha[1]']
beta  = result.params['beta[1]']

lrv       = omega / (1 - alpha - beta)
lr_vol    = np.sqrt(lrv)
half_life = np.log(0.5) / np.log(alpha + beta)

logger.info("\n─── GARCH(1,1) Parameters ───────────────────────────")
logger.info(f"  ω  (omega)        : {omega:.6f}")
logger.info(f"  α  (alpha)        : {alpha:.4f}")
logger.info(f"  β  (beta)         : {beta:.4f}")
logger.info(f"  α + β (persist.)  : {alpha+beta:.4f}")
logger.info(f"  Long-run vol/day  : {lr_vol:.4f}%")
logger.info(f"  Long-run vol/yr   : {lr_vol * np.sqrt(252):.2f}%")
logger.info(f"  Shock half-life   : {half_life:.1f} trading days")

# ── 4. Next-day forecast ──────────────────────────────────────
forecast  = result.forecast(horizon=1)
next_vol  = np.sqrt(forecast.variance.iloc[-1, 0])
logger.info(f"\n  Next-day σ (daily)    : {next_vol:.4f}%")
logger.info(f"  Next-day σ (annual)   : {next_vol * np.sqrt(252):.2f}%")

# ── 5. Volatility Clustering Plot ─────────────────────────────
cond_vol = result.conditional_volatility
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('GARCH(1,1) — Nifty 50 Volatility Clustering', fontweight='bold')

axes[0].plot(returns.index, returns.values, color='#334155', lw=0.5, alpha=0.8)
axes[0].set_ylabel('Return (%)'); axes[0].set_title('Daily Log Returns')

axes[1].plot(cond_vol.index, cond_vol.values, color='#e84545', lw=0.9)
axes[1].set_ylabel('σ (%)'); axes[1].set_title('GARCH Conditional Volatility')

axes[2].fill_between(returns.index, np.abs(returns.values), alpha=0.2, color='steelblue', label='|Returns|')
axes[2].plot(cond_vol.index, cond_vol.values, color='red', lw=0.9, label='GARCH σ')
axes[2].set_ylabel('%'); axes[2].set_title('Clustering Proof: |Returns| vs GARCH σ')
axes[2].legend()

# Mark key NSE events
for label, date, ax_list in [
    ('COVID\nMar20', '2020-03-23', axes),
    ('Adani\nJan23', '2023-01-27', axes),
    ('Election\nJun24', '2024-06-04', axes),
]:
    for ax in ax_list:
        ax.axvline(pd.Timestamp(date), color='orange', lw=1, ls='--', alpha=0.7)

plt.tight_layout()
plt.savefig('assets/garch_nifty50.png', dpi=150, bbox_inches='tight')
logger.info("\n✅ Saved → assets/garch_nifty50.png")
plt.show()

# ============================================================
# GARCH RISK ENGINE — MULTI-STOCK GARCH ENGINE
# ============================================================
# What this block does:
# 1. Loads clean returns from data pipeline
# 2. Fits GARCH(1,1) with Student-t per stock
# 3. Extracts: conditional volatility σ_{i,t} for every stock every day
# 4. Computes standardised residuals z_{i,t} = ε_{i,t} / σ_{i,t}
# 5. Validates each fit (Ljung-Box + ARCH LM tests)
# 6. Saves σ matrix and z matrix for DCC engine

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD RETURNS FROM DATA PIPELINE
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
returns_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_returns.csv"),
    index_col=0,
    parse_dates=True
)

# Scale to percentage — GARCH numerics are better behaved at this scale
# ε_{i,t} will be in % units, σ_{i,t} will be in % units
# z_{i,t} = ε/σ is dimensionless — same regardless of scale
returns_pct = returns_df * 100

logger.info(f"Returns loaded: {returns_pct.shape}")
logger.info(f"Stocks: {list(returns_pct.columns)}")

# ============================================================
# STEP 2: FIT GARCH(1,1) PER STOCK
# ============================================================
# Storage containers
sigma_dict = {}       # σ_{i,t}: conditional vol per stock per day
z_dict = {}           # z_{i,t}: standardised residuals per stock per day
params_dict = {}      # GARCH parameters per stock
validation_dict = {}  # diagnostic test results per stock

logger.info("\n--- FITTING GARCH(1,1) PER STOCK ---")
logger.info(f"{'Stock':<15} {'ω':>10} {'α':>8} {'β':>8} {'α+β':>8} {'Half-life':>12} {'Status'}")
logger.info("-" * 75)



# Stocks that need GJR-GARCH based on validation failures
GJR_STOCKS = {'AXISBANK', 'ONGC'}
# AXISBANK: ARCH LM p=0.001 — clear asymmetric vol response
# ONGC: ARCH LM p=0.046 — borderline, energy sector news asymmetry
# ── SPECIAL MODEL SPECS — ARMA mean fix ───────────────────────
# These three failed Ljung-Box / ARCH LM diagnostics
# Root cause: serial correlation in returns mean, not variance
# Fix: ARMA(1,0) mean removes the autocorrelation before GARCH sees it
ARMA_STOCKS = {
    'AXISBANK': {
        'vol'  : 'GARCH',
        'p'    : 2,        # GARCH(2,1) — two variance lags
        'o'    : 1,        # GJR leverage term
        'q'    : 1,
        'mean' : 'ARX',
        'lags' : 5,        # AR(5) mean — already confirmed fixing LB
        'dist' : 't'
    },
    'ICICIBANK': {
        'vol'  : 'EGARCH',  # keep — it worked
        'p'    : 1,
        'o'    : 0,
        'q'    : 1,
        'mean' : 'ARX',
        'lags' : 5,
        'dist' : 't'
    },
    'TCS': {
        'vol'  : 'GARCH',
        'p'    : 2,
        'o'    : 0,
        'q'    : 1,
        'mean' : 'Constant',
        'lags' : 0,
        'dist' : 't'
    },
}

# AXISBANK also stays in GJR_STOCKS — ARMA_STOCKS overrides everything
# Remove AXISBANK from GJR_STOCKS since ARMA_STOCKS handles it fully
GJR_STOCKS = {'ONGC'}   # ← AXISBANK moved to ARMA_STOCKS

logger.info("\n--- FITTING GARCH(1,1) PER STOCK ---")
logger.info(f"{'Stock':<15} {'ω':>10} {'α':>8} {'β':>8} {'α+β':>8} {'Half-life':>12} {'Model':<12} {'Status'}")
logger.info("-" * 90)

for stock in returns_pct.columns:
    series = returns_pct[stock].dropna()

    # Choose model based on stock
        # ── pick spec ──────────────────────────────────────────────
    if stock in ARMA_STOCKS:
        spec = ARMA_STOCKS[stock]
        ar_order   = spec['lags']
        vol_type   = spec['vol']
        model_name = f"ARMA({ar_order},0)-{vol_type}({spec['p']},1)"

        if vol_type == 'EGARCH':
            # EGARCH — arch library uses EWMAVariance internally
            # o parameter not used — asymmetry via 'o' in EGARCH works differently
            model = arch_model(
                series,
                vol='EGARCH',
                p=spec['p'],
                q=spec['q'],
                mean=spec['mean'],
                lags=spec['lags'],
                dist=spec['dist']
            )
        else:
            model = arch_model(
                series,
                vol=spec['vol'],
                p=spec['p'],
                o=spec['o'],
                q=spec['q'],
                mean=spec['mean'],
                lags=spec['lags'],
                dist=spec['dist']
            )

    elif stock in GJR_STOCKS:
        model_name = "GJR-GARCH(1,1)"
        model = arch_model(
            series,
            vol='GARCH',
            p=1, o=1, q=1,
            mean='Constant',
            dist='t'
        )

    else:
        model_name = "GARCH(1,1)"
        model = arch_model(
            series,
            vol='GARCH',
            p=1, o=0, q=1,
            mean='Constant',
            dist='t'
        )

    try:
        result = model.fit(disp='off', options={'maxiter': 500})
        logger.info(f"\nAXISBANK EGARCH check:")
        logger.info(f"  cond_vol range: {result.conditional_volatility.min():.4f} to {result.conditional_volatility.max():.4f}")
        logger.info(f"  resid range:    {result.resid.min():.4f} to {result.resid.max():.4f}")
        logger.info(f"  z range:        {result.std_resid.min():.4f} to {result.std_resid.max():.4f}")
        logger.info(f"  z² mean:        {(result.std_resid**2).mean():.4f}  ← should be ~1.0")
        logger.info(result.params)

        cond_vol = result.conditional_volatility
        resid = result.resid
        z_scores = resid / cond_vol

        sigma_dict[stock] = cond_vol
        z_dict[stock] = z_scores

        omega = result.params['omega']
        alpha = result.params['alpha[1]']
        beta  = result.params['beta[1]']
        gamma = result.params.get('gamma[1]', 0.0)
        ar1   = result.params.get('Lag 1',    0.0)  # ARMA AR(1) coeff
        # ar1 = 0.0 for all non-ARMA stocks — harmless
        # For standard GARCH gamma = 0 (parameter doesn't exist)
        # For GJR gamma > 0 means bad news amplifies volatility more than good news
        # EGARCH persistence is different — α+β still works as approximation
        # but flag it
                # persistence calculation — works for all model types
        is_egarch = (stock in ARMA_STOCKS and ARMA_STOCKS[stock]['vol'] == 'EGARCH')

        if is_egarch:
            persistence = float(np.abs(beta))
        else:
            persistence = alpha + beta + 0.5 * gamma
            # gamma=0 for standard GARCH — harmless
            # gamma>0 for GJR — adds γ/2 correctly

        # GJR persistence formula: α + β + γ/2
        # γ/2 because leverage only fires on negative shocks (~50% of days)

        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

        params_dict[stock] = {
            'omega': omega,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'ar1': ar1, 
            'persistence': persistence,
            'half_life': half_life,
            'nu': result.params.get('nu', np.nan),
            'model': model_name
        }

        status = "✓"
        logger.info(f"{stock:<15} {omega:>10.5f} {alpha:>8.4f} {beta:>8.4f} "
              f"{persistence:>8.4f} {half_life:>10.1f}d  {model_name:<12} {status}")

    except Exception as e:
        logger.info(f"{stock:<15} {'FAILED':>10} — {str(e)[:40]}")

# ── DRREDDY SANITY CHECK ──────────────────────────────────────
logger.info("\n--- DRREDDY ANOMALY CHECK ---")
logger.info(returns_pct['DRREDDY'].describe().round(4))
logger.info("\nWorst 5 days:")
logger.info(returns_pct['DRREDDY'].sort_values().head(5).round(4))
logger.info("\nBest 5 days:")
logger.info(returns_pct['DRREDDY'].sort_values().tail(5).round(4))
logger.info(f"\nSkewness: {returns_pct['DRREDDY'].skew():.4f}")
logger.info(f"Kurtosis: {returns_pct['DRREDDY'].kurtosis():.4f}")

# ============================================================
# STEP 3: BUILD ALIGNED MATRICES
# ============================================================
# All sigma and z series must share the SAME date index for DCC
# Different stocks have slightly different start dates (IPO, data gaps)
# Solution: inner join — keep only dates where ALL stocks have valid data

sigma_df = pd.DataFrame(sigma_dict)
z_df = pd.DataFrame(z_dict)

# Align on common dates
common_idx = sigma_df.dropna().index
sigma_df = sigma_df.loc[common_idx]
z_df = z_df.loc[common_idx]

logger.info(f"\nAligned matrices shape: {sigma_df.shape}")
logger.info(f"Date range: {sigma_df.index[0].date()} → {sigma_df.index[-1].date()}")
logger.info(f"Stocks with valid GARCH: {list(sigma_df.columns)}")



# ============================================================
# STEP 4: VALIDATE GARCH FITS
# ============================================================
# Good GARCH fit means:
# 1. No autocorrelation left in z scores (Ljung-Box p > 0.05)
#    → GARCH has captured all serial dependence in volatility
# 2. No remaining ARCH effects in z scores (ARCH LM p > 0.05)
#    → No more volatility clustering in the residuals
# If tests fail: GARCH hasn't fully cleaned the series
# DCC on unclean z-scores gives unreliable correlation estimates

logger.info("\n--- GARCH VALIDATION TESTS ---")
logger.info(f"{'Stock':<15} {'LjungBox_p':>12} {'ARCHLM_p':>12} {'Verdict'}")
logger.info("-" * 55)

for stock in z_df.columns:
    z = z_df[stock].dropna()

    # Ljung-Box test on z scores (lag=10)
    lb_result = acorr_ljungbox(z, lags=[10], return_df=True)
    lb_p = lb_result['lb_pvalue'].values[0]

    lb_result = acorr_ljungbox(z, lags=[5, 10, 20], return_df=True)
    lb_p = lb_result['lb_pvalue'].min()   # worst lag — strictest check

    # p > 0.05 → no autocorrelation in z → GARCH removed serial dependence ✓

    # ARCH LM test on z scores
    _, arch_p, _, _ = het_arch(z, nlags=5)
    _, arch_p, _, _ = het_arch(z, nlags=20)
    # p > 0.05 → no remaining ARCH effects → volatility clustering removed ✓

    lb_pass = lb_p > 0.05
    arch_pass = arch_p > 0.05
    verdict = "✓ CLEAN" if (lb_pass and arch_pass) else "⚠ CHECK"

    validation_dict[stock] = {
        'ljungbox_p': lb_p,
        'arch_lm_p': arch_p,
        'lb_pass': lb_pass,
        'arch_pass': arch_pass
    }

    logger.info(f"{stock:<15} {lb_p:>12.4f} {arch_p:>12.4f}   {verdict}")

# ============================================================
# STEP 5: ANNUALISED VOL SUMMARY
# ============================================================
logger.info("\n--- ANNUALISED VOLATILITY SUMMARY ---")
avg_daily_vol = sigma_df.mean()
annual_vol = avg_daily_vol * np.sqrt(252)
vol_summary = pd.DataFrame({
    'Avg Daily Vol (%)': avg_daily_vol.round(3),
    'Annual Vol (%)': annual_vol.round(2)
}).sort_values('Annual Vol (%)', ascending=False)
logger.info(vol_summary)
# BAJFINANCE, TATAMOTORS → expect high (35-45% annual)
# HINDUNILVR, ITC → expect low (18-25% annual)
# If anything shows > 80% annual → data problem

# ============================================================
# STEP 6: SAVE FOR DCC ENGINE
# ============================================================
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

with open(os.path.join(BASE_DIR, "data", "vajra_sigma.pkl"), "wb") as f:
    pickle.dump(sigma_df, f)

with open(os.path.join(BASE_DIR, "data", "vajra_z.pkl"), "wb") as f:
    pickle.dump(z_df, f)
params_df = pd.DataFrame(params_dict).T
params_df.to_csv(os.path.join(BASE_DIR, "data", "vajra_garch_params.csv"))

logger.info("\n✅ Saved:")
logger.info("   data/vajra_sigma.pkl         → σ_{i,t} matrix (T × N)")
logger.info("   data/vajra_z.pkl             → z_{i,t} matrix (T × N)")
logger.info("   data/vajra_garch_params.csv  → GARCH parameters per stock")
