# Code Audit Log — Quant Pipeline
**Date:** 2026-06-09  
**Projects:** Indian Risk Engine · Alpha-Core · Portfolio Optimizer · Live Trading  
**Status:** All 13 issues fixed ✅

---

## HOW TO USE THIS FILE
Each entry has:
- **What the bug was** (what actually happened in the code)
- **Why it matters** (how it breaks the math or the output)
- **The fix** (what was changed)
- **Interview answer** (how to explain it if asked)

---

## CRITICAL FIXES (Interview Killers)

---

### Bug #12 — HMM Regime miscalibrated: sorts by Volatility instead of Direction
**Severity:** CRITICAL  
**File fixed:** `alpha-core/alpha_core/hmm_regime.py`

**Original code:**
```python
features["realised_vol"] = mkt.rolling(20).std() * np.sqrt(252)
features["momentum_20d"] = mkt.rolling(20).sum()
# ... inside label_states()
sharpe_scores = np.array([means_orig[k, 0] / vols_orig[k] ...])
sorted_states = np.argsort(sharpe_scores)
```

**Fixed code:**
```python
features["india_vix"] = vix_aligned / 100.0  # Exogenous forward-looking fear
features["momentum_sharpe"] = roll_mean / (roll_std + 1e-9) # Directional momentum
# ... inside label_states()
bull_state = np.argmax(mom_means)
bear_state = remaining_states[np.argmax([vix_means[i] for i in remaining_states])]
```

**Why it matters:**
Realized volatility lags (stays elevated 20 days post-crash), causing "volatility clustering" instead of directional regimes. The Sharpe-sort in `label_states` was mathematically broken because all regimes had near-zero mean returns in the Indian market; the denominator (volatility) dominated the sort, causing the COVID crash (highest volatility) to accidentally score the highest Sharpe and be mislabelled as a "Bull" market. Using India VIX (exogenous implied vol) and a directional rolling Sharpe fixes the features, and mapping by highest momentum/highest VIX makes the labels semantically pure.

**Interview answer:**  
*"The HMM was clustering by volatility rather than market direction. The COVID crash was misclassified because rolling realized vol stays elevated post-crash, and a Sharpe-based state sorter failed when all state means were near zero. I fixed it by injecting India VIX as an exogenous fear signal, standardizing momentum to a rolling Sharpe, and mapping the states semantically (Highest Momentum = Bull, Highest VIX = Bear)."*

---

### Bug #13 — Alpaca Delta execution bug: per-ticker loops against aggregate ETF holdings
**Severity:** CRITICAL  
**File fixed:** `alpha-core/alpha_core/alpaca_gate.py`

**Original code:**
```python
for _, row in order_book.iterrows():
    symbol = row["etf_proxy"] # e.g. XLF
    target_pct = row["target_pct"]
    current_qty = current_positions.get(symbol, 0)
    delta = target_qty - current_qty
    api.submit_order(symbol, qty=delta, client_order_id=f"alphacore_{nse}_{today}")
```

**Fixed code:**
```python
etf_targets = {}
for _, row in order_book.iterrows():
    etf_targets[row["etf_proxy"]] += row["target_pct"]

for symbol, aggregated_pct in etf_targets.items():
    current_qty = current_positions.get(symbol, 0)
    delta = target_qty - current_qty
    api.submit_order(symbol, qty=delta, client_order_id=f"alphacore_{symbol}_{today}")
```

**Why it matters:**
The script generates signals for 14 Indian stocks, but maps them to 6 US ETF proxies (e.g. HDFCBANK, ICICIBANK both map to XLF). The old loop iterated over individual Indian stocks, calculated a `target_qty` for that stock's %, but subtracted the `current_qty` of the entire ETF holding without updating the holding mid-loop. This caused the bot to submit multiple overlapping orders for the same ETF, resulting in massive over-leveraging and API rejections (`client_order_id must be unique`). The fix aggregates the Kelly targets at the ETF level before computing a single delta per ETF.

**Interview answer:**  
*"The Alpaca paper execution module suffered from a many-to-one mapping flaw. Multiple Indian stock signals were mapping to the same US ETF proxy. The execution loop evaluated orders per-stock against the aggregate ETF holding, causing the bot to buy the same ETF allocation multiple times over and fail on duplicate order IDs. I rewrote the execution engine to aggregate all targets at the ETF level first and submit a single unified delta order per ETF."*

---

### Bug #1 — IC is Pearson, should be Spearman
**Severity:** CRITICAL  
**Files fixed:**
- `alpha-core/alpha_core/xgb_predictor.py` — line 339
- `ml-portfolio-optimizer/portfolio_optimizer/rolling_xgb.py` — line 252

**Original code:**
```python
ic = np.corrcoef(y_pred_test, y_test)[0, 1]   # Pearson
```

**Fixed code:**
```python
from scipy.stats import spearmanr
ic = spearmanr(y_pred_test, y_test).correlation   # Spearman rank
```

**Why it matters:**
Grinold & Kahn (2000) "Active Portfolio Management" explicitly defines IC as **Spearman rank correlation**, not Pearson. The reason:
- Financial returns have fat tails (kurtosis > 3)
- Pearson is distorted by outliers — one correct prediction on a 10-sigma day can inflate IC by 0.02-0.05 points
- Spearman converts both arrays to ranks first, so a 10-sigma prediction counts the same as any other correct directional call
- Practical consequence: Pearson IC was slightly inflated and less stable across windows. Spearman IC will be slightly lower and more consistent.

**Knock-on effect:**  
IC feeds into the Omega (view uncertainty) matrix in Black-Litterman. Lower, more stable IC → slightly wider view uncertainty → BL weights are less concentrated on high-IC stocks.

**Interview answer:**  
*"IC should be Spearman rank correlation, not Pearson, because financial returns are fat-tailed. Pearson would give outsized weight to a single massive correct prediction on a crash day, which inflates the metric without actually indicating model quality. Rank correlation is robust to this."*

---

### Bug #2 — Kelly sizing broken: silent ticker mismatch
**Severity:** CRITICAL  
**File fixed:** `alpha-core/alpha_core/fama_french.py` — line 399

**Original code:**
```python
factor_scores.to_csv(DATA_DIR / "factor_scores.csv", index=False)
# Saves: 0, 1, 2, ... as integer index (no ticker names in index column)
```

**Fixed code:**
```python
factor_scores.to_csv(DATA_DIR / "factor_scores.csv", index=True)
# Saves: RELIANCE, HDFCBANK, ... as named index
```

**Why it matters:**
`kelly_sizing.py` loads with `index_col=0`:
```python
scores = pd.read_csv(DATA_DIR / "factor_scores.csv", index_col=0)
```
When `fama_french.py` saves `index=False`, the CSV has raw integers `0, 1, 2...` in column 0. Loading with `index_col=0` makes those integers the index. When `kelly_sizing.py` then does `resid_vol[ticker]` where `ticker = "HDFCBANK"`, the lookup fails silently and falls back to an approximation. Every Kelly position was an approximation, not the actual formula output.

**What was happening:**
```
kelly_sizing.py line 292:
    if resid_vol is not None and ticker in resid_vol.index:
        sigma_resid = resid_vol[ticker]   # ← always MISSED because index was [0,1,2]
    else:
        sigma_resid = 0.01  # ← fallback approximation — always triggered
```

**Interview answer:**  
*"There was a CSV serialization mismatch — fama_french saved with index=False (integers as index), but kelly_sizing loaded with index_col=0 expecting tickers. The lookup always fell back to a hardcoded approximation. Fixed by saving with index=True so the ticker names appear in column 0."*

---

## HIGH SEVERITY FIXES

---

### Bug #3 — DCC Q_bar: MLE fitted on different matrix than generation
**Severity:** HIGH  
**File fixed:** `indian-risk-engine/risk_engine/dcc_engine.py` — lines 41, 136-139

**Original code:**
```python
Q_bar = np.cov(Z.T)   # Bessel-corrected covariance matrix (NOT a correlation matrix)
# ... fit a_hat, b_hat against this Q_bar via MLE ...
# Then after fitting:
std_qbar = np.sqrt(np.diag(Q_bar))
Q_bar_normalised = Q_bar / np.outer(std_qbar, std_qbar)
Q_bar = Q_bar_normalised   # now normalised — but a_hat/b_hat were fit against UNnormalised
```

**Fixed code:**
```python
Q_bar = np.corrcoef(Z.T)   # proper correlation matrix from the start
# ... fit a_hat, b_hat against this Q_bar via MLE ...
# No post-fit normalisation needed — already a correlation matrix
```

**Why it matters:**
Engle (2002) DCC specification requires Q_bar to be the unconditional correlation matrix (diagonal = 1). Using `np.cov(Z.T)` gives a covariance matrix where diagonal entries are ~1.0 but not exactly 1.0 (Bessel's correction introduces small deviations). The MLE optimised `a` and `b` against this covariance Q_bar, but the generation pass applied them with a normalised Q_bar — a mathematical inconsistency. The parameters are not optimal for the matrix they're used with.

**Engle (2002) formula:**
```
Q_t = Q_bar(1-a-b) + a*z_{t-1}z_{t-1}' + b*Q_{t-1}
```
Q_bar must be a correlation matrix (diag=1) for this recursion to produce valid correlation matrices.

**Interview answer:**  
*"The DCC engine fitted parameters a and b via MLE using np.cov which gives a covariance matrix, then switched to a normalised correlation matrix for the generation pass. These are different matrices, so the parameters are not optimal for the matrix they're applied to. Engle (2002) requires Q_bar to be a correlation matrix throughout — fixed by using np.corrcoef uniformly."*

---

### Bug #4 — BL uses two different covariance matrices (Σ mismatch)
**Severity:** HIGH  
**File fixed:** `ml-portfolio-optimizer/portfolio_optimizer/black_litterman.py` — lines 424-425, 450

**Original code:**
```python
# Equilibrium: Π = δ × Σ_dcc × w_mkt
sigma = bundle.sigma_dcc.values
pi = delta * sigma @ w_market    # uses DCC

# Canonical tilt formula: w = w_mkt + (δΣ)⁻¹(μ_BL - Π)
sigma_for_tilt = bundle.sigma_lw.values   # uses LW ← DIFFERENT MATRIX
tilt = np.linalg.solve(DELTA * sigma_for_tilt, mu_bl_arr - pi)

# Views builder also used sigma_lw:
P, Q, Omega, _ = build_views_from_xgboost(tickers, pi, bundle.sigma_lw.values)
```

**Fixed code:**
```python
sigma = bundle.sigma_dcc.values   # DCC for equilibrium
pi = delta * sigma @ w_market

# Same sigma for views and tilt:
P, Q, Omega, _ = build_views_from_xgboost(tickers, pi, bundle.sigma_dcc.values)
sigma_for_tilt = bundle.sigma_dcc.values   # DCC — same as Π computation
tilt = np.linalg.solve(DELTA * sigma_for_tilt, mu_bl_arr - pi)
```

**Why it matters:**
The canonical BL formula `w = w_mkt + (δΣ)⁻¹(μ_BL - Π)` requires the **same Σ** throughout — it's a single quadratic utility solution. Mixing DCC (regime-conditional, time-varying) for Π and Ledoit-Wolf (shrinkage, static) for the tilt inversion produces a dimensionally inconsistent system. The magnitude of the tilt `(δΣ)⁻¹` depends on the covariance scale — LW and DCC have different scales — so the tilt is wrong.

**Why this caused HDFCBANK/RELIANCE/BAJFINANCE at 20% cap:**
- Σ_LW has smaller off-diagonal entries than Σ_DCC (shrinkage compresses correlations)
- Smaller Σ → larger (δΣ)⁻¹ → larger tilt magnitudes
- Large tilts pushed high-equilibrium-Π stocks (RELIANCE, HDFCBANK) beyond the 20% cap
- After clipping, they all sat at exactly 20% regardless of XGBoost views

**Interview answer:**  
*"Black-Litterman's canonical formula derives from a single quadratic utility problem, so it requires one Σ throughout. I was using DCC for the equilibrium returns and Ledoit-Wolf for the tilt inversion — two different scales. The LW shrinkage compressed off-diagonal entries, making (δΣ)⁻¹ larger, which amplified tilts and pushed high-weight stocks to the 20% cap. Fixed by using DCC uniformly."*

---

### Bug #5 — Turnover double-counted: all cost analysis wrong
**Severity:** HIGH  
**File fixed:** `ml-portfolio-optimizer/portfolio_optimizer/backtester.py` — line 490

**Original code:**
```python
turnover = float(np.abs(w[k] - prev_weights[k]).sum())
# Missing: / 2
```

**Fixed code:**
```python
turnover = float(np.abs(w[k] - prev_weights[k]).sum()) / 2.0
```

**Why it matters:**
Industry standard definition of turnover: `sum(|w_new - w_old|) / 2`

The reason for dividing by 2: if you increase HDFCBANK from 10% to 15%, you must also decrease something else from X% to X-5%. The sum of absolute changes counts both the buy and the sell. But you're only paying transaction costs on one leg (the notional traded is 5% of portfolio, not 10%).

**Impact on reported numbers:**
| Strategy | Reported (wrong) | Actual (correct) |
|---|---|---|
| MVO | 39.16% | ~19.6% |
| HRP | 14.25% | ~7.1% |
| BL | ~25% | ~12.5% |

This made the after-cost Sharpe ratios too pessimistic by 2×. HRP's competitive advantage over MVO after costs is **even larger** than was reported, because HRP has genuinely lower transaction costs.

**Interview answer:**  
*"Turnover should be sum(|Δw|)/2, not sum(|Δw|). Every buy has a matching sell — you're trading 5% of portfolio, not 10%. The /2 normalises for this. My reported turnover was 2× the true cost, making all strategies look more expensive than they are."*

---

### Bug #6 — EGARCH parameter extraction silently fails for ICICIBANK
**Severity:** HIGH  
**File fixed:** `indian-risk-engine/risk_engine/garch_model.py` — lines 250-254

**Original code:**
```python
omega = result.params['omega']      # KeyError if 'omega' named differently
alpha = result.params['alpha[1]']   # KeyError for EGARCH (different param name)
beta  = result.params['beta[1]']    # KeyError for EGARCH
```
The broad `except Exception` at line 291 catches this silently — ICICIBANK is dropped from the DCC portfolio.

**Fixed code:**
```python
omega = result.params.get('omega', result.params.get('Omega', 0.0))
if 'alpha[1]' in result.params:
    alpha = result.params['alpha[1]']
elif 'alpha' in result.params:
    alpha = result.params['alpha']
else:
    alpha_keys = [k for k in result.params.index
                  if k not in ('omega', 'Omega') and 'beta' not in k ...]
    alpha = result.params[alpha_keys[0]] if alpha_keys else 0.0
    logger.warning("'alpha[1]' not found for %s, using '%s'=%.4f", stock, ...)

beta  = result.params.get('beta[1]', result.params.get('beta', 0.0))
gamma = result.params.get('gamma[1]', result.params.get('gamma', 0.0))
```

**Why it matters:**
EGARCH in the `arch` library uses different parameter names than GARCH. If `alpha[1]` raises a KeyError inside the `try` block, the `except Exception` catches it and logs `ICICIBANK: FAILED`. ICICIBANK never gets added to `sigma_dict` or `z_dict`. The DCC matrix then has 13 stocks instead of 14, and the correlation structure is wrong.

**Interview answer:**  
*"The GARCH engine used hardcoded parameter names 'alpha[1]' and 'beta[1]' that work for standard GARCH but fail for EGARCH which uses different names. The broad except caught the KeyError silently — ICICIBANK was dropped from the DCC portfolio without any visible error. Fixed with .get() fallback logic that also logs a warning when the name mismatch occurs."*

---

## MEDIUM SEVERITY FIXES

---

### Bug #7 — Regime labels have no staleness check
**Severity:** MEDIUM  
**File fixed:** `alpha-core/alpha_core/hmm_regime.py` — `detect_current_regime()` function

**Original code:**
```python
df = pd.read_csv(label_path, index_col=0, parse_dates=True)
regime = df["regime_name"].iloc[-1]
logger.info("Current regime (from cache): %s", regime)
return regime
```

**Fixed code:**
```python
df = pd.read_csv(label_path, index_col=0, parse_dates=True)
regime = df["regime_name"].iloc[-1]
last_date = df.index[-1]
days_stale = (pd.Timestamp.today().normalize() - last_date).days
if days_stale > 3:
    logger.warning(
        "⚠ STALE REGIME: last label is %d days old (date: %s, regime: %s). "
        "Re-run hmm_regime.py to refresh. Trading on stale signal.",
        days_stale, last_date.date(), regime
    )
return regime
```

**Why it matters:**
If the pipeline isn't run for a weekend + a few days (5 calendar days), the Alpaca strategy would trade a Friday Bear signal on Wednesday with no warning. HMM regime changes can happen fast — the COVID crash went from Sideways to Bear in 3 days.

**Interview answer:**  
*"detect_current_regime() loaded the cached CSV and returned the last row with no date validation. If the pipeline hadn't run for 2 weeks, the live strategy would trade on a 2-week-old Bear/Bull signal silently. Added a 3-day staleness check that emits a warning — not an exception, because markets might be closed for holidays."*

---

### Bug #8 — Vajra BL uses δ=1.0, Kuber uses δ=2.5
**Severity:** MEDIUM  
**File fixed:** `indian-risk-engine/risk_engine/black_litterman.py` — line 112

**Original code:**
```python
delta = 1.0
```

**Fixed code:**
```python
delta = 2.5   # He-Litterman (1999) standard
```

**Why it matters:**
The equilibrium implied return is `Π = δ × Σ × w_market`. With δ=1.0, all implied returns are 60% of what they should be vs δ=2.5. When both Vajra and Kuber run BL, they produce different equilibrium prior vectors for the same market weights — they can't be compared or chained.

He-Litterman (1999) derive δ=2.5 as the implied risk aversion of a representative investor holding the global market portfolio. This is the standard used by Goldman Sachs and most institutional BL implementations.

**Interview answer:**  
*"Black-Litterman's risk aversion parameter δ should be ~2.5, the He-Litterman standard derived from the global market portfolio. My Vajra module used δ=1.0, understating equilibrium returns by 60%. Since Kuber used 2.5, the two systems were calibrated inconsistently — can't compare or chain their outputs."*

---

### Bug #9 — Greeks T mixes calendar days with trading-day denominator
**Severity:** MEDIUM  
**File fixed:** `indian-risk-engine/risk_engine/greeks_calculator.py` — line 218

**Original code:**
```python
T = max((expiry_date - pd.Timestamp.now()).days, 1) / 252
# (expiry - today).days gives CALENDAR days
# Dividing calendar days by 252 is wrong
```

**Fixed code:**
```python
T = max((expiry_date - pd.Timestamp.now()).days, 1) / 365
```

**Why it matters:**
Black-Scholes requires T in **years**. To convert, you need to divide by the number of units in a year using the same counting convention:
- Calendar days → divide by 365
- Trading days → divide by 252

`.days` counts calendar days. Dividing by 252 gives a T that is too large (30/252=0.119 vs correct 30/365=0.082). This inflates option prices by ~15-20% for short-dated options.

**Example:**
| Expiry | Wrong T | Correct T | Price error |
|---|---|---|---|
| 30 days | 0.119 | 0.082 | ~15% |
| 7 days | 0.028 | 0.019 | ~20% |
| 90 days | 0.357 | 0.247 | ~12% |

**Interview answer:**  
*"Black-Scholes T must be in years, using the same counting convention as the numerator. `.days` returns calendar days so the denominator must be 365, not 252. Using 252 as the denominator with calendar days produced a T that was 1.45× too large, inflating short-dated option prices by 15-20%."*

---

### Bug #10 — Live trading includes partial intraday bar in MA calculation
**Severity:** MEDIUM  
**File fixed:** `live-trading-alpha/AlpacaDaily.py` — line 53

**Original code:**
```python
spy = yf.download(SYMBOL, period="6mo", auto_adjust=True, progress=False)
close = spy["Close"].squeeze()
# During REGULAR hours, iloc[-1] is today's partial bar (e.g. 11am price)
ma_10 = close.rolling(10).mean()   # includes partial bar
```

**Fixed code:**
```python
spy = yf.download(SYMBOL, period="6mo", auto_adjust=True, progress=False)
close = spy["Close"].squeeze()
if session == "REGULAR":
    close = close.iloc[:-1]   # drop incomplete intraday bar
    print("Note: dropped partial intraday bar — MAs computed on last fully-closed session.")
ma_10 = close.rolling(10).mean()
```

**Why it matters:**
yfinance's `period="6mo"` includes today's partial price bar when the market is open. If the strategy runs at 11am, `close.iloc[-1]` is the 11am price, not the previous day's close. The MA signal computed at 11am is different from the signal at 2pm and different from the signal at 4pm close — signals are non-reproducible intraday. For a daily rebalancing strategy, you should only use confirmed closing bars.

**Interview answer:**  
*"yfinance includes the partial intraday bar when downloading during market hours. MA signals computed against an in-progress bar change throughout the day — a signal at 11am is different from 2pm. For a daily rebalancing strategy you want signals from the previous fully-closed session. Fixed by dropping iloc[-1] during REGULAR hours."*

---

### Bug #11 — Stress test cumulative P&L uses linear sum of log returns
**Severity:** MEDIUM  
**File fixed:** `indian-risk-engine/risk_engine/stress_test.py` — line 70

**Original code:**
```python
# daily_returns are log returns from vajra_returns.csv
cumulative_pnl = daily_pnl.cumsum()
# daily_pnl = daily_returns * PORTFOLIO_VALUE
# This computes: sum(log_r) × portfolio_value — WRONG
```

**Fixed code:**
```python
log_cumsum     = np.cumsum(daily_returns)        # sum of log returns
cum_return     = np.exp(log_cumsum) - 1          # geometric compound return
cumulative_pnl = cum_return * PORTFOLIO_VALUE    # correct ₹ P&L
```

**Why it matters:**
`vajra_returns.csv` stores **log returns** (computed as `log(P_t / P_{t-1})`). The correct way to compound log returns over N days is:
```
total_return = exp(sum(log_r_1 ... log_r_N)) - 1
```
The linear cumsum is only correct for **arithmetic** (simple) returns. For a 20-day COVID crash window with -2% log returns daily, the error is:
```
Linear:   sum = -40%  → ₹40L loss reported
Correct:  exp(-0.40)-1 = -33%  → ₹33L loss
```
That's a 21% reporting error in the stress number. Ironically, log-return linear cumsum overstates the loss (because negative compounding is nonlinear).

**Interview answer:**  
*"vajra_returns.csv stores log returns. To get compound P&L from log returns you need exp(sum(log_r)) - 1, not sum(log_r). For a 20-day scenario, the linear error is ~1-7% depending on the magnitude of daily moves — small but technically wrong and overstates losses due to the nonlinearity of negative compounding."*

---

## SUMMARY TABLE

| # | Severity | File | Issue | Status |
|---|---|---|---|---|
| 1 | CRITICAL | xgb_predictor.py L339, rolling_xgb.py L252 | Pearson IC → Spearman | ✅ Fixed |
| 2 | CRITICAL | fama_french.py L399 | index=False → ticker mismatch in Kelly | ✅ Fixed |
| 3 | HIGH | dcc_engine.py L41 | Q_bar: np.cov → np.corrcoef | ✅ Fixed |
| 4 | HIGH | black_litterman.py L424,450 | sigma_lw/sigma_dcc mixed in BL formula | ✅ Fixed |
| 5 | HIGH | backtester.py L490 | Turnover /2 missing | ✅ Fixed |
| 6 | HIGH | garch_model.py L251 | EGARCH KeyError silent drop of ICICIBANK | ✅ Fixed |
| 7 | MEDIUM | hmm_regime.py L611 | No staleness check on regime cache | ✅ Fixed |
| 8 | MEDIUM | black_litterman.py L112 (Vajra) | delta=1.0 → 2.5 | ✅ Fixed |
| 9 | MEDIUM | greeks_calculator.py L218 | T=days/252 → /365 | ✅ Fixed |
| 10 | MEDIUM | AlpacaDaily.py L53 | Partial intraday bar in MA signal | ✅ Fixed |
| 11 | MEDIUM | stress_test.py L70 | cumsum → exp(cumsum)-1 for log returns | ✅ Fixed |
| 12 | CRITICAL | hmm_regime.py | HMM miscalibrated to volatility; Sharpe-sort broken | ✅ Fixed |
| 13 | CRITICAL | alpaca_gate.py | Duplicate orders via unaggregated ETF proxy targets | ✅ Fixed |

---

## KEY REFERENCES
- Grinold & Kahn (2000) "Active Portfolio Management" — IC as Spearman rank correlation
- He & Litterman (1999) "The Intuition Behind Black-Litterman" — δ=2.5, canonical BL formula
- Engle (2002) "Dynamic Conditional Correlation" — Q_bar as correlation matrix, DCC recursion
- Black-Scholes (1973) — T in years using consistent calendar/trading convention

### Bug 14: XGBoost Signal 1-Day Staleness (Lookahead Prevention Error)
- **Location:** `alpha-core/alpha_core/xgb_predictor.py` and `ml-portfolio-optimizer/portfolio_optimizer/rolling_xgb.py`
- **Issue:** The `target` column was defined as `r.shift(-1)`. Calling `df.dropna()` dropped the final row of the dataset because tomorrow’s residual is unknown. The model then extracted `latest_features` from $t-1$, predicting the residual for $t$ (which is already known!). The live signal was entirely stale by 1 day.
- **Fix:** Used `df.dropna(subset=feature_cols)` to preserve the final row $t$ with a `NaN` target. NaNs in the target are only dropped during the train/test split, ensuring the live signal is correctly generated for $t+1$.
- **Interview Answer:** "I discovered a subtle lookahead-prevention bug in the XGBoost pipeline where calling dropna() on the target variable silently discarded the most recent day of features. The model ended up predicting today’s residual instead of tomorrow’s. I fixed this by subsetting the dropna() call, guaranteeing the live signal uses $t$ to predict $t+1$."

### Bug 15: Pairs Trading Kelly Sizing Zero-Crossing Return Blowup
- **Location:** `alpha-core/alpha_core/kelly_sizing.py`
- **Issue:** The daily spread return was calculated using `spread.pct_change()`. Because the spread is a dollar-neutral, zero-crossing mean-reverting series, percentage change is mathematically invalid and resulted in infinite returns when the spread crossed zero. This caused $\sigma^2$ to explode and crushed the Kelly sizing to 0.
- **Fix:** Switched to dollar P&L divided by estimated gross capital: `spread.diff() / (100 * (1 + |beta|))`.
- **Interview Answer:** "I fixed a critical mathematical flaw in pairs sizing where pct_change was applied to a dollar-neutral spread. Since the spread crosses zero, the returns exploded to infinity. I refactored the Kelly calculation to use the absolute difference scaled by the gross capital deployed, stabilizing the variance estimation and producing correct Kelly sizes."

### Bug 16: Portfolio Walk-Forward Backtester Compounding Log Returns
- **Location:** `ml-portfolio-optimizer/portfolio_optimizer/backtester.py`
- **Issue:** The backtester multiplied the portfolio weights by the daily returns loaded from `vajra_returns.csv`, which are log returns. The weighted sum of log returns was then compounded using `(1 + r).cumprod()`. Treating log returns as simple returns understates the true geometric compounding and invalidates cross-sectional dot products.
- **Fix:** Converted log returns to simple returns (`np.exp(r) - 1.0`) at the start of the walk-forward loop. The portfolio dot products and cumulative wealth calculations are now mathematically rigorous.
- **Interview Answer:** "In the backtesting engine, I identified that the asset returns were log returns, but the portfolio return calculation used a simple weighted sum and standard geometric compounding. A linear combination of log returns does not equal the log return of a portfolio. I added a conversion layer to exponentiate the log returns into simple returns before applying portfolio weights, ensuring the walk-forward P&L is perfectly accurate."

---

## AUDIT ROUND 2 — 2026-06-12

---

### Bug A1 — CRITICAL: XGBoost early-stops on the test set (data leakage)
**Severity:** CRITICAL  
**File fixed:** `alpha-core/alpha_core/xgb_predictor.py` (~line 323)

**Original code:**
```python
model.fit(X_train, y_train,
    eval_set=[(X_test, y_test)],   # ← TEST SET used for model selection
    ...)
```

**Fixed code:**
```python
# Carve val slice from end of train window (July–Dec 2023, ~15%)
train_pure = train[train.index < VAL_START]
val        = train[train.index >= VAL_START]

model.fit(X_train_pure, y_train_pure,
    eval_set=[(X_val, y_val)],   # ← val fold only; test stays untouched
    ...)
```

**Why it matters:**  
Early-stopping selects the number of boosting rounds by minimising loss on the eval set. Using the test set here means the test set participated in model selection — every reported test metric (IC, R², DirAcc) was optimistically biased. `ic_test` flows downstream: it gates signals in `main.py` (`ic_test > 0.05`) and calibrates BL view confidence (Idzorek Ω). Inflated IC → inflated view confidence everywhere.

**Interview answer:**  
*"My early stopping used the test fold, which leaks model-selection information. I re-split train into train/val (last 15% of the training window), and test IC fell from the biased value to the honest number. That drop is itself the audit story — it's the clean IC. It still cleared my 0.05 gate on N stocks."*

---

### Bug A2 — CRITICAL: Historical regime labels use full-sample Viterbi (look-ahead)
**Severity:** CRITICAL  
**Files fixed:** `alpha-core/alpha_core/hmm_regime.py` (~line 589), `alpha-core/alpha_core/kelly_sizing.py` (docstring ~line 107)

**Original code:**
```python
regime_ints = model.predict(X)   # Viterbi: uses ALL data → look-ahead for historical rows
```

**Fixed code:**
```python
# Forward-algorithm filtered probabilities — P(state | obs_1..obs_t)
# Conditions ONLY on past data at each t — no look-ahead
filtered_probs = model.predict_proba(X)          # shape: (T, K)
regime_ints    = filtered_probs.argmax(axis=1)   # filtered estimate per date

# Viterbi kept for live readout ONLY (last row is identical to filtered at T)
viterbi_ints             = model.predict(X)
today_regime_int_viterbi = int(viterbi_ints[-1])
```

**Why it matters:**  
Viterbi decodes the globally most probable path — the label on any historical date uses data from after that date. Those smoothed labels were consumed as a lagged feature in `xgb_predictor` (line 261) and as Kelly's regime gate history. The SSRN paper explicitly used forward-pass probabilities to avoid this bias; the production code contradicted the paper. Anyone reading both would catch this immediately.  

The docstring in `kelly_sizing.py` was also fixed: it incorrectly claimed Viterbi rows were "most-probable state given all history up to that date" — which is only true for the forward pass.

**Interview answer:**  
*"Viterbi labels historical dates using future data — it's a smoothed sequence, not a causal one. My paper used forward-pass filtered probabilities to avoid this; my code used Viterbi. I switched historical decoding to predict_proba() argmax (causal, no look-ahead) and kept Viterbi only for the terminal live readout where both methods are equivalent."*

---

### Bug B1 — HIGH: abs(full_kelly) converts negative edge into positive size
**Severity:** HIGH  
**File fixed:** `alpha-core/alpha_core/kelly_sizing.py` (lines ~194, ~312)

**Original code:**
```python
effective_f = HALF_KELLY * abs(full_kelly) * regime_mult
```

**Fixed code:**
```python
# Kelly says "no bet" when μ ≤ 0; floor at 0 so size collapses correctly
effective_f = HALF_KELLY * max(full_kelly, 0.0) * regime_mult
```

**Also removed:** The unreachable `action = "SHORT"` branch in `compute_factor_kelly()`. Negative alpha stocks were already SKIPped before reaching that branch — dead code.

**Why it matters:**  
A pair with negative spread drift (μ < 0) received the same Kelly size as a positive-edge pair. Trade direction came from the z-score signal, but magnitude should collapse to zero when there's no edge. `abs()` was mathematically wrong — Kelly explicitly requires a signed edge.

**Interview answer:**  
*"Kelly's f* = μ/σ² is negative when μ is negative, meaning 'no bet'. Taking abs() converted negative-edge pairs into same-sized positive bets. I floored at 0 so the size correctly goes to zero when there's no edge."*

---

### Bug B2 — HIGH: Crash-on-run logger artifacts
**Severity:** HIGH  
**Files fixed:** `indian-risk-engine/risk_engine/var_calculator.py:167`, `risk_engine/pca_decomp.py:70-82`, `risk_engine/monte_carlo_var.py:101-103`

**Issues and fixes:**
- `var_calculator.py:167`: `logger.info()` with no argument → `TypeError`. Fixed: `logger.info("")`
- `pca_decomp.py:70-82`: `logger.info(f"...", end="")` — logging does not support `end=` kwarg; crashed on import (module-level). Replaced with single f-string per log line, wrapped in `run()` to prevent crash-on-import.
- `monte_carlo_var.py:101-103`: `logger.info("msg:", value)` multi-arg calls → logging treated value as `*args` in `%`-style format, emitting internal errors. Fixed with f-strings: `logger.info(f"msg: {value}")`.

**Interview answer:**  
*"Three files had broken logger calls that crashed on run or import. The pca_decomp.py logger.info calls used end= which the logging module doesn't support. Monte Carlo had multi-arg calls without % placeholders. All fixed with f-strings and bare no-arg calls replaced with logger.info('')."*

---

### Bug B3 — HIGH: AI-paste artifacts in dcc_engine.py + monte_carlo_var.py
**Severity:** HIGH  
**Files fixed:** `indian-risk-engine/risk_engine/dcc_engine.py` (lines ~224-236), `risk_engine/monte_carlo_var.py` (lines 97-104)

**dcc_engine.py:** The SAVE OUTPUTS section existed twice, separated by a literal `# ... rest unchanged` AI editing marker. First block + marker removed; second (complete) block retained.

**monte_carlo_var.py:** Simulation run twice — once inside `np.errstate` block with bad `logger.info` calls, then recomputed outside. First block (debug artifact) removed entirely.

**Why it matters:**  
The `# ... rest unchanged` comment is physical evidence of careless AI-assisted editing. A skeptical interviewer who opens the file sees it immediately. The duplicate simulation was harmless but wasteful and confusing.

**Interview answer:**  
*"I found an AI editing artifact — a '# ... rest unchanged' comment with a duplicate save block. I audited the full file and removed the duplicate, leaving a single clean save section. The Monte Carlo also had a leftover debug simulation block; removed it so the simulation only runs once."*

---

## UPDATED SUMMARY TABLE

| # | Severity | File | Issue | Status |
|---|---|---|---|---|
| 1 | CRITICAL | xgb_predictor.py, rolling_xgb.py | Pearson IC → Spearman | ✅ Fixed |
| 2 | CRITICAL | fama_french.py | index=False → ticker mismatch in Kelly | ✅ Fixed |
| 3 | HIGH | dcc_engine.py | Q_bar: np.cov → np.corrcoef | ✅ Fixed |
| 4 | HIGH | black_litterman.py | sigma_lw/sigma_dcc mixed in BL formula | ✅ Fixed |
| 5 | HIGH | backtester.py | Turnover /2 missing | ✅ Fixed |
| 6 | HIGH | garch_model.py | EGARCH KeyError silent drop of ICICIBANK | ✅ Fixed |
| 7 | MEDIUM | hmm_regime.py | No staleness check on regime cache | ✅ Fixed |
| 8 | MEDIUM | black_litterman.py (Vajra) | delta=1.0 → 2.5 | ✅ Fixed |
| 9 | MEDIUM | greeks_calculator.py | T=days/252 → /365 | ✅ Fixed |
| 10 | MEDIUM | AlpacaDaily.py | Partial intraday bar in MA signal | ✅ Fixed |
| 11 | MEDIUM | stress_test.py | cumsum → exp(cumsum)-1 for log returns | ✅ Fixed |
| 12 | CRITICAL | hmm_regime.py | HMM miscalibrated to volatility; Sharpe-sort broken | ✅ Fixed |
| 13 | CRITICAL | alpaca_gate.py | Duplicate orders via unaggregated ETF proxy targets | ✅ Fixed |
| 14 | CRITICAL | xgb_predictor.py | XGBoost 1-day staleness (lookahead prevention) | ✅ Fixed |
| 15 | CRITICAL | kelly_sizing.py | Pairs Kelly zero-crossing return blowup | ✅ Fixed |
| 16 | MEDIUM | backtester.py | Log returns compounded as simple returns | ✅ Fixed |
| A1 | CRITICAL | xgb_predictor.py | Early stopping on test set → biased IC | ✅ Fixed |
| A2 | CRITICAL | hmm_regime.py, kelly_sizing.py | Viterbi look-ahead on historical labels | ✅ Fixed |
| B1 | HIGH | kelly_sizing.py | abs(full_kelly) sizes negative-edge pairs | ✅ Fixed |
| B2 | HIGH | var_calculator.py, pca_decomp.py, monte_carlo_var.py | Broken logger calls crash on run/import | ✅ Fixed |
| B3 | HIGH | dcc_engine.py, monte_carlo_var.py | AI-paste artifacts: duplicate blocks + marker comment | ✅ Fixed |
