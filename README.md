# 📉 GARCH Risk Engine

> **Institutional-grade portfolio risk analytics for NSE equities.**  
> GARCH(1,1) volatility → DCC dynamic correlation → VaR/CVaR → Monte Carlo → Stress Testing → Greeks → SHAP attribution — all in one pipeline.

[![CI](https://github.com/YASHpPATIL7/indian-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/YASHpPATIL7/indian-risk-engine/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![pytest](https://img.shields.io/badge/tests-44%2F44%20passing-10b981)](https://github.com/YASHpPATIL7/indian-risk-engine/actions)

**Live Dashboard:** [Streamlit Cloud →](https://indian-risk-engine-fdjkqvvpks4wrjaiu5svhm.streamlit.app) &nbsp;|&nbsp; **Paper:** [arXiv →](#)

---

## What This Does

The GARCH Risk Engine answers three questions for every stock in the portfolio:

1. **How much can I lose?** → VaR, CVaR, Monte Carlo VaR
2. **How volatile is it right now?** → GARCH(1,1) conditional volatility, DCC correlations
3. **Where is the risk concentrated?** → SHAP variance decomposition, PCA, stress tests

Every downstream decision — signal generation, position sizing, rebalancing — is gated by this engine's output.

---

## Dashboard

Six-tab Streamlit dashboard: Price & Volatility · Return Distribution · Rolling VaR · Correlation Regimes · SHAP Attribution · Greeks Engine

![Dashboard](assets/dashboard-overview.png)

---

## Risk Modules

### Core Risk (14 stocks, 6+ years of data)

| Module | File | What It Computes |
|---|---|---|
| **VaR Calculator** | `var_calculator.py` | Historical VaR, Parametric VaR, CVaR (Expected Shortfall), Rolling 252-day VaR |
| **GARCH(1,1)** | `garch_model.py` | Per-stock conditional volatility with EGARCH fallback for leverage effects |
| **GARCH Diagnostics** | `garch_diagnostics.py` | Ljung-Box autocorrelation test + ARCH-LM heteroscedasticity test on residuals |
| **DCC Engine** | `dcc_engine.py` | Dynamic Conditional Correlation — time-varying covariance matrix (T × N × N cube) |
| **Portfolio VaR** | `portfolio_var.py` | Daily portfolio VaR/CVaR using DCC covariance, ₹1 Cr equal-weight portfolio |
| **Walk-Forward** | `walkforward.py` | Rolling-window VaR validation with out-of-sample breach rates |
| **Kupiec Backtest** | `kupiec_backtest.py` | Kupiec POF test + Christoffersen independence test — "is the VaR model accurate?" |
| **Stress Testing** | `stress_test.py` | Historical scenarios: Lehman (2008), COVID (2020), Adani-Hindenburg (2023) |
| **Monte Carlo VaR** | `monte_carlo_var.py` | 10K Cholesky-correlated simulations using DCC covariance |
| **Monte Carlo Options** | `monte_carlo_options.py` | GBM + GARCH-calibrated σ for option pricing across strike ladder |
| **Greeks Calculator** | `greeks_calculator.py` | Black-Scholes: Δ, Γ, ν, Θ, ρ with live yfinance/Alpaca spot data |
| **SHAP Attribution** | `shap_attribution.py` | Shapley variance decomposition — which stock drives portfolio risk |
| **PCA Decomposition** | `pca_decomp.py` | Principal components of the correlation matrix — latent risk factors |
| **Black-Litterman** | `black_litterman.py` | Market-implied priors + analyst views → posterior optimal weights |

### Data Pipeline (216 stocks)

| Module | File | What It Does |
|---|---|---|
| **Crawler** | `crawler.py` | Multi-threaded yfinance pull (10 workers), 216 NSE tickers, ~196s |
| **PostgreSQL** | `db.py` | SQLAlchemy ORM, `pool_size=10`, `UniqueConstraint(ticker, date)` |
| **Validator** | `validate.py` | 10-point data quality checks — NULLs, duplicates, extreme returns |
| **Pipeline** | `data_pipeline.py` | Log returns, stationarity (ADF test), rolling vol, clean CSV export |

---

## Key Results

### VaR Output (95% Confidence)

```
─────────────────────────────────────────────────────────
Ticker        VaR 95%    CVaR 95%   CVaR/VaR     N
─────────────────────────────────────────────────────────
RELIANCE       -2.14%      -3.08%      1.44    1236
HDFCBANK       -1.98%      -3.15%      1.59    1236
TCS            -2.02%      -3.11%      1.54    1236
INFY           -2.46%      -3.70%      1.50    1236
ADANIPORTS     -2.95%      -5.70%      1.94    1236
─────────────────────────────────────────────────────────
```

**ADANIPORTS CVaR/VaR = 1.94.** Hindenburg caused -19% single-day moves. VaR predicted -2.95%. Reality was 6× worse. CVaR captured the tail. This is why Expected Shortfall is mandated over VaR in Basel III/IV.

### Correlation Regimes (DCC-GARCH)

| Regime | Date | Mean Corr | % Pairs > 0.7 | Event |
|---|---|---|---|---|
| 🟢 Normal | Jun 2022 | 0.361 | 3.7% | Baseline — diversification works |
| 🔴 Systemic | Mar 2022 | 0.469 | 13.9% | Russia-Ukraine — global FII selloff |
| 🔵 Sector Flight | Feb 2023 | 0.142 | 0.8% | Hindenburg — Adani down, pharma up |
| 🟡 Cluster Shock | Jun 2024 | 0.256 | 12.7% | Election — PSU/infra cluster fell |

### SHAP Variance Attribution (Top 5)

| Stock | Avg. Variance Contribution |
|---|---|
| BAJFINANCE | 10.7% |
| AXISBANK | 8.9% |
| ONGC | 8.4% |
| MARUTI | 8.1% |
| SUNPHARMA | 7.8% |

---

## SEBI Algo Trading Compliance

> This engine is designed with SEBI's 2024 Algorithmic Trading Framework in mind.
> Reference: [SEBI Circular SEBI/HO/MRD2/PoD-2/P/CIR/2024/172](https://www.sebi.gov.in/legal/circulars/dec-2024/framework-for-algorithmic-trading-by-retail-investors_89768.html)

### Compliance Mapping

| SEBI Requirement | How This Engine Addresses It |
|---|---|
| **Algo-ID Registration** | Each strategy can be tagged with a unique Algo-ID before deployment |
| **Risk Controls** | VaR/CVaR position limits enforced before order generation |
| **Kill Switch** | Stress test module flags Lehman/COVID-level events for circuit breakers |
| **Audit Trail** | `risk_engine_audit.log` — every risk calculation timestamped with module name |
| **Order-level Attribution** | SHAP decomposition traces portfolio risk to individual stock contributions |
| **Backtesting** | Kupiec POF + Christoffersen tests validate VaR model accuracy |
| **Volatility Guardrails** | GARCH σ_t feeds position sizing — high vol = smaller position |
| **Correlation Monitoring** | DCC engine detects systemic crisis regime in real-time |

### Audit Log Format

```
2026-05-13 12:45:00 | risk_engine.garch_model    | INFO     | Fitting GARCH(1,1) for RELIANCE
2026-05-13 12:45:01 | risk_engine.var_calculator  | INFO     | VaR 95% = -2.14%, CVaR = -3.08%
2026-05-13 12:45:02 | risk_engine.dcc_engine      | INFO     | DCC params: a=0.0421, b=0.9505
```

Every module uses `logging.getLogger(__name__)` → centralized to `risk_engine_audit.log`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL 16 + SQLAlchemy ORM |
| Risk Models | arch (GARCH/DCC), scipy, numpy |
| ML | XGBoost, scikit-learn, SHAP |
| Options | Black-Scholes (custom), yfinance, Alpaca API |
| Portfolio | PyPortfolioOpt (Black-Litterman, ERC) |
| Dashboard | Streamlit, Plotly, seaborn, matplotlib |
| Testing | pytest |
| Logging | Python logging → `risk_engine_audit.log` |

---

## Project Structure

```
indian-risk-engine/
├── data_pipeline/
│   ├── db.py                     PostgreSQL engine + connection pool
│   ├── models.py                 ORM: PriceData + TickerMetadata
│   ├── crawler.py                Multi-threaded yfinance crawler
│   ├── data_pipeline.py          Returns, ADF test, stationarity
│   ├── validate.py               10-point data quality validator
│   └── nifty200_tickers.py       216-stock NSE universe
├── risk_engine/
│   ├── var_calculator.py         VaR, CVaR, Parametric, Rolling
│   ├── garch_model.py            GARCH(1,1) + EGARCH per stock
│   ├── garch_diagnostics.py      Ljung-Box + ARCH-LM tests
│   ├── dcc_engine.py             DCC dynamic correlation
│   ├── portfolio_var.py          Portfolio VaR/CVaR (DCC-based)
│   ├── walkforward.py            Walk-forward VaR validation
│   ├── kupiec_backtest.py        Kupiec POF + Christoffersen
│   ├── stress_test.py            Lehman / COVID / Hindenburg
│   ├── monte_carlo_var.py        10K Cholesky simulations
│   ├── monte_carlo_options.py    GBM option pricing
│   ├── greeks_calculator.py      BS Greeks + live data (yfinance/Alpaca)
│   ├── shap_attribution.py       SHAP variance decomposition
│   ├── pca_decomp.py             PCA latent risk factors
│   └── black_litterman.py        BL posterior weights
├── dashboard/
│   └── app.py                    Streamlit (5 tabs) + root logger
├── tests/
│   └── test_returns.py           pytest suite
├── data/                         Generated CSVs, PKLs
├── assets/                       Charts and screenshots
└── requirements.txt
```

---

## Known Failure Modes

> Every model has a failure envelope. Documenting it is not weakness — it's the difference between a student project and production risk infrastructure.

### 1. Structural Breaks (GARCH)

**Failure:** GARCH(1,1) assumes volatility is mean-reverting. During structural breaks (COVID Mar 2020, demonetisation Nov 2016), the regime shifts permanently — ω, α, β are estimated from a world that no longer exists.

**Evidence:** GARCH σ for ADANIPORTS in Feb 2023 (Hindenburg) was calibrated on 2019-2022 data. The model predicted σ ≈ 2.8%/day. Realised volatility hit 8.4%/day. **3× underestimation.**

**Mitigation:** Walk-forward re-estimation (252-day expanding window). DCC engine re-calibrates every window. Kill switch halts trading when GARCH σ > 5%.

### 2. Correlation Breakdown (DCC)

**Failure:** DCC assumes correlations evolve smoothly. In liquidity crises, correlations jump to ~1.0 overnight ("all correlations go to 1 in a crash"). Cholesky decomposition in Monte Carlo uses yesterday's DCC matrix — useless for tomorrow's crisis.

**Evidence:** Mar 2020 — mean pairwise correlation jumped from 0.36 to 0.71 in 5 trading days. DCC's a=0.04, b=0.95 parameters mean it adapts slowly (0.95 persistence). By the time DCC caught up, the crash was over.

**Mitigation:** Stress test module hard-codes Lehman/COVID correlation matrices for scenario analysis. Kill switch triggers at mean_corr > 0.45.

### 3. Fat Tails (VaR)

**Failure:** Parametric VaR assumes normality (z = 1.645 for 95%). Real NSE returns have kurtosis > 5 (leptokurtic). This systematically underestimates tail risk.

**Evidence:** ADANIPORTS VaR 95% = -2.95%. Actual worst day = -19.4%. CVaR/VaR ratio = 1.94 — tails are nearly 2× worse than VaR suggests.

**Mitigation:** CVaR (Expected Shortfall) reported alongside VaR. Historical VaR uses actual empirical distribution. Monte Carlo uses 10K simulations with fat-tailed residuals.

### 4. Stale Spot Prices (Greeks)

**Failure:** NSE option chain data is unavailable via yfinance. Greeks engine falls back to GARCH sigma instead of market-implied volatility. During earnings/events, implied vol can be 2-3× realised vol — GARCH won't capture this.

**Evidence:** Pre-earnings INFY implied vol ≈ 35%. GARCH σ = 22%. Delta computed with 22% vol will overestimate call prices by 15-20%.

**Mitigation:** 3-tier fallback (chain → Alpaca → GARCH). Logged with source attribution so the desk knows which vol input was used. Future: integrate NSE API for real implied vol surface.

### 5. Survivorship Bias (Universe)

**Failure:** Nifty 200 universe is current constituents. Stocks that were delisted or dropped from the index (YES Bank, DHFL, Jet Airways) are excluded — this inflates backtested returns and understates tail risk.

**Impact:** Estimated 0.5-1.2% annual return overstatement based on S&P 500 survivorship bias studies (Elton et al., 1996). Indian market likely worse due to higher corporate default rates.

**Mitigation:** Documented. Acknowledged in any backtested Sharpe ratio. Universe is frozen at pull date — no retroactive additions.

### 6. Microstructure Noise (Low-Liquidity Stocks)

**Failure:** Bid-ask bounce in small-cap NSE stocks creates artificial volatility in daily returns. GARCH interprets this as real volatility clustering, inflating σ estimates.

**Evidence:** ONGC 30-day rolling vol shows spikes of 4-5% on days with < ₹10 Cr traded volume. These aren't real risk events — they're tick noise.

**Mitigation:** Volume filter in data pipeline (20-day avg > ₹10 Cr). Risk model constrained to 14 large-cap, high-liquidity stocks. Dashboard labels clearly distinguish "216 stocks in DB" vs "14 stocks in risk model."

---

## Setup

```bash
git clone https://github.com/YASHpPATIL7/indian-risk-engine
cd indian-risk-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# PostgreSQL
createdb nifty_risk
cp .env.example .env   # add DB credentials

# Pull data (~3 min)
PYTHONPATH=. python data_pipeline/crawler.py
PYTHONPATH=. python data_pipeline/validate.py

# Run risk pipeline
PYTHONPATH=. python data_pipeline/data_pipeline.py
PYTHONPATH=. python risk_engine/garch_model.py
PYTHONPATH=. python risk_engine/dcc_engine.py
PYTHONPATH=. python risk_engine/portfolio_var.py

# Launch dashboard
PYTHONPATH=. streamlit run dashboard/app.py
```

---

## Author

**Yash Patil** — CFA Level 1 · B.Tech VJTI Mumbai  
Building quantitative risk infrastructure for Indian markets.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/yashppatil7)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/YASHpPATIL7)