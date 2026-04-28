# 📉 GARCH Risk Engine

> Production-grade risk analytics system for NSE equities — built from scratch in Python.
> Institutional-quality VaR, CVaR, rolling volatility, and dynamic correlation regime detection.







***

## What This Is

A risk management system for Indian equities that computes and visualises the same
risk metrics used by institutional desks — VaR, CVaR, rolling volatility, and
correlation regime detection — on 5 years of live Nifty 200 data.

**Not a tutorial project. Every number comes from real NSE market data.**

***

## Database

| Metric | Value |
|---|---|
| Universe | Nifty 200 (216 tickers) |
| Period | April 2021 → April 2026 |
| Total rows | 262,816 |
| Trading days/stock | ~1,236 |
| Validation | 10/10 checks passing |
| Test suite | 14/14 pytest passing |
| NaN policy | Zero tolerance on log_return |

***

## Features Built (Week 1)

### Data Pipeline
- **Multi-threaded yfinance crawler** — `ThreadPoolExecutor` with 10 workers,
  pulls 5yr OHLCV for 216 NSE tickers simultaneously
- **PostgreSQL schema** — `PriceData` + `TickerMetadata` via SQLAlchemy ORM,
  connection pooling (`pool_size=10`, `pool_pre_ping=True`), UniqueConstraint on
  `(ticker, date)` to prevent duplicates
- **Pre-computed columns** — `log_return` and `rolling_vol_30` computed at
  insert time, not at query time (15ms dashboard refresh vs 300ms lazy compute)
- **10-point validator** — row count, NULL policy, date range, duplicate check,
  extreme return detection, spot price verification

### Risk Engine
- **Historical VaR** — sort 1,236 returns, take 5th percentile. No distribution
  assumptions. RELIANCE 95% VaR = -2.14%.
- **Parametric VaR** — Gaussian assumption: `VaR = μ - 1.645σ`.
  Directly proportional to daily volatility.
- **CVaR / Expected Shortfall** — `E[X | X < VaR]`. Average of the worst 62 days.
  RELIANCE CVaR = -3.08%. ADANIPORTS CVaR = -5.70%.
- **CVaR/VaR ratio** — fat tail detector. ADANIPORTS = 1.93 (Hindenburg effect).
  RELIANCE = 1.44. Same VaR → different tail severity.
- **Rolling 252-day VaR** — time series of risk. Shows VaR spiking at
  Russia-Ukraine (Feb 2022), Hindenburg (Jan 2023), Election (Jun 2024).

### Correlation Analysis
Four market regimes identified from real NSE data:

| Regime | Date | Mean Corr | % Pairs > 0.7 | Interpretation |
|---|---|---|---|---|
| Normal | Jun 2022 | 0.361 | 3.7% | Diversification working |
| Systemic | Mar 2022 | 0.469 | 13.9% | Russia-Ukraine — all stocks fell |
| Sector Flight | Feb 2023 | 0.142 | 0.8% | Hindenburg — Adani down, IT up |
| Cluster Shock | Jun 2024 | 0.256 | 12.7% | Election — PSU cluster fell |

**Key finding:** Mean correlation alone is misleading. Hindenburg mean = 0.142
(looks safe) but hides violent sector divergence. `% pairs > 0.7` is the real
crisis detector.

### Streamlit Dashboard
Four-tab live dashboard connected to PostgreSQL:

```
Tab 1 — Price & Volatility    3-panel: adj close / rolling vol / daily returns
Tab 2 — Return Distribution   Histogram, VaR + CVaR marked, risk metrics table
Tab 3 — Rolling VaR Timeline  252-day rolling VaR, crisis events annotated
Tab 4 — Correlation Regimes   Live heatmap, 4-regime auto-detector
```

***

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL 16 + SQLAlchemy ORM |
| Data | yfinance + pandas |
| Risk math | numpy + scipy.stats |
| Visualisation | Streamlit + Plotly + seaborn |
| Testing | pytest (14/14) |
| Environment | venv + python-dotenv |

***

## Project Structure

```
indian-risk-engine/
├── data_pipeline/
│   ├── db.py                  PostgreSQL engine + connection pool
│   ├── models.py              ORM: PriceData + TickerMetadata
│   ├── crawler.py             Multi-threaded yfinance crawler
│   ├── validate.py            10-point data quality validator
│   └── nifty200_tickers.py    216-stock NSE universe
├── risk_engine/
│   └── var_calculator.py      VaR, CVaR, Rolling VaR
├── tests/
│   └── test_returns.py        14/14 pytest — log returns + vol math
├── dashboard/
│   └── app.py                 Streamlit GARCH Risk Engine
├── notebooks/
│   └── correlation_heatmap.py 4-regime heatmap generator
└── requirements.txt
```

***

## Setup

```bash
# 1. Clone and activate
git clone https://github.com/yashpatil/indian-risk-engine
cd indian-risk-engine
python -m venv venv && source venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. PostgreSQL
createdb nifty_risk
cp .env.example .env  # add your DB credentials

# 4. Pull data (takes ~3-5 minutes)
PYTHONPATH=. python data_pipeline/crawler.py

# 5. Validate
PYTHONPATH=. python data_pipeline/validate.py

# 6. Tests
PYTHONPATH=. pytest tests/ -v

# 7. Dashboard
PYTHONPATH=. streamlit run dashboard/app.py
```

***

## Key Risk Metrics — Sample Output

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

ADANIPORTS CVaR/VaR = 1.94 — Hindenburg report (Jan 2023) caused -19% single-day
moves. VaR said -2.95%. Reality was 6× worse. CVaR captured the tail. VaR didn't.

***

## Roadmap

| Week | Module | Status |
|---|---|---|
| 1 | Data pipeline + VaR/CVaR + Correlation | ✅ Complete |
| 2 | GARCH(1,1) volatility forecasting | 🔴 Next |
| 3-4 | Options Greeks engine (Delta, Gamma, Vega, Theta) | ⬜ Planned |
| 5-6 | Monte Carlo VaR + stress testing | ⬜ Planned |
| 7-8 | HMM regime detection | ⬜ Planned |
| 9-11 | FastAPI risk service | ⬜ Planned |
| 12-14 | Black-Litterman portfolio optimisation | ⬜ Planned |
| 15-19 | Execution engine + live paper trading | ⬜ Planned |

***

## Why This Exists

Built to develop institutional-grade quantitative finance skills targeting
algo trading and risk roles. Every component mirrors production systems:
connection pooling, pre-computed columns, zero-NaN policy, tested math,
and regime-aware risk models.

CFA Level 1 | 