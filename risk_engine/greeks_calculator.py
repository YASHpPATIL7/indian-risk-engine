"""
Black-Scholes Greeks Engine
GreeksCalculator class — price + all 5 Greeks for calls and puts
Test on RELIANCE, HDFCBANK, INFY using last GARCH sigma_t
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

# Configure logging for standalone execution (app.py overrides this when imported)
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)-30s | %(message)s',
    )

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. BLACK-SCHOLES GREEKS CLASS ─────────────────────────────────────────────
class GreeksCalculator:
    """
    Closed-form Black-Scholes pricer and Greeks calculator.
    All inputs in consistent units:
      S     : current stock price (₹)
      K     : strike price (₹)
      T     : time to expiry (years)  e.g. 30 days = 30/252
      r     : risk-free rate (decimal) e.g. 6.5% = 0.065
      sigma : annualised volatility (decimal) e.g. 22% = 0.22
    """

    def __init__(self, S, K, T, r, sigma):
        self.S     = S
        self.K     = K
        self.T     = T
        self.r     = r
        self.sigma = sigma
        self._compute_d1_d2()

    def _compute_d1_d2(self):
        self.d1 = (np.log(self.S / self.K) +
                   (self.r + 0.5 * self.sigma**2) * self.T) / \
                  (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    # ── PRICES ────────────────────────────────────────────────────────────────
    def call_price(self):
        return (self.S * norm.cdf(self.d1) -
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))

    def put_price(self):
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) -
                self.S * norm.cdf(-self.d1))

    # ── GREEKS ────────────────────────────────────────────────────────────────
    def delta(self):
        call = norm.cdf(self.d1)
        put  = norm.cdf(self.d1) - 1
        return {"call": call, "put": put}

    def gamma(self):
        g = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        return {"call": g, "put": g}          # same for both

    def vega(self):
        v = self.S * np.sqrt(self.T) * norm.pdf(self.d1) * 0.01  # per 1% vol
        return {"call": v, "put": v}

    def theta(self):
        common  = -(self.S * norm.pdf(self.d1) * self.sigma) / \
                   (2 * np.sqrt(self.T))
        call_th = (common - self.r * self.K *
                   np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 252
        put_th  = (common + self.r * self.K *
                   np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 252
        return {"call": call_th, "put": put_th}  # per calendar day

    def rho(self):
        call_r = (self.K * self.T * np.exp(-self.r * self.T) *
                  norm.cdf(self.d2)) * 0.01       # per 1% rate move
        put_r  = (-self.K * self.T * np.exp(-self.r * self.T) *
                  norm.cdf(-self.d2)) * 0.01
        return {"call": call_r, "put": put_r}

    def all_greeks(self):
        return {
            "call_price" : self.call_price(),
            "put_price"  : self.put_price(),
            "delta_call" : self.delta()["call"],
            "delta_put"  : self.delta()["put"],
            "gamma"      : self.gamma()["call"],
            "vega"       : self.vega()["call"],
            "theta_call" : self.theta()["call"],
            "theta_put"  : self.theta()["put"],
            "rho_call"   : self.rho()["call"],
            "rho_put"    : self.rho()["put"],
        }


# ── 2. LOAD GARCH SIGMA (FALLBACK VOLATILITY) ────────────────────────────────
sigma_path = os.path.join(BASE_DIR, "data", "vajra_sigma.pkl")
if os.path.exists(sigma_path):
    with open(sigma_path, "rb") as f:
        sigma_df = pickle.load(f)
    last_sigma = (sigma_df.iloc[-1] / 100) * np.sqrt(252)
    logger.info("GARCH annualised sigma loaded (fallback volatility)")
else:
    last_sigma = pd.Series()
    logger.warning("vajra_sigma.pkl not found — GARCH fallback unavailable")

R_FREE = 0.065  # RBI repo rate ~6.5%
US_RFREE = 0.05  # US Fed rate ~5%


# ── 3. LIVE DATA FETCHERS ─────────────────────────────────────────────────────
def fetch_yfinance_chain(ticker_ns: str):
    """
    Fetch live spot + option chain from yfinance for NSE stocks.
    Returns: (spot, chain_df, source) or (None, None, None) on failure.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker_ns)
        # Spot price
        hist = tk.history(period="1d")
        if hist.empty:
            return None, None, None
        spot = float(hist["Close"].iloc[-1])

        # Option chain — nearest expiry
        expiries = tk.options
        if not expiries:
            logger.info(f"  {ticker_ns}: no option chain available, using GARCH fallback")
            return spot, None, "yfinance_spot_only"

        nearest = expiries[0]
        chain = tk.option_chain(nearest)
        calls = chain.calls[["strike", "lastPrice", "impliedVolatility", "volume", "openInterest"]].copy()
        calls["expiry"] = nearest
        calls["type"] = "call"
        puts = chain.puts[["strike", "lastPrice", "impliedVolatility", "volume", "openInterest"]].copy()
        puts["expiry"] = nearest
        puts["type"] = "put"
        chain_df = pd.concat([calls, puts], ignore_index=True)

        logger.info(f"  {ticker_ns}: spot=₹{spot:.0f}, expiry={nearest}, "
                    f"{len(calls)} calls + {len(puts)} puts")
        return spot, chain_df, "yfinance_chain"
    except Exception as e:
        logger.warning(f"  {ticker_ns}: yfinance failed — {e}")
        return None, None, None


def fetch_alpaca_options(symbol: str):
    """
    Fetch live spot + option snapshot from Alpaca for US stocks (SPY, AAPL etc).
    Returns: (spot, chain_df, source) or (None, None, None) on failure.
    """
    try:
        import alpaca_trade_api as tradeapi
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(BASE_DIR), "live-trading-alpha", ".env")
        load_dotenv(env_path, override=True)

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            return None, None, None

        api = tradeapi.REST(api_key, secret_key,
                            "https://paper-api.alpaca.markets", api_version="v2")
        # Spot
        trade = api.get_latest_trade(symbol)
        spot = float(trade.p)

        logger.info(f"  {symbol} (Alpaca): spot=${spot:.2f}")
        return spot, None, "alpaca_spot"
    except Exception as e:
        logger.warning(f"  {symbol}: Alpaca failed — {e}")
        return None, None, None


def get_live_greeks(tickers_nse: list, tickers_us: list = None):
    """
    Compute Greeks using live data with 3-tier fallback:
      1. yfinance option chain → real strikes + implied vol
      2. Alpaca API → US stocks spot price
      3. GARCH sigma → last resort (real model output, not hardcoded)
    """
    if tickers_us is None:
        tickers_us = []

    records = []
    EXPIRIES_FALLBACK = {"30d": 30/252, "60d": 60/252, "90d": 90/252}
    STRIKE_OFFSETS = {"ATM": 1.00, "OTM_5pct": 1.05, "OTM_10pct": 1.10}

    # ── NSE STOCKS ──
    for ticker in tickers_nse:
        ticker_ns = f"{ticker}.NS"
        spot, chain_df, source = fetch_yfinance_chain(ticker_ns)

        if spot is None:
            # Tier 3: pure GARCH fallback with approximate spot
            logger.warning(f"  {ticker}: all sources failed, skipping")
            continue

        if chain_df is not None and not chain_df.empty:
            # Tier 1: REAL option chain — compute Greeks on actual strikes
            from datetime import datetime
            expiry_date = pd.to_datetime(chain_df["expiry"].iloc[0])
            T = max((expiry_date - pd.Timestamp.now()).days, 1) / 252

            calls = chain_df[chain_df["type"] == "call"]
            for _, row in calls.iterrows():
                K = float(row["strike"])
                iv = float(row["impliedVolatility"]) if row["impliedVolatility"] > 0 else 0.25
                gc = GreeksCalculator(spot, K, T, R_FREE, iv)
                greeks = gc.all_greeks()
                greeks.update({
                    "stock": ticker, "spot": spot, "strike": K,
                    "expiry": str(expiry_date.date()),
                    "strike_type": "CHAIN",
                    "sigma_pct": iv * 100, "T_years": T,
                    "source": "yfinance_chain",
                    "market_price": float(row["lastPrice"]),
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    "open_interest": int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0,
                })
                records.append(greeks)
        else:
            # Tier 2: spot from yfinance, vol from GARCH
            sigma = last_sigma.get(ticker, 0.25)
            for exp_label, T in EXPIRIES_FALLBACK.items():
                for s_label, mult in STRIKE_OFFSETS.items():
                    K = spot * mult
                    gc = GreeksCalculator(spot, K, T, R_FREE, sigma)
                    greeks = gc.all_greeks()
                    greeks.update({
                        "stock": ticker, "spot": spot, "strike": K,
                        "expiry": exp_label, "strike_type": s_label,
                        "sigma_pct": sigma * 100, "T_years": T,
                        "source": "garch_fallback",
                    })
                    records.append(greeks)

    # ── US STOCKS (Alpaca) ──
    for symbol in tickers_us:
        spot, _, source = fetch_alpaca_options(symbol)
        if spot is None:
            continue
        sigma = 0.18  # SPY historical vol ~18%
        for exp_label, T in EXPIRIES_FALLBACK.items():
            for s_label, mult in STRIKE_OFFSETS.items():
                K = spot * mult
                gc = GreeksCalculator(spot, K, T, US_RFREE, sigma)
                greeks = gc.all_greeks()
                greeks.update({
                    "stock": symbol, "spot": spot, "strike": K,
                    "expiry": exp_label, "strike_type": s_label,
                    "sigma_pct": sigma * 100, "T_years": T,
                    "source": "alpaca",
                })
                records.append(greeks)

    return pd.DataFrame(records)


# ── 4. RUN LIVE GREEKS ────────────────────────────────────────────────────────
NSE_TICKERS = ["RELIANCE", "HDFCBANK", "INFY"]
US_TICKERS = ["SPY"]

logger.info("\n" + "="*70)
logger.info("LIVE GREEKS COMPUTATION")
logger.info("="*70)
results_df = get_live_greeks(NSE_TICKERS, US_TICKERS)

# ── 5. PRINT RESULTS ──────────────────────────────────────────────────────────
logger.info("\n" + "="*80)
logger.info("BLACK-SCHOLES GREEKS — FULL TABLE")
logger.info("="*80)

if not results_df.empty:
    for stock in results_df["stock"].unique():
        sub = results_df[results_df["stock"] == stock]
        spot = sub["spot"].iloc[0]
        src = sub["source"].iloc[0] if "source" in sub.columns else "unknown"
        logger.info(f"\n{'─'*70}")
        logger.info(f"  {stock}  |  Spot: {spot:.0f}  |  σ: {sub['sigma_pct'].mean():.2f}%  |  Source: {src}")
        logger.info(f"{'─'*70}")
        logger.info(f"  {'Expiry':<8} {'Strike':<12} {'Call₹':>8} {'Put₹':>8} "
              f"{'Δcall':>7} {'Γ':>8} {'ν₹':>7} {'Θ₹/d':>8} {'ρ₹':>7}")
        logger.info(f"  {'─'*78}")
        display_rows = sub.head(15) if len(sub) > 15 else sub
        for _, r in display_rows.iterrows():
            label = r.get('strike_type', 'CHAIN')
            logger.info(f"  {str(r['expiry']):<8} {label:<12} "
                  f"₹{r['call_price']:>6.1f} ₹{r['put_price']:>6.1f} "
                  f"{r['delta_call']:>7.3f} {r['gamma']:>8.5f} "
                  f"₹{r['vega']:>5.2f} ₹{r['theta_call']:>6.3f} "
                  f"₹{r['rho_call']:>5.2f}")
        if len(sub) > 15:
            logger.info(f"  ... {len(sub) - 15} more rows")
else:
    logger.warning("No results — all data sources failed")

# ── 6. PUT-CALL PARITY CHECK ──────────────────────────────────────────────────
logger.info("\nPut-Call Parity Check (C - P = S - Ke^-rT):")
logger.info(f"{'Stock':<12} {'Expiry':<8} {'Strike':<12} {'C-P':>8} {'S-PV(K)':>10} {'Error':>8}")
logger.info("─"*55)
for _, r in results_df[results_df["strike_type"]=="ATM"].iterrows():
    c_minus_p = r["call_price"] - r["put_price"]
    s_minus_pvk = r["spot"] - r["strike"] * np.exp(-R_FREE * r["T_years"])
    error = abs(c_minus_p - s_minus_pvk)
    logger.info(f"{r['stock']:<12} {r['expiry']:<8} {r['strike_type']:<12} "
          f"₹{c_minus_p:>6.2f} ₹{s_minus_pvk:>8.2f} ₹{error:>6.4f}")

# ── 7. DELTA SURFACE — STRIKE vs EXPIRY ───────────────────────────────────────
logger.info("\nDelta Surface (Call) — RELIANCE:")
rel = results_df[results_df["stock"]=="RELIANCE"]
pivot = rel.pivot(index="strike_type", columns="expiry", values="delta_call")
logger.info(pivot.to_string())

# ── 8. SAVE CSV ───────────────────────────────────────────────────────────────
results_df.to_csv(
    os.path.join(BASE_DIR, "data", "greeks_results.csv"), index=False
)
logger.info(f"\nSaved → data/greeks_results.csv")

# ── 9. PLOT ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)

# Get stocks that used GARCH fallback (have ATM/OTM structure)
fallback_stocks = results_df[results_df["strike_type"].isin(["ATM", "OTM_5pct", "OTM_10pct"])]["stock"].unique()
chain_stocks = results_df[results_df.get("source", pd.Series()) == "yfinance_chain"]["stock"].unique() if "source" in results_df.columns else []

COLORS_STOCK = {"RELIANCE": "#00d4aa", "HDFCBANK": "#ffd93d", "INFY": "#ff6b6b", "SPY": "#6b9dfc"}
EXPIRY_ORDER = ["30d", "60d", "90d"]
STRIKE_ORDER = ["ATM", "OTM_5pct", "OTM_10pct"]
STRIKE_LABELS = {"ATM": "ATM", "OTM_5pct": "5% OTM", "OTM_10pct": "10% OTM"}

if len(fallback_stocks) >= 2:
    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f0f")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — Call Price vs Strike (fallback stocks, 30d)
    ax1 = fig.add_subplot(gs[0, 0])
    for stock in fallback_stocks:
        sub = results_df[(results_df["stock"] == stock) & (results_df["expiry"] == "30d")]
        if sub.empty:
            continue
        vals = [sub[sub["strike_type"] == s]["call_price"].values[0] for s in STRIKE_ORDER if len(sub[sub["strike_type"] == s]) > 0]
        labels = [STRIKE_LABELS[s] for s in STRIKE_ORDER if len(sub[sub["strike_type"] == s]) > 0]
        ax1.plot(labels, vals, marker="o", label=stock,
                 color=COLORS_STOCK.get(stock, "#888"), linewidth=1.5)
    ax1.set_facecolor("#0f0f0f")
    ax1.set_title("Call Price vs Strike (30-day expiry)", color="white", fontsize=10)
    ax1.set_ylabel("Option Price", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax1.spines[["top","right","left","bottom"]].set_color("#333333")
    ax1.grid(alpha=0.15, color="white")

    # Panel 2 — Delta vs Strike (first stock, all expiries)
    ax2 = fig.add_subplot(gs[0, 1])
    first_stock = fallback_stocks[0]
    for exp in EXPIRY_ORDER:
        sub = results_df[(results_df["stock"] == first_stock) & (results_df["expiry"] == exp)]
        if sub.empty:
            continue
        vals = [sub[sub["strike_type"] == s]["delta_call"].values[0] for s in STRIKE_ORDER if len(sub[sub["strike_type"] == s]) > 0]
        labels = [STRIKE_LABELS[s] for s in STRIKE_ORDER if len(sub[sub["strike_type"] == s]) > 0]
        ax2.plot(labels, vals, marker="o", label=exp, linewidth=1.5)
    ax2.axhline(0.5, color="white", linewidth=0.5, linestyle="--", alpha=0.4, label="Δ=0.5")
    ax2.set_facecolor("#0f0f0f")
    ax2.set_title(f"Delta vs Strike — {first_stock}\n(All Expiries)", color="white", fontsize=10)
    ax2.set_ylabel("Delta", color="white", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax2.spines[["top","right","left","bottom"]].set_color("#333333")
    ax2.grid(alpha=0.15, color="white")

    # Panel 3 — Theta vs Expiry (ATM, all fallback stocks)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(EXPIRY_ORDER))
    w = 0.25
    for i, stock in enumerate(fallback_stocks[:3]):
        sub = results_df[(results_df["stock"] == stock) & (results_df["strike_type"] == "ATM")]
        if sub.empty:
            continue
        vals = [abs(sub[sub["expiry"] == e]["theta_call"].values[0]) for e in EXPIRY_ORDER if len(sub[sub["expiry"] == e]) > 0]
        ax3.bar(x[:len(vals)] + i*w, vals, w, label=stock,
                color=COLORS_STOCK.get(stock, "#888"), alpha=0.85)
    ax3.set_facecolor("#0f0f0f")
    ax3.set_title("Daily Theta Decay — ATM Calls\n(per day)", color="white", fontsize=10)
    ax3.set_ylabel("|Theta| /day", color="white", fontsize=9)
    ax3.set_xticks(x + w)
    ax3.set_xticklabels(EXPIRY_ORDER, color="white")
    ax3.tick_params(colors="white", labelsize=8)
    ax3.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax3.spines[["top","right","left","bottom"]].set_color("#333333")
    ax3.grid(alpha=0.15, color="white", axis="y")

    # Panel 4 — Gamma vs Strike
    ax4 = fig.add_subplot(gs[1, 0])
    for stock in fallback_stocks[:3]:
        sub = results_df[(results_df["stock"] == stock) & (results_df["expiry"] == "30d")]
        if sub.empty:
            continue
        vals = [sub[sub["strike_type"] == s]["gamma"].values[0] for s in STRIKE_ORDER if len(sub[sub["strike_type"] == s]) > 0]
        labels = [STRIKE_LABELS[s] for s in STRIKE_ORDER if len(sub[sub["strike_type"] == s]) > 0]
        ax4.plot(labels, vals, marker="o", label=stock,
                 color=COLORS_STOCK.get(stock, "#888"), linewidth=1.5)
    ax4.set_facecolor("#0f0f0f")
    ax4.set_title("Gamma vs Strike (30-day)\nPeak at ATM", color="white", fontsize=10)
    ax4.set_ylabel("Gamma", color="white", fontsize=9)
    ax4.tick_params(colors="white", labelsize=8)
    ax4.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax4.spines[["top","right","left","bottom"]].set_color("#333333")
    ax4.grid(alpha=0.15, color="white")

    # Panel 5 — Vega vs Expiry
    ax5 = fig.add_subplot(gs[1, 1])
    for stock in fallback_stocks[:3]:
        sub = results_df[(results_df["stock"] == stock) & (results_df["strike_type"] == "ATM")]
        if sub.empty:
            continue
        vals = [sub[sub["expiry"] == e]["vega"].values[0] for e in EXPIRY_ORDER if len(sub[sub["expiry"] == e]) > 0]
        ax5.plot(EXPIRY_ORDER[:len(vals)], vals, marker="o", label=stock,
                 color=COLORS_STOCK.get(stock, "#888"), linewidth=1.5)
    ax5.set_facecolor("#0f0f0f")
    ax5.set_title("Vega vs Expiry — ATM Options\n(per 1% vol move)", color="white", fontsize=10)
    ax5.set_ylabel("Vega", color="white", fontsize=9)
    ax5.tick_params(colors="white", labelsize=8)
    ax5.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax5.spines[["top","right","left","bottom"]].set_color("#333333")
    ax5.grid(alpha=0.15, color="white")

    # Panel 6 — Greeks summary for first stock ATM 30d
    ax6 = fig.add_subplot(gs[1, 2])
    summary_stock = fallback_stocks[0]
    summary_row = results_df[(results_df["stock"] == summary_stock) &
                             (results_df["expiry"] == "30d") &
                             (results_df["strike_type"] == "ATM")]
    if not summary_row.empty:
        r = summary_row.iloc[0]
        greeks = ["Delta", "Gamma×100", "Vega", "|Theta|×10", "Rho"]
        values = [r["delta_call"], r["gamma"]*100, r["vega"], abs(r["theta_call"])*10, r["rho_call"]]
        bar_colors = ["#00d4aa","#ffd93d","#ff6b6b","#a29bfe","#fd79a8"]
        bars = ax6.bar(greeks, values, color=bar_colors, alpha=0.85)
        for bar, val in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    ax6.set_facecolor("#0f0f0f")
    ax6.set_title(f"{summary_stock} ATM Call — All Greeks\n(30-day, scaled)", color="white", fontsize=10)
    ax6.tick_params(colors="white", labelsize=8)
    ax6.spines[["top","right","left","bottom"]].set_color("#333333")
    ax6.grid(alpha=0.15, color="white", axis="y")

    # Source summary in title
    sources = results_df["source"].unique() if "source" in results_df.columns else ["unknown"]
    fig.suptitle(
        f"GARCH Risk Engine — Black-Scholes Greeks\n"
        f"Sources: {', '.join(sources)}  |  RBI r={R_FREE*100:.1f}%",
        color="white", fontsize=13
    )

    plt.savefig(
        os.path.join(BASE_DIR, "assets", "greeks.png"),
        dpi=150, bbox_inches="tight", facecolor="#0f0f0f"
    )
    plt.close()
    logger.info("Saved → assets/greeks.png")
else:
    logger.info("Skipping plot — insufficient fallback data for 6-panel layout")

logger.info(f"\nTotal Greeks computed: {len(results_df)} rows across {results_df['stock'].nunique()} stocks")
logger.info(f"Sources used: {results_df['source'].value_counts().to_dict() if 'source' in results_df.columns else 'N/A'}")

