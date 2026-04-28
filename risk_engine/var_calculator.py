import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import text
from data_pipeline.db import get_engine


class VaRCalculator:
    """
    Computes Value at Risk (VaR) and Conditional VaR (Expected Shortfall)
    for a single stock using historical log returns from PostgreSQL.
    """

    def __init__(self, ticker: str, confidence: float = 0.95):
        """
        ticker     : NSE symbol e.g. "RELIANCE"
        confidence : 0.95 = 95% confidence level (industry standard)
        """
        self.ticker     = ticker
        self.confidence = confidence
        self.returns    = self._load_returns()

    def _load_returns(self) -> np.ndarray:
        """Fetch log returns from PostgreSQL for this ticker."""
        engine = get_engine(echo=False)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT log_return
                FROM price_data
                WHERE ticker = :ticker
                AND   log_return IS NOT NULL
                ORDER BY date ASC
            """), {"ticker": self.ticker})
            returns = np.array([row[0] for row in result.fetchall()])

        if len(returns) == 0:
            raise ValueError(f"No return data found for ticker: {self.ticker}")

        return returns

    # ── Method 1: Historical Simulation VaR ───────────────────
    def var_historical(self) -> float:
        """
        Sort all historical returns, take the (1-confidence) percentile.
        No assumptions about distribution — uses actual data.

        Returns negative float e.g. -0.018 = -1.8% loss
        """
        return float(np.percentile(self.returns, (1 - self.confidence) * 100))

    # ── Method 2: Parametric VaR (Gaussian) ───────────────────
    def var_parametric(self) -> float:
        """
        Assumes returns are normally distributed.
        Uses mean and std of historical returns.

        Formula: VaR = μ - z × σ
        where z = norm.ppf(1 - confidence)

        Returns negative float e.g. -0.019 = -1.9% loss
        """
        mu    = self.returns.mean()
        sigma = self.returns.std()
        z     = norm.ppf(1 - self.confidence)   # e.g. -1.645 for 95%
        return float(mu + z * sigma)

    # ── Method 3: CVaR / Expected Shortfall ───────────────────
    def cvar(self) -> float:
        """
        Average of all returns BELOW the VaR threshold.
        Always more negative than VaR — captures true tail risk.

        Formula: E[X | X < VaR]

        Returns negative float e.g. -0.029 = -2.9% expected loss
                                              on worst 5% of days
        """
        var_threshold = self.var_historical()
        tail_losses   = self.returns[self.returns < var_threshold]

        if len(tail_losses) == 0:
            return var_threshold

        return float(tail_losses.mean())

    # ── Method 4: Parametric CVaR ─────────────────────────────
    def cvar_parametric(self) -> float:
        """
        Parametric CVaR assuming normal distribution.
        Used when you want an analytical formula (e.g. inside GARCH engine).

        Formula: CVaR = μ - σ × φ(z) / (1 - confidence)
        where φ = normal PDF, z = VaR z-score
        """
        mu    = self.returns.mean()
        sigma = self.returns.std()
        alpha = 1 - self.confidence
        z     = norm.ppf(alpha)
        return float(mu - sigma * (norm.pdf(z) / alpha))

    # ── Method 5: Rolling VaR (time series) ───────────────────
    def rolling_var(self, window: int = 252) -> pd.Series:
        """
        Compute VaR over a rolling window.
        Shows how risk changed over time — spikes during COVID, Hindenburg.

        window=252 → 1 year of trading days
        """
        returns_series = pd.Series(self.returns)
        return returns_series.rolling(window).apply(
            lambda x: np.percentile(x, (1 - self.confidence) * 100),
            raw=True
        )

    # ── Summary Report ─────────────────────────────────────────
    def summary(self) -> dict:
        """Returns all risk metrics as a dict — easy to pass to dashboard."""
        var_h = self.var_historical()
        var_p = self.var_parametric()
        cvar  = self.cvar()
        cvar_p = self.cvar_parametric()

        return {
            "ticker"             : self.ticker,
            "confidence"         : f"{self.confidence*100:.0f}%",
            "observations"       : len(self.returns),
            "var_historical"     : round(var_h,  5),
            "var_parametric"     : round(var_p,  5),
            "cvar_historical"    : round(cvar,   5),
            "cvar_parametric"    : round(cvar_p, 5),
            "var_pct"            : f"{var_h*100:.2f}%",
            "cvar_pct"           : f"{cvar*100:.2f}%",
            "cvar_vs_var_ratio"  : round(cvar / var_h, 3),  # always > 1
        }


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    tickers = ["RELIANCE", "HDFCBANK", "TCS", "INFY", "ADANIPORTS"]

    print(f"\n{'─'*65}")
    print(f"{'Ticker':<12} {'VaR 95%':>10} {'CVaR 95%':>10} "
          f"{'CVaR/VaR':>10} {'N':>6}")
    print(f"{'─'*65}")

    for t in tickers:
        try:
            calc = VaRCalculator(ticker=t, confidence=0.95)
            s    = calc.summary()
            print(f"{s['ticker']:<12} "
                  f"{s['var_pct']:>10} "
                  f"{s['cvar_pct']:>10} "
                  f"{s['cvar_vs_var_ratio']:>10} "
                  f"{s['observations']:>6}")
        except ValueError as e:
            print(f"{t:<12} ERROR: {e}")

    print(f"{'─'*65}\n")

    # Deep dive on RELIANCE
    print("── RELIANCE Full Risk Report ──────────────────────────────")
    calc = VaRCalculator("RELIANCE", confidence=0.95)
    for k, v in calc.summary().items():
        print(f"  {k:<22} : {v}")
    print()

    # Quick rolling VaR print for RELIANCE
calc = VaRCalculator("RELIANCE", confidence=0.95)
rolling = calc.rolling_var(window=252)

returns_df = pd.Series(calc.returns)

# Show VaR at key dates by index position
print("\n── Rolling VaR snapshots (252-day window) for Reliance Stock ──")
print(f"  Earliest available : {rolling.dropna().iloc[0]*100:.2f}%")
print(f"  Median (5yr)       : {rolling.dropna().median()*100:.2f}%")
print(f"  Worst ever         : {rolling.dropna().min()*100:.2f}%")
print(f"  Best ever (calmest): {rolling.dropna().max()*100:.2f}%")
print(f"  Current (today)    : {rolling.dropna().iloc[-1]*100:.2f}%")