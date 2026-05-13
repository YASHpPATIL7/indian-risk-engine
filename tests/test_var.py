"""
Unit tests for VaR / CVaR calculations.

Tests use synthetic returns (no DB needed) to verify:
  - Historical VaR picks the correct percentile
  - Parametric VaR matches μ - z·σ formula
  - CVaR is always worse (more negative) than VaR
  - CVaR/VaR ratio > 1 for fat-tailed distributions
  - Rolling VaR produces correct-length output
  - Edge cases: constant returns, single large loss
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


# ── Standalone VaR functions (no DB dependency) ──────────────
def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    return float(np.percentile(returns, (1 - confidence) * 100))

def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return float(mu + z * sigma)

def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    threshold = var_historical(returns, confidence)
    tail = returns[returns < threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold

def cvar_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std()
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    return float(mu - sigma * (norm.pdf(z) / alpha))

def rolling_var(returns: np.ndarray, window: int = 252, confidence: float = 0.95) -> pd.Series:
    return pd.Series(returns).rolling(window).apply(
        lambda x: np.percentile(x, (1 - confidence) * 100), raw=True
    )


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def normal_returns():
    """1000 normally distributed returns, μ=0.0005, σ=0.02 (typical stock)."""
    np.random.seed(42)
    return np.random.normal(0.0005, 0.02, 1000)

@pytest.fixture
def fat_tail_returns():
    """Returns with injected crash days (fat tails)."""
    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.02, 1000)
    # Inject 5 crash days (-10% to -20%)
    rets[100] = -0.10
    rets[200] = -0.15
    rets[300] = -0.12
    rets[400] = -0.19
    rets[500] = -0.08
    return rets

@pytest.fixture
def constant_returns():
    """Zero volatility — all returns identical."""
    return np.full(500, 0.001)


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 1: Historical VaR
# ═══════════════════════════════════════════════════════════════
class TestHistoricalVaR:

    def test_returns_negative_value(self, normal_returns):
        """VaR should be negative (it's a loss threshold)."""
        var = var_historical(normal_returns, 0.95)
        assert var < 0

    def test_95_var_less_negative_than_99(self, normal_returns):
        """99% VaR should be more negative (stricter) than 95% VaR."""
        var_95 = var_historical(normal_returns, 0.95)
        var_99 = var_historical(normal_returns, 0.99)
        assert var_99 < var_95

    def test_approximates_theoretical_for_normal(self, normal_returns):
        """For normal returns, historical VaR ≈ μ - 1.645σ (within tolerance)."""
        var = var_historical(normal_returns, 0.95)
        mu, sigma = normal_returns.mean(), normal_returns.std()
        expected = mu + norm.ppf(0.05) * sigma
        assert abs(var - expected) < 0.005  # within 50 bps

    def test_constant_returns_var_equals_return(self, constant_returns):
        """If all returns are identical, VaR = that return value."""
        var = var_historical(constant_returns, 0.95)
        assert abs(var - 0.001) < 1e-10


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 2: Parametric VaR
# ═══════════════════════════════════════════════════════════════
class TestParametricVaR:

    def test_formula_matches_manual_calc(self, normal_returns):
        """VaR_p = μ + z·σ where z = norm.ppf(0.05)."""
        var = var_parametric(normal_returns, 0.95)
        mu, sigma = normal_returns.mean(), normal_returns.std()
        expected = mu + norm.ppf(0.05) * sigma
        assert abs(var - expected) < 1e-10

    def test_higher_sigma_gives_worse_var(self):
        """More volatile stock → more negative VaR."""
        np.random.seed(42)
        calm = np.random.normal(0, 0.01, 500)   # σ = 1%
        wild = np.random.normal(0, 0.04, 500)   # σ = 4%
        assert var_parametric(wild) < var_parametric(calm)

    def test_returns_negative(self, normal_returns):
        var = var_parametric(normal_returns, 0.95)
        assert var < 0


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 3: CVaR (Expected Shortfall)
# ═══════════════════════════════════════════════════════════════
class TestCVaR:

    def test_cvar_worse_than_var(self, normal_returns):
        """CVaR is always more negative than VaR (it's the average beyond VaR)."""
        var = var_historical(normal_returns, 0.95)
        cvar = cvar_historical(normal_returns, 0.95)
        assert cvar < var

    def test_cvar_var_ratio_greater_than_1(self, normal_returns):
        """CVaR/VaR ratio > 1 (tails are worse than the threshold)."""
        var = var_historical(normal_returns, 0.95)
        cvar = cvar_historical(normal_returns, 0.95)
        ratio = cvar / var  # both negative, ratio > 1
        assert ratio > 1.0

    def test_fat_tails_increase_ratio(self, normal_returns, fat_tail_returns):
        """Injected crash days should push CVaR/VaR ratio higher."""
        ratio_normal = cvar_historical(normal_returns) / var_historical(normal_returns)
        ratio_fat = cvar_historical(fat_tail_returns) / var_historical(fat_tail_returns)
        assert ratio_fat > ratio_normal

    def test_parametric_cvar_worse_than_parametric_var(self, normal_returns):
        var = var_parametric(normal_returns, 0.95)
        cvar = cvar_parametric(normal_returns, 0.95)
        assert cvar < var


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 4: Rolling VaR
# ═══════════════════════════════════════════════════════════════
class TestRollingVaR:

    def test_output_length_matches_input(self, normal_returns):
        rv = rolling_var(normal_returns, window=252)
        assert len(rv) == len(normal_returns)

    def test_first_window_minus_one_are_nan(self, normal_returns):
        rv = rolling_var(normal_returns, window=252)
        assert rv.iloc[:251].isna().all()

    def test_values_after_window_are_negative(self, normal_returns):
        rv = rolling_var(normal_returns, window=252)
        valid = rv.dropna()
        assert (valid < 0).all()

    def test_crash_spike_appears_in_rolling(self, normal_returns, fat_tail_returns):
        """Crash days make rolling VaR worse compared to clean returns."""
        # Use 90% VaR so each crash day is a larger fraction of the tail
        rv_clean = rolling_var(normal_returns, window=50, confidence=0.90).dropna()
        rv_crash = rolling_var(fat_tail_returns, window=50, confidence=0.90).dropna()
        assert rv_crash.min() < rv_clean.min()
