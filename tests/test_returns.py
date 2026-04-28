import pytest
import numpy as np
import pandas as pd


# ── Functions being tested (pure math — no DB needed) ─────────
def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def rolling_volatility(log_returns: pd.Series, window: int = 30) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(252)


def annualised_vol(log_returns: pd.Series) -> float:
    return float(log_returns.std() * np.sqrt(252))


# ── Tests ──────────────────────────────────────────────────────

class TestLogReturns:

    def test_correct_formula(self):
        """ln(110/100) = 0.09531"""
        prices = pd.Series([100.0, 110.0])
        lr = compute_log_returns(prices)
        assert abs(lr.iloc[1] - np.log(110 / 100)) < 1e-10

    def test_first_row_is_nan(self):
        """First row always NaN — no previous price"""
        prices = pd.Series([100, 101, 102, 103])
        lr = compute_log_returns(prices)
        assert pd.isna(lr.iloc[0])

    def test_no_nan_after_first_row(self):
        """Zero NaN policy — no NaN after row 0"""
        prices = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])
        lr = compute_log_returns(prices)
        assert lr.iloc[1:].isna().sum() == 0

    def test_negative_return_on_price_drop(self):
        """Price drop = negative log return"""
        prices = pd.Series([100.0, 90.0])
        lr = compute_log_returns(prices)
        assert lr.iloc[1] < 0

    def test_zero_return_on_unchanged_price(self):
        """Same price two days = exactly 0.0 return"""
        prices = pd.Series([100.0, 100.0])
        lr = compute_log_returns(prices)
        assert lr.iloc[1] == 0.0

    def test_symmetry(self):
        """log return up then down should nearly cancel"""
        prices = pd.Series([100.0, 110.0, 100.0])
        lr = compute_log_returns(prices)
        assert abs(lr.iloc[1] + lr.iloc[2]) < 1e-10


class TestRollingVolatility:

    def test_output_length_matches_input(self):
        """Output same length as input"""
        lr = pd.Series(np.random.randn(252) * 0.01)
        vol = rolling_volatility(lr, window=30)
        assert len(vol) == 252

    def test_first_window_minus_one_rows_are_nan(self):
        """First 29 rows NaN for window=30"""
        lr = pd.Series(np.random.randn(252) * 0.01)
        vol = rolling_volatility(lr, window=30)
        assert vol.iloc[:29].isna().all()

    def test_no_nan_after_window(self):
        """Zero NaN policy — no NaN after window fills"""
        lr = pd.Series(np.random.randn(252) * 0.01)
        vol = rolling_volatility(lr, window=30)
        assert vol.iloc[30:].isna().sum() == 0

    def test_volatility_is_positive(self):
        """Volatility always positive"""
        lr = pd.Series(np.random.randn(252) * 0.01)
        vol = rolling_volatility(lr, window=30)
        assert (vol.dropna() > 0).all()

    def test_higher_variance_gives_higher_vol(self):
        """More volatile series → higher rolling vol"""
        low_vol  = pd.Series(np.random.randn(252) * 0.005)
        high_vol = pd.Series(np.random.randn(252) * 0.05)
        assert (rolling_volatility(high_vol, 30).mean() >
                rolling_volatility(low_vol,  30).mean())

    def test_constant_prices_give_zero_vol(self):
        """Zero variance = zero volatility"""
        lr  = pd.Series([0.0] * 100)
        vol = rolling_volatility(lr, window=30)
        assert (vol.dropna() == 0.0).all()


class TestAnnualisedVol:

    def test_annualisation_factor(self):
        """Daily vol × √252 = annual vol"""
        np.random.seed(0)
        daily_std = 0.01
    # Use a series with KNOWN std, not constant values
        lr  = pd.Series(np.random.normal(loc=0, scale=daily_std, size=10000))
        vol = annualised_vol(lr)
        expected = lr.std() * np.sqrt(252)
        assert abs(vol - expected) < 1e-10

    def test_reasonable_range_for_indian_stocks(self):
        """Annual vol for Indian large-caps: 15%–60%"""
        np.random.seed(42)
        lr  = pd.Series(np.random.randn(252) * 0.012)
        vol = annualised_vol(lr)
        assert 0.10 < vol < 0.80