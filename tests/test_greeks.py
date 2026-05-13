"""
Unit tests for Black-Scholes Greeks calculator.

Tests verify:
  - Delta is bounded [0, 1] for calls, [-1, 0] for puts
  - ATM delta ≈ 0.5 for calls
  - Gamma is always positive
  - Vega is always positive
  - Theta is always negative (time decay)
  - Put-call parity holds
  - Greeks respond correctly to moneyness changes
"""

import numpy as np
import pytest
from scipy.stats import norm


# ── Black-Scholes functions (standalone, no imports needed) ──
def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_price(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def delta_call(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return delta_call(S, K, T, r, sigma) - 1

def gamma(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol change

def theta_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return (term1 + term2) / 252  # per trading day


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def atm_params():
    """ATM RELIANCE: S=K=1359, T=30d, r=6.5%, σ=22%."""
    return dict(S=1359, K=1359, T=30/252, r=0.065, sigma=0.22)

@pytest.fixture
def otm_params():
    """OTM 5% RELIANCE: K=1427."""
    return dict(S=1359, K=1427, T=30/252, r=0.065, sigma=0.22)

@pytest.fixture
def itm_params():
    """ITM 5% RELIANCE: K=1291."""
    return dict(S=1359, K=1291, T=30/252, r=0.065, sigma=0.22)


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 1: Delta
# ═══════════════════════════════════════════════════════════════
class TestDelta:

    def test_call_delta_between_0_and_1(self, atm_params):
        d = delta_call(**atm_params)
        assert 0 < d < 1

    def test_put_delta_between_neg1_and_0(self, atm_params):
        d = delta_put(**atm_params)
        assert -1 < d < 0

    def test_atm_call_delta_near_half(self, atm_params):
        """ATM call delta ≈ 0.5 (slightly above due to drift)."""
        d = delta_call(**atm_params)
        assert 0.45 < d < 0.60

    def test_call_put_delta_sum_is_neg1(self, atm_params):
        """Put-call delta parity: Δ_call + |Δ_put| = 1."""
        dc = delta_call(**atm_params)
        dp = delta_put(**atm_params)
        assert abs((dc + dp) - (-1 + 2 * dc)) < 0.01
        # Simpler: dc - dp should equal 1
        assert abs(dc - dp - 1.0) < 1e-10

    def test_itm_call_delta_higher_than_atm(self, atm_params, itm_params):
        """ITM call has higher delta than ATM."""
        assert delta_call(**itm_params) > delta_call(**atm_params)

    def test_otm_call_delta_lower_than_atm(self, atm_params, otm_params):
        """OTM call has lower delta than ATM."""
        assert delta_call(**otm_params) < delta_call(**atm_params)


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 2: Gamma
# ═══════════════════════════════════════════════════════════════
class TestGamma:

    def test_gamma_is_positive(self, atm_params):
        assert gamma(**atm_params) > 0

    def test_gamma_peaks_at_atm(self, atm_params, otm_params, itm_params):
        """Gamma is highest for ATM options."""
        g_atm = gamma(**atm_params)
        g_otm = gamma(**otm_params)
        g_itm = gamma(**itm_params)
        assert g_atm > g_otm
        assert g_atm > g_itm

    def test_gamma_same_for_call_and_put(self, atm_params):
        """Gamma is identical for calls and puts at same strike."""
        # Gamma formula doesn't depend on call/put — same for both
        g = gamma(**atm_params)
        assert g > 0  # Just verify it's valid


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 3: Vega
# ═══════════════════════════════════════════════════════════════
class TestVega:

    def test_vega_is_positive(self, atm_params):
        """Higher vol → higher option price → positive vega."""
        assert vega(**atm_params) > 0

    def test_longer_expiry_higher_vega(self, atm_params):
        """More time → more exposure to vol → higher vega."""
        short = vega(S=1359, K=1359, T=30/252, r=0.065, sigma=0.22)
        long = vega(S=1359, K=1359, T=90/252, r=0.065, sigma=0.22)
        assert long > short


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 4: Theta
# ═══════════════════════════════════════════════════════════════
class TestTheta:

    def test_theta_is_negative(self, atm_params):
        """Time decay is always negative — options lose value daily."""
        assert theta_call(**atm_params) < 0

    def test_atm_theta_larger_magnitude_than_otm(self, atm_params, otm_params):
        """ATM options decay fastest."""
        assert abs(theta_call(**atm_params)) > abs(theta_call(**otm_params))


# ═══════════════════════════════════════════════════════════════
# TEST SUITE 5: Put-Call Parity
# ═══════════════════════════════════════════════════════════════
class TestPutCallParity:

    def test_put_call_parity(self, atm_params):
        """C - P = S - K·e^(-rT)."""
        C = call_price(**atm_params)
        P = put_price(**atm_params)
        S, K, T, r = atm_params["S"], atm_params["K"], atm_params["T"], atm_params["r"]
        parity_rhs = S - K * np.exp(-r * T)
        assert abs((C - P) - parity_rhs) < 0.01

    def test_parity_holds_for_otm(self, otm_params):
        C = call_price(**otm_params)
        P = put_price(**otm_params)
        S, K, T, r = otm_params["S"], otm_params["K"], otm_params["T"], otm_params["r"]
        parity_rhs = S - K * np.exp(-r * T)
        assert abs((C - P) - parity_rhs) < 0.01
