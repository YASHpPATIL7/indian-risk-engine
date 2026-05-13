"""
SEBI Algo Trading Compliance Module
GARCH Risk Engine

Implements pre-trade risk controls per SEBI Circular
SEBI/HO/MRD2/PoD-2/P/CIR/2024/172

Components:
    AlgoRegistry        — unique Algo-ID tagging per strategy
    pre_trade_check()   — VaR/exposure limits before order placement
    kill_switch()       — auto-halt on systemic crisis detection
    compliance_report() — generates audit-ready summary
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. ALGO-ID REGISTRY ──────────────────────────────────────────────────────
class AlgoRegistry:
    """
    SEBI requires every algo strategy to have a unique Algo-ID
    registered with the broker before deployment.

    This registry maintains a local JSON ledger of all strategies,
    their IDs, creation timestamps, and current status.
    """

    REGISTRY_PATH = os.path.join(BASE_DIR, "data", "algo_registry.json")

    def __init__(self):
        self.strategies = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.REGISTRY_PATH):
            with open(self.REGISTRY_PATH, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.REGISTRY_PATH), exist_ok=True)
        with open(self.REGISTRY_PATH, "w") as f:
            json.dump(self.strategies, f, indent=2, default=str)

    def register(self, strategy_name: str, description: str,
                 max_order_value: float = 1_00_00_000,
                 max_position_pct: float = 0.10) -> str:
        """
        Register a new algo strategy. Returns the Algo-ID.

        Args:
            strategy_name:   Human-readable name (e.g., "GARCH_MeanRevert_V1")
            description:     What the strategy does
            max_order_value:  Max single order in ₹ (default ₹1 Cr)
            max_position_pct: Max single-stock exposure as % of portfolio (default 10%)
        """
        algo_id = f"SEBI-ALGO-{strategy_name.upper().replace(' ', '_')}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        self.strategies[algo_id] = {
            "strategy_name": strategy_name,
            "description": description,
            "algo_id": algo_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "status": "ACTIVE",
            "risk_limits": {
                "max_order_value_inr": max_order_value,
                "max_position_pct": max_position_pct,
                "max_portfolio_var_pct": 0.05,   # 5% daily VaR limit
                "max_garch_sigma": 0.04,          # halt if daily vol > 4%
                "correlation_crisis_threshold": 0.45,
            },
        }
        self._save()
        logger.info(f"Registered algo: {algo_id} | {strategy_name}")
        return algo_id

    def deactivate(self, algo_id: str):
        """Kill switch at strategy level — marks algo as HALTED."""
        if algo_id in self.strategies:
            self.strategies[algo_id]["status"] = "HALTED"
            self.strategies[algo_id]["halted_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            logger.warning(f"HALTED algo: {algo_id}")

    def list_active(self) -> list:
        return [s for s in self.strategies.values() if s["status"] == "ACTIVE"]

    def get_limits(self, algo_id: str) -> dict:
        if algo_id in self.strategies:
            return self.strategies[algo_id]["risk_limits"]
        raise ValueError(f"Algo-ID {algo_id} not found in registry")


# ── 2. PRE-TRADE RISK CHECK ──────────────────────────────────────────────────
class PreTradeCheck:
    """
    Enforces risk limits BEFORE any order is placed.
    Returns PASS/FAIL with reason — order should only proceed on PASS.

    Checks:
        1. Single-stock exposure < max_position_pct
        2. Order value < max_order_value
        3. Portfolio VaR < max_portfolio_var_pct
        4. GARCH sigma < crisis threshold
    """

    def __init__(self, algo_id: str, registry: AlgoRegistry):
        self.algo_id = algo_id
        self.limits = registry.get_limits(algo_id)
        self.status = registry.strategies[algo_id]["status"]

    def check(self, ticker: str, order_value_inr: float,
              current_portfolio_value: float,
              current_position_value: float = 0,
              portfolio_var_pct: float = None,
              garch_sigma: float = None) -> dict:
        """
        Run all pre-trade checks. Returns dict with pass/fail and reasons.

        Args:
            ticker:                  Stock to trade
            order_value_inr:         ₹ value of the proposed order
            current_portfolio_value: Total portfolio value in ₹
            current_position_value:  Current holding in this stock (₹)
            portfolio_var_pct:       Latest portfolio VaR (decimal, e.g., 0.024)
            garch_sigma:             Latest GARCH daily sigma for this stock
        """
        result = {"ticker": ticker, "algo_id": self.algo_id,
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "checks": [], "passed": True}

        # Check 0: Is algo even active?
        if self.status != "ACTIVE":
            result["passed"] = False
            result["checks"].append({
                "check": "algo_status", "passed": False,
                "reason": f"Algo {self.algo_id} is {self.status} — all orders blocked"
            })
            logger.warning(f"PRE-TRADE BLOCKED: {self.algo_id} is {self.status}")
            return result

        # Check 1: Single order value
        max_order = self.limits["max_order_value_inr"]
        c1 = order_value_inr <= max_order
        result["checks"].append({
            "check": "order_value", "passed": c1,
            "value": f"₹{order_value_inr:,.0f}",
            "limit": f"₹{max_order:,.0f}",
            "reason": None if c1 else f"Order ₹{order_value_inr:,.0f} exceeds limit ₹{max_order:,.0f}"
        })

        # Check 2: Single-stock concentration
        new_position = current_position_value + order_value_inr
        position_pct = new_position / current_portfolio_value if current_portfolio_value > 0 else 1.0
        max_pct = self.limits["max_position_pct"]
        c2 = position_pct <= max_pct
        result["checks"].append({
            "check": "concentration", "passed": c2,
            "value": f"{position_pct*100:.1f}%",
            "limit": f"{max_pct*100:.0f}%",
            "reason": None if c2 else f"{ticker} would be {position_pct*100:.1f}% of portfolio (limit {max_pct*100:.0f}%)"
        })

        # Check 3: Portfolio VaR limit
        if portfolio_var_pct is not None:
            max_var = self.limits["max_portfolio_var_pct"]
            c3 = abs(portfolio_var_pct) <= max_var
            result["checks"].append({
                "check": "portfolio_var", "passed": c3,
                "value": f"{portfolio_var_pct*100:.2f}%",
                "limit": f"{max_var*100:.1f}%",
                "reason": None if c3 else f"Portfolio VaR {portfolio_var_pct*100:.2f}% exceeds {max_var*100:.1f}% limit"
            })
            if not c3:
                result["passed"] = False

        # Check 4: GARCH volatility guardrail
        if garch_sigma is not None:
            max_sigma = self.limits["max_garch_sigma"]
            c4 = garch_sigma <= max_sigma
            result["checks"].append({
                "check": "garch_sigma", "passed": c4,
                "value": f"{garch_sigma*100:.2f}%",
                "limit": f"{max_sigma*100:.1f}%",
                "reason": None if c4 else f"GARCH σ={garch_sigma*100:.2f}% exceeds crisis threshold {max_sigma*100:.1f}%"
            })
            if not c4:
                result["passed"] = False

        if not c1 or not c2:
            result["passed"] = False

        # Log result
        status = "✅ PASSED" if result["passed"] else "🚫 BLOCKED"
        logger.info(f"PRE-TRADE {status}: {ticker} | ₹{order_value_inr:,.0f} | {self.algo_id}")
        for c in result["checks"]:
            if not c["passed"]:
                logger.warning(f"  FAILED: {c['check']} — {c['reason']}")

        return result


# ── 3. KILL SWITCH ────────────────────────────────────────────────────────────
def kill_switch(registry: AlgoRegistry,
                mean_correlation: float = None,
                max_garch_sigma: float = None,
                portfolio_drawdown_pct: float = None) -> dict:
    """
    System-wide kill switch. Checks market conditions and halts ALL active
    algos if crisis thresholds are breached.

    Triggers:
        1. Mean pairwise correlation > 0.45 (systemic crisis — all stocks falling together)
        2. Max GARCH sigma across any stock > 5% daily (extreme volatility)
        3. Portfolio drawdown > 8% from peak (capital preservation)

    Returns: dict with triggered/not and which conditions fired.
    """
    triggers = []
    halt = False

    # Correlation crisis
    if mean_correlation is not None and mean_correlation > 0.45:
        triggers.append({
            "trigger": "SYSTEMIC_CORRELATION",
            "value": f"{mean_correlation:.3f}",
            "threshold": "0.450",
            "action": "All stocks moving together — diversification has failed"
        })
        halt = True

    # Extreme volatility
    if max_garch_sigma is not None and max_garch_sigma > 0.05:
        triggers.append({
            "trigger": "EXTREME_VOLATILITY",
            "value": f"{max_garch_sigma*100:.2f}%",
            "threshold": "5.00%",
            "action": "Daily vol exceeds 5% — market in crisis mode"
        })
        halt = True

    # Drawdown circuit breaker
    if portfolio_drawdown_pct is not None and abs(portfolio_drawdown_pct) > 0.08:
        triggers.append({
            "trigger": "DRAWDOWN_BREACH",
            "value": f"{portfolio_drawdown_pct*100:.2f}%",
            "threshold": "8.00%",
            "action": "Portfolio drawdown exceeds capital preservation limit"
        })
        halt = True

    if halt:
        logger.critical("=" * 60)
        logger.critical("🚨 KILL SWITCH ACTIVATED — HALTING ALL ALGOS")
        logger.critical("=" * 60)
        for t in triggers:
            logger.critical(f"  {t['trigger']}: {t['value']} > {t['threshold']}")
            logger.critical(f"    → {t['action']}")

        # Deactivate all active algos
        active = registry.list_active()
        for algo in active:
            registry.deactivate(algo["algo_id"])
            logger.critical(f"  HALTED: {algo['algo_id']}")

    return {
        "kill_switch_triggered": halt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggers": triggers,
        "algos_halted": len(registry.list_active()) if not halt else len(triggers),
    }


# ── 4. COMPLIANCE REPORT ─────────────────────────────────────────────────────
def compliance_report(registry: AlgoRegistry) -> str:
    """
    Generate audit-ready compliance summary.
    Can be exported as part of quarterly SEBI reporting.
    """
    report_lines = [
        "=" * 70,
        "SEBI ALGO TRADING COMPLIANCE REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Reference: SEBI/HO/MRD2/PoD-2/P/CIR/2024/172",
        "=" * 70,
        "",
        f"Total Registered Algos: {len(registry.strategies)}",
        f"Active: {len(registry.list_active())}",
        f"Halted: {len(registry.strategies) - len(registry.list_active())}",
        "",
    ]

    for algo_id, info in registry.strategies.items():
        limits = info["risk_limits"]
        report_lines.extend([
            f"─── {algo_id} ───",
            f"  Strategy:    {info['strategy_name']}",
            f"  Status:      {info['status']}",
            f"  Registered:  {info['registered_at']}",
            f"  Max Order:   ₹{limits['max_order_value_inr']:,.0f}",
            f"  Max Position:{limits['max_position_pct']*100:.0f}%",
            f"  Max VaR:     {limits['max_portfolio_var_pct']*100:.1f}%",
            f"  Vol Guardrail: {limits['max_garch_sigma']*100:.1f}%",
            "",
        ])

    report = "\n".join(report_lines)
    logger.info(report)

    # Save to file
    report_path = os.path.join(BASE_DIR, "data", "sebi_compliance_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved → {report_path}")

    return report


# ── 5. DEMO RUN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Register a strategy
    reg = AlgoRegistry()
    algo_id = reg.register(
        strategy_name="GARCH_MeanRevert_V1",
        description="Mean-reversion strategy gated by GARCH volatility regime",
        max_order_value=50_00_000,   # ₹50 Lakh
        max_position_pct=0.10,       # 10% max per stock
    )

    # Pre-trade check — should PASS
    checker = PreTradeCheck(algo_id, reg)
    result = checker.check(
        ticker="RELIANCE",
        order_value_inr=5_00_000,            # ₹5 Lakh order
        current_portfolio_value=1_00_00_000,  # ₹1 Cr portfolio
        current_position_value=3_00_000,      # Already hold ₹3 Lakh
        portfolio_var_pct=0.024,              # 2.4% VaR — safe
        garch_sigma=0.018,                    # 1.8% daily vol — normal
    )
    logger.info(f"Result: {'PASS' if result['passed'] else 'BLOCKED'}")

    # Pre-trade check — should FAIL (concentration)
    result2 = checker.check(
        ticker="ADANIPORTS",
        order_value_inr=15_00_000,            # ₹15 Lakh — too big
        current_portfolio_value=1_00_00_000,
        current_position_value=0,
        portfolio_var_pct=0.024,
        garch_sigma=0.018,
    )
    logger.info(f"Result: {'PASS' if result2['passed'] else 'BLOCKED'}")

    # Kill switch test — systemic crisis
    ks = kill_switch(
        registry=reg,
        mean_correlation=0.52,     # crisis level
        max_garch_sigma=0.061,     # extreme vol
        portfolio_drawdown_pct=-0.09,
    )
    logger.info(f"Kill switch triggered: {ks['kill_switch_triggered']}")

    # Generate report
    compliance_report(reg)
