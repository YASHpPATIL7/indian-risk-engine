import pandas as pd
import numpy as np
import logging
from sqlalchemy import text
from data_pipeline.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s"
)
log = logging.getLogger(__name__)

engine = get_engine(echo=False)


def run_validation():
    log.info("🔍 Starting data validation...\n")
    passed = 0
    failed = 0

    with engine.connect() as conn:

        # ── Check 1: Total row count ───────────────────────────
        total = conn.execute(text(
            "SELECT COUNT(*) FROM price_data"
        )).scalar()
        if total > 200_000:
            log.info(f"✅ Check 1 PASSED │ Total rows: {total:,}")
            passed += 1
        else:
            log.error(f"❌ Check 1 FAILED │ Only {total:,} rows — expected 200K+")
            failed += 1

        # ── Check 2: Ticker count ──────────────────────────────
        tickers = conn.execute(text(
            "SELECT COUNT(DISTINCT ticker) FROM price_data"
        )).scalar()
        if tickers >= 190:
            log.info(f"✅ Check 2 PASSED │ Tickers: {tickers}")
            passed += 1
        else:
            log.error(f"❌ Check 2 FAILED │ Only {tickers} tickers")
            failed += 1

        # ── Check 3: Date range ────────────────────────────────
        result = conn.execute(text(
            "SELECT MIN(date), MAX(date) FROM price_data"
        )).fetchone()
        min_date, max_date = result
        days_covered = (max_date - min_date).days
        if days_covered >= 1800:   # 5 years ≈ 1825 days
            log.info(f"✅ Check 3 PASSED │ Date range: {min_date} → {max_date} ({days_covered} days)")
            passed += 1
        else:
            log.error(f"❌ Check 3 FAILED │ Only {days_covered} days of data")
            failed += 1

        # ── Check 4: Zero NULL close prices ───────────────────
        null_close = conn.execute(text(
            "SELECT COUNT(*) FROM price_data WHERE close IS NULL"
        )).scalar()
        if null_close == 0:
            log.info(f"✅ Check 4 PASSED │ NULL close prices: {null_close}")
            passed += 1
        else:
            log.error(f"❌ Check 4 FAILED │ {null_close} NULL close prices found")
            failed += 1

        # ── Check 5: Zero NULL log_returns ────────────────────
        null_lr = conn.execute(text(
            "SELECT COUNT(*) FROM price_data WHERE log_return IS NULL"
        )).scalar()
        if null_lr == 0:
            log.info(f"✅ Check 5 PASSED │ NULL log_returns: {null_lr}")
            passed += 1
        else:
            log.error(f"❌ Check 5 FAILED │ {null_lr} NULL log_returns — investigate!")
            failed += 1

        # ── Check 6: Rolling vol NULLs within expected range ──
        null_vol = conn.execute(text(
            "SELECT COUNT(*) FROM price_data WHERE rolling_vol_30 IS NULL"
        )).scalar()
        max_allowed = tickers * 29   # first 29 rows per ticker always NULL
        if null_vol <= max_allowed:
            log.info(f"✅ Check 6 PASSED │ NULL rolling_vol: {null_vol:,} (≤ {max_allowed:,} expected)")
            passed += 1
        else:
            log.error(f"❌ Check 6 FAILED │ {null_vol:,} NULL vols — more than expected {max_allowed:,}")
            failed += 1

        # ── Check 7: No duplicate ticker+date pairs ───────────
        dupes = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT ticker, date, COUNT(*) AS cnt
                FROM price_data
                GROUP BY ticker, date
                HAVING COUNT(*) > 1
            ) sub
        """)).scalar()
        if dupes == 0:
            log.info(f"✅ Check 7 PASSED │ Duplicate rows: {dupes}")
            passed += 1
        else:
            log.error(f"❌ Check 7 FAILED │ {dupes} duplicate ticker+date pairs found!")
            failed += 1

        # ── Check 8: Log returns are sensible (no extreme values) ──
        extreme = conn.execute(text("""
            SELECT COUNT(*) FROM price_data
            WHERE ABS(log_return) > 1.0
        """)).scalar()
        if extreme == 0:
            log.info(f"✅ Check 8 PASSED │ Extreme log_returns (>100% daily): {extreme}")
            passed += 1
        else:
            log.warning(f"⚠️  Check 8 WARN  │ {extreme} returns > 100% in a day — check these tickers")
            # Warning not failure — could be genuine events
            passed += 1

        # ── Check 9: Rolling vol is positive where not NULL ───
        neg_vol = conn.execute(text("""
            SELECT COUNT(*) FROM price_data
            WHERE rolling_vol_30 IS NOT NULL
            AND rolling_vol_30 <= 0
        """)).scalar()
        if neg_vol == 0:
            log.info(f"✅ Check 9 PASSED │ Negative/zero volatility values: {neg_vol}")
            passed += 1
        else:
            log.error(f"❌ Check 9 FAILED │ {neg_vol} non-positive volatility values!")
            failed += 1

        # ── Check 10: Sample RELIANCE data spot check ─────────
        sample = conn.execute(text("""
            SELECT date, close, log_return
            FROM price_data
            WHERE ticker = 'RELIANCE'
            ORDER BY date DESC
            LIMIT 1
        """)).fetchone()
        if sample and sample[1] > 0:
            log.info(f"✅ Check 10 PASSED │ RELIANCE latest: date={sample[0]}, close=₹{sample[1]:.2f}, log_return={sample[2]:.5f}")
            passed += 1
        else:
            log.error(f"❌ Check 10 FAILED │ RELIANCE spot check failed")
            failed += 1

    # ── Final Report ──────────────────────────────────────────
    log.info(f"\n{'─'*50}")
    log.info(f"📊 Validation Complete")
    log.info(f"✅ Passed : {passed}/10")
    log.info(f"❌ Failed : {failed}/10")
    if failed == 0:
        log.info("🎯 DATA QUALITY: PRODUCTION READY")
    else:
        log.warning("⚠️  DATA QUALITY: ISSUES FOUND — review above")
    log.info(f"{'─'*50}\n")


if __name__ == "__main__":
    run_validation()