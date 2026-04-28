import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from data_pipeline.db import get_engine
from data_pipeline.models import PriceData, TickerMetadata
from data_pipeline.nifty200_tickers import NIFTY200

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── Step 1: Fetch one ticker from yfinance ─────────────────────
def fetch_ticker(ticker: str) -> pd.DataFrame:
    """
    Downloads 5 years of daily OHLCV for one NSE ticker.
    Returns empty DataFrame on any failure.
    """
    try:
        df = yf.download(
            f"{ticker}.NS",      # NSE suffix — RELIANCE → RELIANCE.NS
            period="5y",         # 5 years back from today
            interval="1d",       # daily bars
            auto_adjust=True,    # gives adj_close automatically
            progress=False       # silence yfinance progress bars
        )

        if df.empty:
            log.warning(f"⚠️  No data returned for {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })

        # adj_close = Close when auto_adjust=True
        df["adj_close"] = df["close"]
        df["ticker"]    = ticker
        df.index.name   = "date"
        df = df.reset_index()
        df["date"]      = pd.to_datetime(df["date"]).dt.date

        # ── Compute log returns ────────────────────────────────
        df = df.sort_values("date").reset_index(drop=True)
        df["log_return"] = np.log(
            df["adj_close"] / df["adj_close"].shift(1)
        )

        # ── Compute 30-day rolling volatility (annualised) ────
        df["rolling_vol_30"] = (
            df["log_return"]
            .rolling(window=30)
            .std() * np.sqrt(252)
        )

        # ── Zero NaN policy ───────────────────────────────────
        # First row always has NaN log_return (no previous price)
        # First 29 rows have NaN rolling_vol — both expected
        # Drop only the very first row (NaN log_return)
        df = df.dropna(subset=["log_return"])

        log.info(f"✅ {ticker:15s} → {len(df):>5} rows fetched")
        return df

    except Exception as e:
        log.error(f"❌ {ticker} failed: {e}")
        return pd.DataFrame()


# ── Step 2: Insert one ticker's rows into PostgreSQL ──────────
def insert_ticker(df: pd.DataFrame, Session) -> int:
    """
    Inserts rows for one ticker. Skips duplicates safely.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    session   = Session()
    inserted  = 0

    try:
        for _, row in df.iterrows():
            record = PriceData(
                ticker         = row["ticker"],
                date           = row["date"],
                open           = float(row["open"])    if pd.notna(row["open"])    else None,
                high           = float(row["high"])    if pd.notna(row["high"])    else None,
                low            = float(row["low"])     if pd.notna(row["low"])     else None,
                close          = float(row["close"]),
                adj_close      = float(row["adj_close"]),
                volume         = int(row["volume"])    if pd.notna(row["volume"])  else None,
                log_return     = float(row["log_return"]) if pd.notna(row["log_return"]) else None,
                rolling_vol_30 = float(row["rolling_vol_30"]) if pd.notna(row["rolling_vol_30"]) else None,
            )
            session.add(record)

        session.commit()
        inserted = len(df)

    except IntegrityError:
        # Duplicate rows — crawler ran before, data already exists
        session.rollback()
        log.warning(f"⚠️  Duplicate skipped for {df['ticker'].iloc[0]}")

    except Exception as e:
        session.rollback()
        log.error(f"❌ Insert failed for {df['ticker'].iloc[0]}: {e}")

    finally:
        session.close()   # always return connection to pool

    return inserted


# ── Step 3: Run crawler with thread pool ──────────────────────
def run_crawler(tickers: list = NIFTY200, max_workers: int = 10):
    """
    Fetches all tickers in parallel threads.
    Inserts each into PostgreSQL.
    """
    engine  = get_engine(echo=False)   # echo=False — no SQL spam
    Session = sessionmaker(bind=engine)

    log.info(f"🚀 Starting crawler: {len(tickers)} tickers │ {max_workers} threads")
    start   = datetime.now()

    total_rows  = 0
    failed      = []

    # ── ThreadPoolExecutor: fetch all tickers in parallel ─────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch jobs simultaneously
        future_to_ticker = {
            executor.submit(fetch_ticker, t): t
            for t in tickers
        }

        # Process results as they complete (not in order)
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            df     = future.result()

            if df.empty:
                failed.append(ticker)
                continue

            rows_inserted = insert_ticker(df, Session)
            total_rows   += rows_inserted

    elapsed = (datetime.now() - start).seconds
    log.info(f"\n{'─'*50}")
    log.info(f"✅ Done in {elapsed}s")
    log.info(f"📊 Total rows inserted : {total_rows:,}")
    log.info(f"❌ Failed tickers      : {len(failed)} → {failed}")
    log.info(f"{'─'*50}\n")


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    run_crawler()