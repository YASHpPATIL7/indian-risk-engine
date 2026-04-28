import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sqlalchemy import text
from data_pipeline.db import get_engine

engine = get_engine(echo=False)


# ── Step 1: Load all returns from PostgreSQL ──────────────────
def load_returns_pivot() -> pd.DataFrame:
    """
    Loads log_return for all tickers, pivots into wide format.

    Returns DataFrame:
      rows    = dates
      columns = tickers
      values  = log_return
    """
    print("⏳ Loading returns from PostgreSQL...")

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date, ticker, log_return
            FROM price_data
            WHERE log_return IS NOT NULL
            ORDER BY date ASC
        """))
        rows = result.fetchall()

    df = pd.DataFrame(rows, columns=["date", "ticker", "log_return"])
    df["date"] = pd.to_datetime(df["date"])

    # Pivot: rows=date, columns=ticker, values=log_return
    pivot = df.pivot(index="date", columns="ticker", values="log_return")

    print(f"✅ Loaded: {pivot.shape[0]} dates × {pivot.shape[1]} tickers")
    return pivot


# ── Step 2: Select top 50 tickers by data completeness ────────
def select_top50(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Pick 50 most important tickers — the ones that actually
    drive Indian market correlation during crises.
    Priority: large-caps, financials, Adani group, IT, energy
    """
    # Tier 1 — must include (crisis-relevant stocks)
    priority = [
        # Large-cap financials (most sensitive to systemic risk)
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIGI",
        # Adani group (Hindenburg targets)
        "ADANIPORTS", "ADANIENSOL", "GMRAIRPORT",
        # Large-cap IT (correlated via FII selling)
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM",
        # Large-cap energy/industrial
        "RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA",
        # Large-cap consumer/pharma
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "SUNPHARMA",
        "DRREDDY", "CIPLA", "DIVISLAB",
        # Large-cap auto/industrial
        "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "LT",
        "TITAN", "ASIANPAINT", "ULTRACEMCO",
        # Telecom + others
        "BHARTIARTL", "INDUSINDBK", "GRASIM", "TATACONSUM",
        "HINDALCO", "JSWSTEEL", "TATASTEEL", "VEDL",
        "DMART", "NAUKRI",
    ]

    # Keep only tickers that exist in our DB
    available = [t for t in priority if t in pivot.columns]

    # If we have fewer than 50, fill with most complete remaining
    if len(available) < 50:
        remaining   = [t for t in pivot.columns if t not in available]
        completeness = pivot[remaining].notna().sum().sort_values(ascending=False)
        filler      = completeness.head(50 - len(available)).index.tolist()
        available   += filler

    result = pivot[available[:50]]
    print(f"  Selected tickers sample: {available[:5]}...")
    return result


# ── Step 3: Compute correlation matrix for a date window ──────
def get_correlation_matrix(pivot: pd.DataFrame,
                            end_date: str,
                            window: int = 30) -> pd.DataFrame:
    """
    Returns correlation matrix of 50 stocks
    over `window` days ending at `end_date`
    """
    end   = pd.Timestamp(end_date)
    start = end - pd.tseries.offsets.BDay(window)
    window_data = pivot.loc[start:end].dropna(axis=1, how="any")
    return window_data.corr()


# ── Step 4: Plot heatmap ───────────────────────────────────────
def plot_heatmap(corr_matrix: pd.DataFrame,
                 title: str,
                 filename: str):
    """Plots and saves a correlation heatmap."""

    fig, ax = plt.subplots(figsize=(18, 15))

    # Mask upper triangle — removes redundant mirror
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdYlGn",          # Red=negative, Yellow=zero, Green=positive
        vmin=-1, vmax=1,
        center=0,
        annot=False,             # 50×50 = 2500 cells, too dense for numbers
        linewidths=0.3,
        linecolor="white",
        square=True,
        cbar_kws={
            "label"       : "Pearson Correlation",
            "shrink"      : 0.6,
            "orientation" : "vertical"
        },
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Ticker", fontsize=10)
    ax.set_ylabel("Ticker", fontsize=10)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {filename}")
    plt.show()
    plt.close()


# ── Step 5: Print correlation statistics ──────────────────────
def print_stats(corr_matrix: pd.DataFrame, label: str):
    """Print summary stats for a correlation matrix."""
    # Extract lower triangle values only (no diagonal)
    lower = corr_matrix.where(
        np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
    )
    vals = lower.stack().dropna().values

    print(f"\n── {label} ──────────────────────────────")
    print(f"  Mean correlation   : {vals.mean():.3f}")
    print(f"  Median correlation : {np.median(vals):.3f}")
    print(f"  Min correlation    : {vals.min():.3f}")
    print(f"  Max correlation    : {vals.max():.3f}")
    print(f"  % pairs above 0.7  : {(vals > 0.7).mean()*100:.1f}%")
    print(f"  % pairs above 0.9  : {(vals > 0.9).mean()*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    pivot = load_returns_pivot()
    top50 = select_top50(pivot)

   
# DEBUG — check what the index looks like
    print("\n── DEBUG ──────────────────────────────────────")
    print(f"Index type  : {type(pivot.index[0])}")
    print(f"Index sample: {pivot.index[:3].tolist()}")
    print(f"Index range : {pivot.index.min()} → {pivot.index.max()}")

# Check specific window manually
    test_end   = pd.Timestamp("2023-02-15")
    test_start = test_end - pd.tseries.offsets.BDay(30)
    window_test = pivot.loc[test_start:test_end]
    print(f"\nWindow test : {test_start.date()} → {test_end.date()}")
    print(f"Rows found  : {len(window_test)}")
    print(f"Tickers     : {window_test.notna().any().sum()}")
    print("── END DEBUG ──────────────────────────────────\n")

    print(f"\n📊 Using top 50 tickers: {list(top50.columns[:5])}... etc")

    # ── 3 snapshots: normal, COVID, Hindenburg ─────────────────
    snapshots = [
        {
            "date"     : "2022-06-15",
            "label"    : "Normal Market (Jun 2022)",
            "filename" : "heatmap_normal_2022.png",
        },
        {
            "date"     : "2022-03-15",
            "label"    : "Russia-Ukraine Shock (Mar 2022)",
            "filename" : "heatmap_russia_ukraine_2022.png",
        },
        {
            "date"     : "2023-02-15",
            "label"    : "Adani-Hindenburg Crisis (Feb 2023)",
            "filename" : "heatmap_hindenburg_2023.png",
        },
        {
            "date"     : "2024-06-04",
            "label"    : "Election Results Shock (Jun 2024)",
            "filename" : "heatmap_election_2024.png",
        },
    ]

    # NOTE: Our data starts April 2021 — no COVID March 2020 data
    # Hindenburg Jan 2023 is our key crisis event instead

    results = {}

    for snap in snapshots:
        print(f"\n⏳ Computing correlation for {snap['label']}...")
        try:
            corr = get_correlation_matrix(top50, snap["date"], window=30)

            if corr.shape[0] < 10:
                print(f"⚠️  Not enough tickers for {snap['date']}, skipping")
                continue

            plot_heatmap(
                corr_matrix = corr,
                title       = (f"Nifty 200 — Rolling 30-Day Correlation\n"
                               f"{snap['label']} "
                               f"(window ending {snap['date']})"),
                filename    = snap["filename"]
            )

            print_stats(corr, snap["label"])
            results[snap["label"]] = corr

        except Exception as e:
            print(f"❌ Error for {snap['date']}: {e}")

    # ── Comparison summary ─────────────────────────────────────
    if len(results) >= 2:
        print(f"\n{'═'*55}")
        print(f"  CORRELATION REGIME COMPARISON")
        print(f"{'═'*55}")
        for label, corr in results.items():
            lower = corr.where(
                np.tril(np.ones(corr.shape), k=-1).astype(bool)
            )
            vals = lower.stack().dropna().values
            print(f"\n  {label}")
            print(f"    Mean corr      : {vals.mean():.3f}")
            print(f"    % pairs > 0.7  : {(vals > 0.7).mean()*100:.1f}%")
            print(f"    % pairs > 0.9  : {(vals > 0.9).mean()*100:.1f}%")
        print(f"{'═'*55}")
        print("\n  KEY INSIGHT: During crisis, % pairs > 0.7 spikes sharply")
        print("  Diversification collapses exactly when you need it most.\n")