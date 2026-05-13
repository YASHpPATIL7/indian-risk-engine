import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import text
import sys, os
import logging

# ═══════════════════════════════════════════════════════════════
# ROOT LOGGER — All module logs flow here (SEBI Audit Trail)
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_engine_audit.log', mode='a'),
    ]
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db import get_engine
from risk_engine.var_calculator import VaRCalculator

st.set_page_config(
    page_title="GARCH Risk Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #080b12; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2433;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
}
.kpi-label {
    font-size: 11px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
}
.kpi-value { font-size: 24px; font-weight: 700; color: #f9fafb; line-height: 1; }
.kpi-sub   { font-size: 12px; margin-top: 4px; }
.pos { color: #10b981; } .neg { color: #ef4444; } .muted { color: #6b7280; }
.regime-badge {
    display: inline-block; padding: 8px 20px; border-radius: 20px;
    font-size: 13px; font-weight: 600; letter-spacing: 0.03em; margin-top: 12px;
}
.r-sys  { background:#450a0a; color:#fca5a5; border:1px solid #ef4444; }
.r-clus { background:#451a03; color:#fcd34d; border:1px solid #f59e0b; }
.r-flt  { background:#0c1a3a; color:#93c5fd; border:1px solid #3b82f6; }
.r-nrm  { background:#052e16; color:#86efac; border:1px solid #22c55e; }
.sec-hdr {
    font-size: 11px; font-weight: 600; color: #4b5563;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 16px 0 8px 0; padding-bottom: 6px; border-bottom: 1px solid #1e2433;
}
.risk-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #1a1f2e; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

engine = get_engine(echo=False)

PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#080b12",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#9ca3af", size=12),
    margin=dict(l=0, r=0, t=36, b=0),
    xaxis=dict(gridcolor="#1a1f2e", linecolor="#1e2433"),
    yaxis=dict(gridcolor="#1a1f2e", linecolor="#1e2433"),
)

PRIORITY = [
    "HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN",
    "BAJFINANCE","BAJAJFINSV","RELIANCE","TCS","INFY",
    "HCLTECH","WIPRO","TECHM","LTIM","HINDUNILVR",
    "ITC","NESTLEIND","SUNPHARMA","DRREDDY","CIPLA",
    "MARUTI","BAJAJ-AUTO","EICHERMOT","LT","TITAN",
    "ASIANPAINT","ULTRACEMCO","BHARTIARTL","NTPC","POWERGRID",
    "COALINDIA","ONGC","BPCL","ADANIPORTS","ADANIENSOL",
    "INDUSINDBK","GRASIM","TATACONSUM","HINDALCO","JSWSTEEL",
    "TATASTEEL","VEDL","DMART","NAUKRI","HDFCLIFE",
    "SBILIFE","ICICIGI","DIVISLAB","GMRAIRPORT","BRITANNIA"
]

# ── helpers ───────────────────────────────────────────────────
def add_event_line(fig, date_str, label, row=None, col=None):
    """add_vline replacement that works with datetime axes on all plotly versions"""
    kwargs = dict(
        x0=date_str, x1=date_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#4b5563", width=1.2, dash="dot"),
        layer="below"
    )
    if row is not None:
        kwargs["xref"] = f"x{row}" if row > 1 else "x"
    fig.add_shape(**kwargs)
    ann_kwargs = dict(
        x=date_str, y=0.98, xref=kwargs["xref"], yref="paper",
        text=label, showarrow=False,
        font=dict(size=10, color="#9ca3af"),
        textangle=-90, xanchor="right"
    )
    fig.add_annotation(**ann_kwargs)

# ── cached loaders ────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_tickers():
    with engine.connect() as c:
        return [r[0] for r in c.execute(
            text("SELECT DISTINCT ticker FROM price_data ORDER BY ticker")
        ).fetchall()]

@st.cache_data(ttl=300)
def load_price(ticker):
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT date, close, adj_close, volume, log_return, rolling_vol_30
            FROM price_data WHERE ticker=:t ORDER BY date
        """), {"t": ticker}).fetchall()
    df = pd.DataFrame(rows, columns=["date","close","adj_close",
                                      "volume","log_return","rolling_vol_30"])
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_pivot():
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT date, ticker, log_return FROM price_data
            WHERE log_return IS NOT NULL ORDER BY date
        """)).fetchall()
    df = pd.DataFrame(rows, columns=["date","ticker","log_return"])
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="log_return")

@st.cache_data(ttl=300)
def get_risk(ticker, conf):
    calc = VaRCalculator(ticker=ticker, confidence=conf)
    return calc.summary(), calc.rolling_var(window=252), calc.returns

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 20px'>
        <div style='font-size:34px'>📉</div>
        <div style='font-size:19px;font-weight:700;color:#f9fafb;letter-spacing:.02em'>
            GARCH Risk Engine</div>
        <div style='font-size:10px;color:#4b5563;margin-top:4px;
                    text-transform:uppercase;letter-spacing:.1em'>
            NSE Large-Caps · Portfolio Risk Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Configuration</div>', unsafe_allow_html=True)
    tickers = load_tickers()
    ticker  = st.selectbox("Stock", tickers,
                           index=tickers.index("RELIANCE") if "RELIANCE" in tickers else 0)
    conf    = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, format="%.2f")

    st.markdown('<div class="sec-hdr">Universe</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    ca.metric("VaR Stocks", f"{len(tickers)}"); ca.metric("Rows", "262K")
    cb.metric("Risk Model", "14");   cb.metric("From", "Apr 2021")

    st.markdown('<div class="sec-hdr">Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px;color:#6b7280;line-height:2'>
    • Historical VaR<br>• Parametric VaR<br>• CVaR / Expected Shortfall<br>
    • Rolling 252-day VaR<br>• GARCH(1,1) + DCC<br>
    • SHAP Variance Attribution<br>• Correlation Regimes
    </div>""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────
df = load_price(ticker)
summary, rv_series, returns = get_risk(ticker, conf)

last_ret   = df["log_return"].iloc[-1]
last_vol   = df["rolling_vol_30"].dropna().iloc[-1]
last_price = df["close"].iloc[-1]

# ── header ────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:baseline;justify-content:space-between;
            padding-bottom:20px'>
    <div>
        <span style='font-size:28px;font-weight:700;color:#f9fafb'>{ticker}</span>
        <span style='font-size:13px;color:#4b5563;margin-left:10px'>NSE</span>
    </div>
    <div style='font-size:12px;color:#374151'>{conf*100:.0f}% Confidence
        · Apr 2021 → Apr 2026</div>
</div>""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
ret_cls  = "pos" if last_ret >= 0 else "neg"
ret_sign = "▲" if last_ret >= 0 else "▼"
ratio    = summary["cvar_vs_var_ratio"]
ratio_cls = "neg" if float(str(ratio)) > 1.7 else "pos"

kpis = [
    (k1,"Price",   f"₹{last_price:,.2f}",
     f'<span class="{ret_cls}">{ret_sign} {abs(last_ret)*100:.2f}% today</span>'),
    (k2,f"VaR {conf*100:.0f}%", summary["var_pct"],
     '<span class="neg">Historical</span>'),
    (k3,f"CVaR {conf*100:.0f}%", summary["cvar_pct"],
     '<span class="neg">Exp. Shortfall</span>'),
    (k4,"CVaR/VaR", str(ratio),
     f'<span class="{ratio_cls}">Tail Risk Ratio</span>'),
    (k5,"30d Vol",  f"{last_vol*100:.1f}%",
     '<span class="muted">Annualised</span>'),
    (k6,"Days",     f"{summary['observations']:,}",
     '<span class="muted">Trading Days</span>'),
]
for col,lbl,val,sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "  📈  Price & Volatility  ",
    "  📊  Return Distribution  ",
    "  📉  Rolling VaR  ",
    "  🔥  Correlation Regimes  ",
    "  🧩  SHAP Attribution  ",
    "  ⚡  Greeks Engine  ",
])

# ─── TAB 1 ────────────────────────────────────────────────────
with tab1:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[
            f"{ticker} — Adjusted Close (₹)",
            "30-Day Rolling Volatility % (Annualised)",
            "Daily Log Returns %"
        ],
        vertical_spacing=0.05
    )
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["adj_close"], mode="lines",
        line=dict(color="#3b82f6", width=1.8), name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rolling_vol_30"]*100, mode="lines",
        line=dict(color="#f59e0b", width=1.4),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.07)", name="Vol"
    ), row=2, col=1)
    ret_cols = ["#ef4444" if r < 0 else "#10b981"
                for r in df["log_return"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["log_return"]*100,
        marker_color=ret_cols, opacity=0.8, name="Returns"
    ), row=3, col=1)
    fig.update_layout(height=680, showlegend=False, **PLOT_THEME)
    for r in [1,2,3]:
        fig.update_xaxes(gridcolor="#1a1f2e", row=r, col=1)
        fig.update_yaxes(gridcolor="#1a1f2e", row=r, col=1)
    st.plotly_chart(fig, width="stretch")

# ─── TAB 2 ────────────────────────────────────────────────────
with tab2:
    left, right = st.columns([3, 1])
    with left:
        var_h  = summary["var_historical"]
        cvar_h = summary["cvar_historical"]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=returns*100, nbinsx=80, name="All Returns",
            marker_color="#3b82f6", opacity=0.55
        ))
        tail = returns[returns < var_h] * 100
        fig2.add_trace(go.Histogram(
            x=tail, nbinsx=20,
            name=f"Tail ({(1-conf)*100:.0f}%)",
            marker_color="#ef4444", opacity=0.9
        ))
        # Use add_shape+add_annotation instead of add_vline
        for x_val, color, label in [
            (var_h*100,  "#f59e0b", f"VaR {var_h*100:.2f}%"),
            (cvar_h*100, "#ef4444", f"CVaR {cvar_h*100:.2f}%"),
        ]:
            fig2.add_shape(type="line",
                x0=x_val, x1=x_val, y0=0, y1=1, yref="paper",
                line=dict(color=color, width=2, dash="dash"))
            fig2.add_annotation(
                x=x_val, y=0.95, yref="paper",
                text=label, showarrow=False,
                font=dict(color=color, size=11), xanchor="left"
            )
        fig2.update_layout(
            height=420, barmode="overlay", showlegend=True,
            xaxis_title="Daily Return (%)", yaxis_title="Frequency",
            legend=dict(bgcolor="#0d1117", bordercolor="#1e2433"),
            **PLOT_THEME
        )
        st.plotly_chart(fig2, width="stretch")
    with right:
        st.markdown('<div class="sec-hdr">Risk Metrics</div>', unsafe_allow_html=True)
        rows_data = [
            ("VaR Historical",   summary["var_pct"]),
            ("VaR Parametric",   f"{summary['var_parametric']*100:.2f}%"),
            ("CVaR Historical",  summary["cvar_pct"]),
            ("CVaR Parametric",  f"{summary['cvar_parametric']*100:.2f}%"),
            ("CVaR/VaR Ratio",   str(summary["cvar_vs_var_ratio"])),
            ("Confidence",       summary["confidence"]),
            ("Observations",     str(summary["observations"])),
        ]
        for lbl, val in rows_data:
            st.markdown(f"""
            <div class="risk-row">
                <span style='color:#9ca3af'>{lbl}</span>
                <span style='color:#f9fafb;font-weight:600'>{val}</span>
            </div>""", unsafe_allow_html=True)

# ─── TAB 3 ────────────────────────────────────────────────────
with tab3:
    rolling_s = pd.Series(
        rv_series.values,
        index=df["date"].iloc[:len(rv_series)]
    ).dropna()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=rolling_s.index, y=rolling_s.values*100,
        mode="lines", name="Rolling VaR",
        line=dict(color="#ef4444", width=1.8),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.09)"
    ))

    events = {
        "Russia-Ukraine": "2022-02-24",
        "Hindenburg":     "2023-01-24",
        "Election":       "2024-06-04",
    }
    for label, date_str in events.items():
        ts = pd.Timestamp(date_str)
        if rolling_s.index.min() <= ts <= rolling_s.index.max():
            add_event_line(fig3, date_str, label)

    fig3.update_layout(
        height=440,
        xaxis_title="Date", yaxis_title="Rolling VaR (%)",
        **PLOT_THEME
    )
    st.plotly_chart(fig3, width="stretch")

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Current VaR",  f"{rolling_s.iloc[-1]*100:.2f}%")
    s2.metric("Worst VaR",    f"{rolling_s.min()*100:.2f}%")
    s3.metric("Calmest VaR",  f"{rolling_s.max()*100:.2f}%")
    s4.metric("Median VaR",   f"{rolling_s.median()*100:.2f}%")

# ─── TAB 4 ────────────────────────────────────────────────────
with tab4:
    REGIMES = {
        "🟢  Normal · Jun 2022"                   : "2022-06-15",
        "🔴  Russia-Ukraine Shock · Mar 2022"     : "2022-03-15",
        "🔵  Hindenburg Sector Flight · Feb 2023" : "2023-02-15",
        "🟡  Election Cluster Shock · Jun 2024"   : "2024-06-04",
    }
    selected = st.radio("Regime", list(REGIMES.keys()),
                        horizontal=True, label_visibility="collapsed")
    sel_date = REGIMES[selected]

    with st.spinner("Computing correlation matrix..."):
        pivot     = load_pivot()
        available = [t for t in PRIORITY if t in pivot.columns]
        end_ts    = pd.Timestamp(sel_date)
        start_ts  = end_ts - pd.tseries.offsets.BDay(30)
        win       = pivot[available].loc[start_ts:end_ts].ffill().dropna(axis=1)
        corr      = win.corr()

        fig4, ax = plt.subplots(figsize=(16, 13))
        fig4.patch.set_facecolor("#080b12")
        ax.set_facecolor("#080b12")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = sns.diverging_palette(10, 130, s=80, l=55, n=256, as_cmap=True)
        sns.heatmap(
            corr, mask=mask, cmap=cmap,
            vmin=-1, vmax=1, center=0,
            annot=False, linewidths=0.2, linecolor="#1a1f2e",
            square=True,
            cbar_kws={"shrink": 0.55, "label": "Pearson Correlation"},
            ax=ax
        )
        ax.set_title(
            f"30-Day Rolling Correlation  ·  {selected.split('·')[1].strip()}\n"
            f"{start_ts.date()} → {end_ts.date()}  ·  {corr.shape[0]} stocks",
            color="#e5e7eb", fontsize=12, pad=14
        )
        ax.tick_params(colors="#6b7280", labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=90, color="#9ca3af")
        plt.setp(ax.get_yticklabels(), rotation=0,  color="#9ca3af")
        ax.figure.axes[-1].tick_params(colors="#9ca3af")
        ax.figure.axes[-1].yaxis.label.set_color("#9ca3af")
        st.pyplot(fig4)
        plt.close()

    lower = corr.where(np.tril(np.ones_like(corr, dtype=bool), k=-1))
    vals  = lower.stack().dropna().values
    mean_c = vals.mean()
    pct07  = (vals > 0.7).mean() * 100
    pct09  = (vals > 0.9).mean() * 100

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Mean Correlation", f"{mean_c:.3f}")
    m2.metric("% Pairs > 0.7",    f"{pct07:.1f}%")
    m3.metric("% Pairs > 0.9",    f"{pct09:.1f}%")
    m4.metric("Min Correlation",  f"{vals.min():.3f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if mean_c > 0.45:
        cls, msg = "r-sys",  "🔴 SYSTEMIC CRISIS — All stocks falling together. Diversification has failed. Reduce all equity exposure."
    elif pct07 > 10:
        cls, msg = "r-clus", "🟡 CLUSTER SHOCK — Sector-level correlation spike. Specific themes under stress. Rotate defensively."
    elif mean_c < 0.20:
        cls, msg = "r-flt",  "🔵 SECTOR FLIGHT — Capital rotating between sectors. Diversified portfolio is protected. Hold positions."
    else:
        cls, msg = "r-nrm",  "🟢 NORMAL REGIME — Diversification working. Stocks moving independently. Standard risk models apply."

    st.markdown(f'<div class="regime-badge {cls}">{msg}</div>',
                unsafe_allow_html=True)

# ─── TAB 5 ────────────────────────────────────────────────────
with tab5:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shap_pct_path = os.path.join(BASE_DIR, "data", "shap_variance.csv")
    shap_rs_path  = os.path.join(BASE_DIR, "data", "shap_var_rupees.csv")

    if not os.path.exists(shap_pct_path):
        st.warning("SHAP data not found. Run `risk_engine/shap_attribution.py` first.")
    else:
        shap_pct = pd.read_csv(shap_pct_path, parse_dates=["date"], index_col="date")
        shap_rs  = pd.read_csv(shap_rs_path,  parse_dates=["date"], index_col="date") if os.path.exists(shap_rs_path) else None

        stocks = shap_pct.columns.tolist()

        # ── Stacked Area: Variance Contribution Over Time ──
        fig_area = go.Figure()
        colors = [
            "#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6",
            "#ec4899","#06b6d4","#84cc16","#f97316","#6366f1",
            "#14b8a6","#e11d48","#a855f7","#eab308"
        ]
        for i, stock in enumerate(stocks):
            fig_area.add_trace(go.Scatter(
                x=shap_pct.index, y=shap_pct[stock],
                name=stock, stackgroup="one",
                line=dict(width=0),
                fillcolor=colors[i % len(colors)],
            ))
        fig_area.update_layout(
            **PLOT_THEME,
            title="SHAP Variance Decomposition — % Contribution Over Time",
            yaxis_title="% of Portfolio Variance",
            xaxis_title="",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5,
                        font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_area, use_container_width=True)

        # ── Bar: Average % Contribution ──
        avg_pct = shap_pct.mean().sort_values(ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=avg_pct.values,
            y=avg_pct.index,
            orientation="h",
            marker=dict(
                color=avg_pct.values,
                colorscale=[[0, "#1e40af"], [0.5, "#3b82f6"], [1, "#ef4444"]],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in avg_pct.values],
            textposition="outside",
            textfont=dict(color="#9ca3af", size=11),
        ))
        fig_bar.update_layout(
            **PLOT_THEME,
            title="Average Variance Contribution by Stock (%)",
            xaxis_title="% of Portfolio Variance",
            height=420,
        )
        fig_bar.update_yaxes(tickfont=dict(size=11))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── KPI row ──
        top3 = shap_pct.mean().nlargest(3)
        bot3 = shap_pct.mean().nsmallest(3)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        for col, (stock, val) in zip([c1,c2,c3], top3.items()):
            col.metric(f"🔴 {stock}", f"{val:.1f}%", "High Risk")
        for col, (stock, val) in zip([c4,c5,c6], bot3.items()):
            col.metric(f"🟢 {stock}", f"{val:.1f}%", "Low Risk")

        # ── ₹ Exposure Table ──
        if shap_rs is not None:
            st.markdown('<div class="sec-hdr">₹ Risk Exposure (₹1 Cr Portfolio)</div>',
                        unsafe_allow_html=True)
            latest_rs = shap_rs.iloc[-1].sort_values(ascending=False)
            rs_df = pd.DataFrame({
                "Stock": latest_rs.index,
                "₹ VaR Contribution": [f"₹{v:,.0f}" for v in latest_rs.values],
                "% Share": [f"{shap_pct.iloc[-1][s]:.1f}%" for s in latest_rs.index],
            })
            st.dataframe(rs_df, use_container_width=True, hide_index=True)

# ─── TAB 6: GREEKS ENGINE ─────────────────────────────────────
with tab6:
    greeks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "greeks_results.csv"
    )
    if os.path.exists(greeks_path):
        gdf = pd.read_csv(greeks_path)
        stocks_avail = gdf["stock"].unique().tolist()

        st.markdown('<div class="sec-hdr">Black-Scholes Greeks — Live Data</div>',
                    unsafe_allow_html=True)

        # ── Source badges ──
        sources = gdf["source"].unique()
        badge_html = " ".join(
            f'<span style="background:#1e2433;padding:4px 12px;border-radius:12px;'
            f'font-size:12px;color:{"#10b981" if s=="garch_fallback" else "#6b9dfc"};'
            f'border:1px solid {"#10b981" if s=="garch_fallback" else "#6b9dfc"};">{s}</span>'
            for s in sources
        )
        st.markdown(f"Data Sources: {badge_html}", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Stock selector ──
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            sel_stock = st.selectbox("Stock", stocks_avail, index=0, key="greeks_stock")
        with col_sel2:
            sel_expiry = st.selectbox("Expiry", ["30d", "60d", "90d"], index=0, key="greeks_expiry")

        sub = gdf[(gdf["stock"] == sel_stock) & (gdf["expiry"] == sel_expiry)]

        if not sub.empty:
            spot = sub["spot"].iloc[0]
            sigma = sub["sigma_pct"].iloc[0]
            source = sub["source"].iloc[0]

            # ── KPI Row ──
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Spot Price</div>
                    <div class="kpi-value">{'$' if sel_stock == 'SPY' else '₹'}{spot:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            k2.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">GARCH σ (annualised)</div>
                    <div class="kpi-value">{sigma:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            k3.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Data Source</div>
                    <div class="kpi-value" style="font-size:16px">{source}</div>
                </div>""", unsafe_allow_html=True)
            atm_row = sub[sub["strike_type"] == "ATM"]
            atm_delta = atm_row["delta_call"].iloc[0] if not atm_row.empty else 0
            k4.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">ATM Delta</div>
                    <div class="kpi-value">{atm_delta:.3f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # ── Greeks Table ──
            st.markdown(f'<div class="sec-hdr">{sel_stock} — {sel_expiry} Greeks</div>',
                        unsafe_allow_html=True)
            display_df = sub[["strike_type", "strike", "call_price", "put_price",
                              "delta_call", "gamma", "vega", "theta_call", "rho_call"]].copy()
            display_df.columns = ["Strike Type", "Strike", "Call ₹", "Put ₹",
                                  "Delta", "Gamma", "Vega", "Theta/day", "Rho"]
            for c in ["Call ₹", "Put ₹"]:
                display_df[c] = display_df[c].apply(lambda x: f"₹{x:,.2f}")
            for c in ["Delta", "Gamma", "Vega", "Theta/day", "Rho"]:
                display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Delta Surface Heatmap (all expiries) ──
        st.markdown(f'<div class="sec-hdr">{sel_stock} — Delta Surface</div>',
                    unsafe_allow_html=True)
        stock_df = gdf[gdf["stock"] == sel_stock]
        if not stock_df.empty:
            pivot = stock_df.pivot_table(
                values="delta_call", index="strike_type", columns="expiry"
            )
            order = ["ATM", "OTM_5pct", "OTM_10pct"]
            pivot = pivot.reindex([s for s in order if s in pivot.index])
            col_order = ["30d", "60d", "90d"]
            pivot = pivot[[c for c in col_order if c in pivot.columns]]

            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[s.replace("_", " ") for s in pivot.index.tolist()],
                colorscale=[[0, "#1e1e3f"], [0.5, "#00d4aa"], [1, "#ffd93d"]],
                text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>Delta: %{z:.4f}<extra></extra>",
            ))
            fig_heat.update_layout(
                **PLOT_THEME,
                title=f"{sel_stock} Call Delta — Strike × Expiry",
                xaxis_title="Expiry",
                yaxis_title="Strike Type",
                height=350,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Greeks Radar Chart (ATM, selected expiry) ──
        atm_all = gdf[(gdf["stock"] == sel_stock) & (gdf["strike_type"] == "ATM") &
                      (gdf["expiry"] == sel_expiry)]
        if not atm_all.empty:
            r = atm_all.iloc[0]
            categories = ["Delta", "Gamma×100", "Vega", "|Theta|×10", "Rho"]
            values = [r["delta_call"], r["gamma"]*100, r["vega"],
                      abs(r["theta_call"])*10, r["rho_call"]]
            values.append(values[0])  # close the radar

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(0,212,170,0.15)",
                line=dict(color="#00d4aa", width=2),
                name=f"{sel_stock} ATM {sel_expiry}",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0f0f0f",
                    radialaxis=dict(visible=True, gridcolor="#1e2433",
                                    tickfont=dict(color="#6b7280", size=10)),
                    angularaxis=dict(gridcolor="#1e2433",
                                     tickfont=dict(color="#e5e7eb", size=12)),
                ),
                paper_bgcolor="#0f0f0f",
                font=dict(color="#e5e7eb"),
                title=dict(text=f"{sel_stock} ATM Greeks Radar — {sel_expiry}",
                           font=dict(color="#e5e7eb", size=14)),
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Greeks data not found. Run `PYTHONPATH=. python risk_engine/greeks_calculator.py` first.")

