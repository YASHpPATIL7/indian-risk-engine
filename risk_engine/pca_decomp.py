"""
PCA Risk Decomposition
Decomposes portfolio variance into orthogonal risk factors
Answers: what drives portfolio risk? Is it one factor or many?
Input:  data/vajra_returns.csv
Output: data/pca_results.csv
        data/pca_loadings.csv
        assets/pca.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
logger = logging.getLogger(__name__)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
returns_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "vajra_returns.csv"),
    index_col=0, parse_dates=True
)
stocks = list(returns_df.columns)
N      = len(stocks)
logger.info(f"Stocks: {N}, Days: {len(returns_df)}")

# ── 2. STANDARDIZE ────────────────────────────────────────────────────────────
scaler      = StandardScaler()
returns_std = scaler.fit_transform(returns_df[stocks])

# ── 3. PCA — FULL DECOMPOSITION ───────────────────────────────────────────────
pca         = PCA(n_components=N)
scores      = pca.fit_transform(returns_std)          # factor scores (T x N)
loadings    = pca.components_                          # (N x N) eigenvectors
eigenvalues = pca.explained_variance_                  # variance per PC
exp_var     = pca.explained_variance_ratio_            # % explained per PC
cum_var     = np.cumsum(exp_var)

# ── 4. PRINT VARIANCE EXPLAINED ───────────────────────────────────────────────
logger.info("\nPCA Variance Decomposition:")
logger.info(f"{'PC':<6} {'Eigenvalue':>12} {'Var%':>8} {'Cumulative%':>12}")
logger.info("─" * 42)
for i in range(N):
    marker = " ← Kaiser" if eigenvalues[i] > 1.0 else ""
    logger.info(f"PC{i+1:<4} {eigenvalues[i]:>12.4f} {exp_var[i]*100:>7.2f}% "
          f"{cum_var[i]*100:>11.2f}%{marker}")

# ── 5. KAISER CRITERION — HOW MANY FACTORS? ───────────────────────────────────
n_kaiser    = np.sum(eigenvalues > 1.0)
n_80pct     = np.argmax(cum_var >= 0.80) + 1
n_90pct     = np.argmax(cum_var >= 0.90) + 1
logger.info(f"\nKaiser criterion (eigenvalue > 1): {n_kaiser} components")
logger.info(f"Components to explain 80% variance: {n_80pct}")
logger.info(f"Components to explain 90% variance: {n_90pct}")

# ── 6. LOADINGS TABLE — PC1, PC2, PC3 ─────────────────────────────────────────
loadings_df = pd.DataFrame(
    loadings[:5].T,
    index=stocks,
    columns=[f"PC{i+1}" for i in range(5)]
)
loadings_df["communality"] = (loadings_df ** 2).sum(axis=1)

logger.info("\nFactor Loadings (PC1–PC5):")
logger.info(f"{'Stock':<14}", end="")
for col in loadings_df.columns:
    logger.info(f"{col:>10}", end="")
logger.info()
logger.info("─" * 75)
for stock in stocks:
    row = loadings_df.loc[stock]
    logger.info(f"{stock:<14}", end="")
    for col in loadings_df.columns:
        val = row[col]
        flag = " *" if abs(val) > 0.3 else "  "
        logger.info(f"{val:>9.3f}{flag}", end="")
    logger.info()

# ── 7. PORTFOLIO VARIANCE ATTRIBUTION ─────────────────────────────────────────
# How much of equal-weight portfolio variance comes from each PC?
weights   = np.full(N, 1.0 / N)
port_var  = []
for i in range(N):
    pc_loading   = loadings[i]                     # PC_i loadings across stocks
    port_exposure = weights @ pc_loading            # portfolio exposure to PC_i
    pc_var_contrib = (port_exposure ** 2) * eigenvalues[i]
    port_var.append(pc_var_contrib)

port_var      = np.array(port_var)
port_var_pct  = port_var / port_var.sum() * 100

logger.info("\nPortfolio Variance Attribution by PC:")
logger.info(f"{'PC':<6} {'Contribution%':>14}")
logger.info("─" * 22)
for i in range(8):
    logger.info(f"PC{i+1:<4} {port_var_pct[i]:>13.2f}%")
logger.info(f"{'PC9-14':<6} {port_var_pct[8:].sum():>13.2f}%")

# ── 8. SECTOR GROUPING FROM PC1 LOADINGS ──────────────────────────────────────
pc1_loadings = pd.Series(loadings[0], index=stocks).sort_values()
logger.info(f"\nPC1 Loadings (Market Factor) — sorted:")
for s, v in pc1_loadings.items():
    bar = "█" * int(abs(v) * 30)
    logger.info(f"  {s:<14} {v:>7.3f}  {bar}")

# ── 9. SAVE CSVs ───────────────────────────────────────────────────────────────
pca_summary = pd.DataFrame({
    "PC"           : [f"PC{i+1}" for i in range(N)],
    "eigenvalue"   : eigenvalues,
    "var_explained": exp_var * 100,
    "cumulative"   : cum_var * 100,
    "port_var_pct" : port_var_pct
})
pca_summary.to_csv(
    os.path.join(BASE_DIR, "data", "pca_results.csv"), index=False
)
loadings_df.to_csv(
    os.path.join(BASE_DIR, "data", "pca_loadings.csv")
)
logger.info("\nSaved → data/pca_results.csv")
logger.info("Saved → data/pca_loadings.csv")

# ── 10. PLOT ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor="#0f0f0f")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1 — Scree plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(1, N+1), exp_var * 100,
        color="#00d4aa", alpha=0.8, label="Var % per PC")
ax1.plot(range(1, N+1), cum_var * 100,
         color="#ffd93d", linewidth=2, marker="o",
         markersize=4, label="Cumulative %")
ax1.axhline(80, color="#ff6b6b", linestyle="--",
            linewidth=0.8, label="80% threshold")
ax1.axvline(n_kaiser, color="#a29bfe", linestyle="--",
            linewidth=0.8, label=f"Kaiser = {n_kaiser}")
ax1.set_facecolor("#0f0f0f")
ax1.set_title("Scree Plot — Variance Explained per PC",
              color="white", fontsize=10)
ax1.set_xlabel("Principal Component", color="white", fontsize=9)
ax1.set_ylabel("Variance Explained (%)", color="white", fontsize=9)
ax1.tick_params(colors="white", labelsize=8)
ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
ax1.spines[["top","right","left","bottom"]].set_color("#333333")
ax1.grid(alpha=0.1, color="white")

# Panel 2 — PC1 vs PC2 scores (factor map)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(scores[:, 0], scores[:, 1],
            alpha=0.2, s=4, color="#00d4aa")
ax2.set_facecolor("#0f0f0f")
ax2.set_title("PC1 vs PC2 Factor Scores\n(each dot = one trading day)",
              color="white", fontsize=10)
ax2.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% var)", color="white", fontsize=9)
ax2.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% var)", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[["top","right","left","bottom"]].set_color("#333333")
ax2.grid(alpha=0.1, color="white")

# Panel 3 — Heatmap of loadings PC1–PC5
ax3 = fig.add_subplot(gs[1, 0])
load_matrix = loadings[:5].T
im = ax3.imshow(load_matrix, cmap="RdYlGn", aspect="auto",
                vmin=-0.5, vmax=0.5)
ax3.set_xticks(range(5))
ax3.set_xticklabels([f"PC{i+1}" for i in range(5)],
                    color="white", fontsize=8)
ax3.set_yticks(range(N))
ax3.set_yticklabels(stocks, color="white", fontsize=7)
ax3.set_title("Factor Loadings Heatmap (PC1–PC5)\nGreen=positive  Red=negative",
              color="white", fontsize=10)
for i in range(N):
    for j in range(5):
        ax3.text(j, i, f"{load_matrix[i,j]:.2f}",
                 ha="center", va="center",
                 color="black", fontsize=6)
plt.colorbar(im, ax=ax3)
ax3.spines[["top","right","left","bottom"]].set_color("#333333")

# Panel 4 — Portfolio variance attribution
ax4 = fig.add_subplot(gs[1, 1])
labels  = [f"PC{i+1}" for i in range(8)] + ["PC9-14"]
values  = list(port_var_pct[:8]) + [port_var_pct[8:].sum()]
colors  = ["#00d4aa" if i == 0 else "#a29bfe" if i < 3
           else "#ffd93d" if i < 6 else "#ff6b6b"
           for i in range(len(labels))]
bars = ax4.bar(labels, values, color=colors, alpha=0.85)
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center", va="bottom",
             color="white", fontsize=7)
ax4.set_facecolor("#0f0f0f")
ax4.set_title("Portfolio Variance Attribution by PC\n"
              "PC1 = Market Factor dominance",
              color="white", fontsize=10)
ax4.set_ylabel("% of Portfolio Variance", color="white", fontsize=9)
ax4.tick_params(colors="white", labelsize=8)
ax4.spines[["top","right","left","bottom"]].set_color("#333333")
ax4.grid(alpha=0.1, color="white", axis="y")

fig.suptitle(
    "GARCH Risk Engine — PCA Risk Decomposition\n"
    "14 NSE Large-Cap Stocks | 2019–2024",
    color="white", fontsize=13
)

os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(
    os.path.join(BASE_DIR, "assets", "pca.png"),
    dpi=150, bbox_inches="tight", facecolor="#0f0f0f"
)
plt.close()
logger.info("Saved → assets/pca.png")
