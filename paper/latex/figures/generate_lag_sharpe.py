"""
Figure 2: Web intelligence utility by modality lag (bar-aligned BTC dataset).
Produces paper/latex/figures/lag_sharpe.pdf
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "latex" / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Load bar-aligned dataset (most recent run)
# ---------------------------------------------------------------------------
RUNS_DIR = ROOT / "experiments" / "results_mm"
runs = sorted(RUNS_DIR.glob("mm_deep_btc_bar_aligned_*"))
if not runs:
    raise FileNotFoundError("No bar-aligned run found under experiments/results_mm/")
RUN = runs[-1]

df = pd.read_parquet(RUN / "BTCUSDT_15m_lb64_hz4_bar_deep.parquet")
lags = df["modality_age_minutes"].values
signal = df["news_direction_score"].values
future_ret = df["future_return"].values

# ---------------------------------------------------------------------------
# Lag-stratified signal Sharpe
# ---------------------------------------------------------------------------
BIN_EDGES = [0, 30, 60, 90, 120]
BIN_LABELS = ["0–30\nmin", "30–60\nmin", "60–90\nmin", "90–120\nmin"]

sharpes = []
ns = []
for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
    mask = (lags >= lo) & (lags < hi) & (np.abs(signal) > 0.01)
    n = mask.sum()
    if n < 2:
        sharpes.append(np.nan)
        ns.append(n)
        continue
    rets = np.sign(signal[mask]) * future_ret[mask]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252 * 4))
    sharpes.append(sharpe)
    ns.append(n)

sharpes = np.array(sharpes, dtype=float)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)

x = np.arange(len(BIN_LABELS))
bar_colors = ["#415A77" if s >= 0 else "#C0392B" for s in sharpes]
bars = ax.bar(x, sharpes, color=bar_colors, edgecolor="black", linewidth=0.6,
              width=0.55, zorder=3)

# Annotate N per bar
for xi, (s, n) in enumerate(zip(sharpes, ns)):
    offset = 0.10 if s >= 0 else -0.18
    ax.text(xi, s + offset, f"N={n}", ha="center", va="bottom" if s >= 0 else "top",
            fontsize=7.0, color="#333333")

ax.axhline(0, color="black", linewidth=0.8, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(BIN_LABELS)
ax.set_ylabel("Annualised signal Sharpe")
ax.set_xlabel(r"Modality lag $\tau_{\mathrm{lag}}$")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8, zorder=0)
ax.set_axisbelow(True)

pos_patch = mpatches.Patch(color="#415A77", label="Positive")
neg_patch = mpatches.Patch(color="#C0392B", label="Negative")
legend = ax.legend(handles=[pos_patch, neg_patch], loc="upper right",
                   fontsize=7.5, frameon=True)
legend.get_frame().set_linewidth(0.6)
legend.get_frame().set_edgecolor("#BBBBBB")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#555555")
ax.spines["bottom"].set_color("#555555")

output_path = FIG_DIR / "lag_sharpe.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {output_path}")

# Print summary
print("\nLag-Sharpe summary (bar-aligned BTC, n=7295 samples):")
for label, s, n in zip(BIN_LABELS, sharpes, ns):
    tag = label.replace('\n', ' ')
    print(f"  {tag}: Sharpe={s:+.3f}, N={n}")
