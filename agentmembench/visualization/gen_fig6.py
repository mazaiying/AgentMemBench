"""
Fig. 6 — Overall Performance grouped bar chart
STYLE: Reference right-side image — BLUE palette throughout
  Series 1: white fill + '////' diagonal hatch + blue edge
  Series 2: light blue fill + '....' dots hatch + blue edge
  Series 3: solid medium blue fill + blue edge
Font: 8pt, axes black, spines black
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "text.color": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "hatch.linewidth": 0.7,
    "axes.linewidth": 0.8,
})

FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")

# Blue palette — matching right reference image exactly
BLUE_EDGE   = "#2E75B6"    # blue border for all bars
BLUE_LIGHT  = "#BDD7EE"    # light blue fill
BLUE_MED    = "#5BA3D0"    # medium blue solid
BLUE_DARK   = "#2E75B6"    # dark/solid blue

# 3 series — blue-family fills, BLACK edges (as required)
FILLS   = ["white",      BLUE_LIGHT,  BLUE_DARK ]
HATCHES = ["////",       "....",       ""         ]
LABELS  = ["Baseline (w/o taxonomy)", "MemSysBench (w/o MESA)", "MemSysBench"]

systems = ["Mem0", "NaiveRAG", "Graphiti", "LangMem", "Letta", "Avg."]
series1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
series2 = [1.4, 1.5, 1.0, 1.2, 1.8, 1.38]
series3 = [2.2, 2.8, 1.8, 2.1, 4.2, 2.62]

x = np.arange(len(systems))
w = 0.26

fig, ax = plt.subplots(figsize=(3.5, 2.4))

for si, (vals, fc, hatch, lbl) in enumerate(zip(
        [series1, series2, series3], FILLS, HATCHES, LABELS)):
    ax.bar(x + (si-1)*w, vals, w, zorder=3,
           facecolor=fc, edgecolor="black",   # BLACK edges
           hatch=hatch, linewidth=0.8, label=lbl)

ax.set_ylabel("Normalized Score", color="black")
ax.set_ylim(0, 5.3)
ax.set_xticks(x)
ax.set_xticklabels(systems)

for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", colors="black", direction="in",
               length=2.5, width=0.6, labelcolor="black")
ax.yaxis.grid(True, alpha=0.3, linewidth=0.4, zorder=0, color="gray")
ax.set_axisbelow(True)

leg = ax.legend(loc="upper left", frameon=True, fontsize=7.0,
                framealpha=1.0, edgecolor="black", ncol=1,
                handlelength=1.6, handleheight=1.0,
                borderpad=0.5, labelspacing=0.4)
leg.get_frame().set_linewidth(0.6)

fig.tight_layout(pad=0.5)
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig6_overall.{ext}", bbox_inches="tight")
print("✅ fig6_overall saved")
plt.close()
