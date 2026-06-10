"""
Fig. 2 Motivation grouped bar
STYLE: Blue palette (matching reference right image) + BLACK edges + BLACK axes
  Series 1 (Traditional): white fill + '////' diagonal hatch + BLACK edge
  Series 2 (MemSysBench): solid dark blue fill + BLACK edge
Font: 8pt unified, all text BLACK
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
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "text.color": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0.8,
    "hatch.linewidth": 0.7,
})

FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")

# Blue palette — matching reference right image + BLACK borders
BLUE_LIGHT = "#BDD7EE"   # light blue for hatch fill
BLUE_DARK  = "#2E75B6"   # solid dark blue

metrics = [
    "Write\nLatency\nSensitivity",
    "Temporal\nConsistency\nRequirement",
    "LLM-\nCoupling\nOverhead",
    "Memory\nConflict\nRate",
    "Cross-LLM\nPortability\nVariance",
]
traditional = [5, 3, 2, 4, 6]
memsys      = [88, 91, 95, 87, 82]

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(3.5, 2.4))

# Series 1: white + '////' hatch + BLACK edge (Traditional)
ax.bar(x - w/2, traditional, w, zorder=3,
       facecolor="white", edgecolor="black",   # BLACK edge
       hatch="////", linewidth=0.8,
       label="Traditional Benchmarks")

# Series 2: solid blue + BLACK edge (MemSysBench)
ax.bar(x + w/2, memsys, w, zorder=3,
       facecolor=BLUE_DARK, edgecolor="black",  # BLACK edge
       linewidth=0.8,
       label="MemSysBench")

ax.set_ylabel("Normalized Characteristic (%)", color="black")
ax.set_ylim(0, 118)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=7.0, color="black")

# Legend — black border
leg = ax.legend(loc="upper right", frameon=True, fontsize=7.0,
                framealpha=1.0, edgecolor="black",
                handlelength=1.4, handleheight=0.9)
leg.get_frame().set_linewidth(0.6)

# Spines: bottom + left BLACK, top + right hidden
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="both", colors="black", direction="in",
               length=2.5, width=0.6, labelcolor="black")
ax.yaxis.grid(True, alpha=0.25, linewidth=0.4, color="gray", zorder=0)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.4)
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig2_motivation.{ext}",
                bbox_inches="tight", facecolor="white")
print("✅ fig2_motivation saved")
plt.close()
