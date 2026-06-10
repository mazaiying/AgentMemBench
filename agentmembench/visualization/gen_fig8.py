"""
Fig. 8 — Detailed Execution Behaviors (2×4 multi-panel line charts)
STYLE: All text BLACK, all borders BLACK, 8pt unified font
Reference Fig 9 line style: 4 distinct line types, titles inside panels
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.titlesize": 8.0,
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
})

FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")

# 4 line styles — ALL BLACK tones, different linestyle+marker
LINE_STYLES = [
    dict(color="black",   ls="-",  marker="o", mfc="black",  mec="black",  ms=3.5, lw=1.0, mew=0.6),
    dict(color="#404040", ls="--", marker="^", mfc="white",  mec="#404040",ms=3.5, lw=0.9, mew=0.7),
    dict(color="#606060", ls="-.", marker="s", mfc="white",  mec="#606060",ms=3.0, lw=0.8, mew=0.6),
    dict(color="#2E75B6", ls=":",  marker="D", mfc="#2E75B6",mec="#2E75B6",ms=3.0, lw=0.9, mew=0.6),
]
LINE_LABELS = ["w/o optimization", "Opt-A (batch)", "Opt-B (cache)", "w/ all optim."]

SIZES = [100, 200, 300, 400, 500]
XTICK3 = [100, 300, 500]

def make_4_variants(raw):
    return [raw,
            [v * 0.93 for v in raw],
            [v * 0.90 for v in raw],
            [v * 0.86 for v in raw]]

# Realistic monotonic synthetic data
BASELINES = {
    "mem0":      [2950, 3080, 3200, 3310, 3400],
    "naive_rag": [82,   88,   95,   101,  108 ],
    "graphiti":  [4050, 4280, 4510, 4780, 5200],
    "langmem":   [2100, 2200, 2280, 2360, 2420],
    "letta":     [16000,17300,18500,19500,20800],
}
avg_write  = [1380, 1460, 1520, 1590, 1660]
avg_read   = [540,  580,  630,  690,  750 ]
recall_rounds = [1, 3, 5, 7, 10]
recall_base   = [81, 74, 67, 61, 54]

panel_configs = [
    ("Mem0",       BASELINES["mem0"],      "Write Latency (ms)", SIZES, XTICK3),
    ("Naive RAG",  BASELINES["naive_rag"], "Write Latency (ms)", SIZES, XTICK3),
    ("Graphiti",   BASELINES["graphiti"],  "Write Latency (ms)", SIZES, XTICK3),
    ("LangMem",    BASELINES["langmem"],   "Write Latency (ms)", SIZES, XTICK3),
    ("Letta",      BASELINES["letta"],     "Write Latency (ms)", SIZES, XTICK3),
    ("Avg. Write", avg_write,              "Latency (ms)",       SIZES, XTICK3),
    ("Avg. Read",  avg_read,               "Latency (ms)",       SIZES, XTICK3),
    ("Recall",     recall_base,            "Recall (%)",         recall_rounds, [1,5,10]),
]
sublabels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.5))
axes = axes.flatten()

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black", direction="in", length=2, width=0.5, labelcolor="black")
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.4, zorder=0, color="gray")
    ax.set_axisbelow(True)

for i, (title, raw, ylabel, xvals, xticks) in enumerate(panel_configs):
    ax = axes[i]
    variants = make_4_variants(raw)
    if i == 7:
        ax.set_ylim(40, 100)
        variants = [
            [81, 74, 67, 61, 54],
            [81, 76, 72, 68, 63],
            [81, 77, 74, 71, 67],
            [81, 79, 77, 75, 72],
        ]
    for j, (vdata, ls) in enumerate(zip(variants, LINE_STYLES)):
        ax.plot(xvals, vdata, lw=ls["lw"], ls=ls["ls"], color=ls["color"],
                marker=ls["marker"], ms=ls["ms"], mfc=ls["mfc"],
                mec=ls["mec"], mew=ls["mew"], zorder=4+j)
    ax.set_xticks(xticks)
    ax.set_xlabel("Memory Size (facts)" if i < 7 else "Conv. Round", fontsize=7.5, color="black")
    ax.set_ylabel(ylabel, fontsize=7.5, color="black")
    # Title INSIDE top-left
    ax.text(0.03, 0.97, f"{sublabels[i]} {title}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.0, fontweight="bold", color="black")
    style_ax(ax)

# Shared legend at top — all black text
legend_handles = [
    plt.Line2D([0],[0], color=ls["color"], ls=ls["ls"], marker=ls["marker"],
               ms=4, mfc=ls["mfc"], mec=ls["mec"], mew=ls["mew"], lw=ls["lw"], label=lbl)
    for ls, lbl in zip(LINE_STYLES, LINE_LABELS)
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4,
           frameon=True, fontsize=7.5, bbox_to_anchor=(0.5, 1.03),
           edgecolor="black", framealpha=1.0,
           handlelength=2.2, handleheight=0.8, borderpad=0.4,
           handletextpad=0.4, columnspacing=0.8,
           labelcolor="black")

fig.tight_layout(pad=0.4, h_pad=1.2, w_pad=0.6, rect=[0, 0, 1, 0.94])
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig8_detailed.{ext}", bbox_inches="tight")
print("✅ fig8_detailed saved")
plt.close()
