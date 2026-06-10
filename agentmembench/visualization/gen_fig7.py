"""
Fig. 7 — New-Fact Rate by System (single-series bar chart)
STYLE: Blue palette matching reference right image
Each bar uses alternating hatch pattern (like reference paper style):
  white + '////' hatch + black edge
  lightblue + '....' + black edge
  lightblue + '\\\\\\\\' + black edge
  solid darkblue + black edge
Font: 8pt, axes black, all text black
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
    "axes.labelsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
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
RESULTS = Path("/Volumes/Elements SE/科研/icde agentmemory/MemSysBench/results")

BLUE_LIGHT = "#BDD7EE"
BLUE_DARK  = "#2E75B6"

# All bars use the same solid blue color with black edges (single series, no legend)
BAR_STYLES = [
    dict(facecolor=BLUE_DARK, hatch="", edgecolor="black", linewidth=0.8)
] * 4

def load(f):
    p = RESULTS / f
    return json.loads(p.read_text()) if p.exists() else {}

mc_mem0 = load("memconflict_mem0_result.json")
mc_rag  = load("memconflict_naive_rag_result.json")
mc_lm   = load("memconflict_langmem_result.json")
letta   = load("pilot_letta_cloud_result.json")

systems  = ["Mem0", "Naive RAG", "LangMem", "Letta"]
new_fact = [
    mc_mem0.get("new_fact_rate", 0.16) * 100,
    mc_rag.get("new_fact_rate",  0.20) * 100,
    mc_lm.get("new_fact_rate",   0.16) * 100,
    letta.get("memconflict", {}).get("new_fact_rate", 1.0) * 100,
]

fig, ax = plt.subplots(figsize=(3.5, 2.0))
x = np.arange(len(systems))
w = 0.55

for j, (val, sty) in enumerate(zip(new_fact, BAR_STYLES)):
    bar = ax.bar(j, val, w, zorder=3, **sty)

# Value labels — black text
for j, val in enumerate(new_fact):
    ax.text(j, val + 1.5, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=7.0, color="black")

ax.set_ylabel("New-Fact Rate (%)", color="black")
ax.set_ylim(0, 120)
ax.set_xticks(x)
ax.set_xticklabels(systems, color="black")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black");   ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_color("black"); ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(axis="both", colors="black", direction="in", length=2.5, width=0.6)
ax.yaxis.grid(True, alpha=0.25, linewidth=0.4, zorder=0, color="gray")
ax.set_axisbelow(True)

fig.tight_layout(pad=0.4)
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig7_temporal.{ext}", bbox_inches="tight")
print("✅ fig7_temporal saved")
plt.close()
