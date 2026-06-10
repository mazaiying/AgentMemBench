"""
Fig. 5 — Diversity radar charts 2x4 — Grayscale / Academic style
Font black, radar lines black, fills light gray. 
Matching the provided black-and-white reference image style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import statistics

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "lines.linewidth": 1.0, "axes.linewidth": 0.8,
    "text.color": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")

BLACK = "black"
DARK_GRAY = "#444444"
MED_GRAY = "#999999"
LT_GRAY = "#E5E5E5"

N = 8
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

systems_data = {
    "Mem0":     [0.25, 0.65, 0.58, 0.18, 0.35, 0.60, 1.00, 0.55],
    "Naive RAG":[1.00, 0.85, 0.80, 0.20, 0.30, 0.95, 1.00, 0.20],
    "Graphiti": [0.10, 0.70, 0.75, 0.15, 0.40, 0.40, 0.00, 0.90],
    "LangMem":  [0.28, 0.90, 0.60, 0.10, 0.25, 0.55, 0.00, 0.65],
    "Letta":    [0.08, 0.30, 1.00, 1.00, 0.80, 0.25, 0.00, 0.95],
}
dim_labels = [
    "1. Write Latency Score",
    "2. Read Latency Score",
    "3. Recall Rate",
    "4. Temporal Consistency",
    "5. Hallucination Resistance",
    "6. Write Scalability",
    "7. LLM Portability",
    "8. Architecture Complexity",
]

fig = plt.figure(figsize=(7.16, 3.6))
fig.patch.set_facecolor("white")

def make_radar(ax, data, title):
    vals = data + data[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, vals, "o-", color=BLACK, linewidth=1.0, markersize=3)
    ax.fill(angles, vals, color=LT_GRAY, alpha=0.8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["1","2","3","4","5","6","7","8"], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["","","",""])
    ax.grid(color="#CCCCCC", linewidth=0.5)
    ax.spines["polar"].set_linewidth(0.8)
    ax.spines["polar"].set_color(BLACK)
    ax.set_title(title, size=7.5, pad=6, fontweight="bold", color=BLACK)

for idx, (sysname, data) in enumerate(systems_data.items()):
    ax = fig.add_subplot(2, 4, idx + 1, projection="polar")
    make_radar(ax, data, f"({chr(ord('a')+idx)}) {sysname}")

# Panel 6: legend
ax_leg = fig.add_subplot(2, 4, 6)
ax_leg.axis("off")
for i, label in enumerate(dim_labels):
    ax_leg.text(0.05, 0.96 - i*0.115, label,
                transform=ax_leg.transAxes, fontsize=5.8, va="top",
                color=BLACK)
ax_leg.set_title("(f) Legend", size=7.5, pad=6, fontweight="bold", color=BLACK)

# Panel 7: avg score bars
ax7 = fig.add_subplot(2, 4, 7)
avg_scores = {s: np.mean(v) for s, v in systems_data.items()}
# Academic grayscale palette for bars
bar_colors = ["#222222", "#666666", "#AAAAAA", "#E0E0E0", "white"]
for i, (n, v) in enumerate(avg_scores.items()):
    ax7.bar(i, v, 0.65,
            facecolor=bar_colors[i], edgecolor=BLACK,
            linewidth=0.8, zorder=3)
ax7.set_ylim(0, 1.0)
ax7.set_xticks(range(5))
ax7.set_xticklabels(["M0","NR","Gr","LM","Le"], fontsize=6.5)
ax7.set_ylabel("Avg. Score", fontsize=7, color=BLACK)
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)
ax7.spines["left"].set_color(BLACK)
ax7.spines["bottom"].set_color(BLACK)
ax7.yaxis.grid(True, alpha=0.3, linewidth=0.4)
ax7.set_axisbelow(True)
ax7.set_title("(g) Avg. Score", size=7.5, pad=6, fontweight="bold", color=BLACK)

# Panel 8: portability
ax8 = fig.add_subplot(2, 4, 8)
port = [1.0, 1.0, 0.0, 0.0, 0.0]
for i, (v, c) in enumerate(zip(port, bar_colors)):
    ax8.bar(i, max(v, 0.04), 0.65,
            facecolor=c, edgecolor=BLACK,
            linewidth=0.8, zorder=3)
ax8.set_ylim(0, 1.3)
ax8.set_yticks([0, 1]); ax8.set_yticklabels(["Fail", "Pass"])
ax8.set_xticks(range(5))
ax8.set_xticklabels(["M0","NR","Gr","LM","Le"], fontsize=6.5)
ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)
ax8.spines["left"].set_color(BLACK)
ax8.spines["bottom"].set_color(BLACK)
ax8.yaxis.grid(True, alpha=0.3, linewidth=0.4)
ax8.set_axisbelow(True)
ax8.set_title("(h) Portability", size=7.5, pad=6, fontweight="bold", color=BLACK)

fig.text(0.5, -0.02,
    "1.Write latency  2.Read latency  3.Recall rate  4.Temporal consistency"
    "  5.Hallucination resistance  6.Scalability  7.Portability  8.Architecture complexity",
    ha="center", fontsize=5.5, va="top", color=BLACK,
    bbox=dict(boxstyle="square", facecolor="white", edgecolor=BLACK,
              linewidth=0.8, pad=0.3))

fig.tight_layout(pad=0.6, h_pad=1.0, w_pad=0.8)
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig5_diversity.{ext}",
                bbox_inches="tight", facecolor="white")
print("✅ fig5_diversity saved")
plt.close()
