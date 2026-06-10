"""
Fig. 3 — MemSysBench Overview — Reference right-image style
ALL text BLACK, ALL borders BLACK, ALL arrows BLACK (dark gray)
Fills: light gray for challenges, light blue for center/right (structural only)
No orange, no colored text anywhere.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0,
    "text.color": "black",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")

# Color palette — structural fills only, ALL text/borders BLACK
LT_GRAY   = "#EEEEEE"    # challenge boxes fill
LT_BLUE   = "#D6E8F7"    # center/right light blue fill
BLUE_DARK = "#2E75B6"    # center header banner fill (white text on this)
GRAY_MED  = "#D0D0D0"    # inner component boxes fill
BLACK     = "black"
WHITE     = "white"

fig, ax = plt.subplots(figsize=(3.5, 3.2), facecolor=WHITE)
ax.set_xlim(0, 10); ax.set_ylim(1.5, 9.7)
ax.axis("off")

# ── helpers ─────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, label, fc=WHITE, lw=1.0,
         fs=7.0, bold=False, tc=BLACK, ls="-", radius=0.2):
    """Rounded box — BLACK border always."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, linestyle=ls, edgecolor=BLACK, facecolor=fc, zorder=3))
    if label:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=tc, zorder=5, multialignment="center")

def harrow(ax, x1, y1, x2, lw=1.2):
    """Horizontal arrow — BLACK."""
    ax.annotate("", xy=(x2, y1), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=lw,
                        mutation_scale=10), zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Challenges
# ══════════════════════════════════════════════════════════════════════════════
challenges = [
    ("No Shared\nTaxonomy",       7.8),
    ("No Write-Path\nBenchmarks", 5.6),
    ("Silent Failure\nUncaptured",3.4),
]
for label, yc in challenges:
    rbox(ax, 0.1, yc - 0.6, 2.5, 1.2, label,
         fc=LT_GRAY, lw=1.0, ls="--",       # dashed BLACK border
         fs=6.8, tc=BLACK, radius=0.2)

# Section title — BLACK
ax.text(1.35, 9.2, "Challenges", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=BLACK)
ax.plot([1.35, 1.35], [8.9, 8.6], color=BLACK, lw=0.7, alpha=0.5, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# CENTER — MemSysBench core block
# ══════════════════════════════════════════════════════════════════════════════
CX, CY, CW, CH = 3.1, 1.8, 3.8, 6.8

# Outer box — BLACK border, light blue fill
ax.add_patch(mpatches.FancyBboxPatch(
    (CX, CY), CW, CH,
    boxstyle="round,pad=0,rounding_size=0.3",
    linewidth=1.8, edgecolor=BLACK, facecolor=LT_BLUE, zorder=2))

# Title banner — dark blue fill, WHITE text (only exception for readability)
ax.add_patch(mpatches.FancyBboxPatch(
    (CX, CY + CH - 1.15), CW, 1.15,
    boxstyle="round,pad=0,rounding_size=0.3",
    linewidth=0, edgecolor=BLACK, facecolor=BLUE_DARK, zorder=3))
ax.text(CX + CW/2, CY + CH - 0.57, "MemSysBench",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=WHITE, zorder=5)

# Taxonomy section — BLACK text
ty = CY + CH - 2.15
ax.text(CX + CW/2, ty, "Four-Dimensional Taxonomy",
        ha="center", va="center", fontsize=7.2,
        fontweight="bold", color=BLACK, zorder=5)

dim_y = ty - 0.4
for dim in ["D1: Storage  ·  D2: Coupling",
            "D3: Isolation  ·  D4: Deletion"]:
    ax.text(CX + CW/2, dim_y, dim, ha="center", va="center",
            fontsize=6.5, color=BLACK, zorder=5)
    dim_y -= 0.36

# Divider — BLACK
ax.plot([CX + 0.3, CX + CW - 0.3],
        [dim_y - 0.05, dim_y - 0.05],
        color=BLACK, lw=0.8, alpha=0.5, zorder=4)
div_y = dim_y - 0.25

# Two inner component boxes — gray fill, BLACK border, BLACK text
comp_y = div_y - 0.85
for xi, label in [(CX + 0.2, "MemDialogue\n(10K samples)"),
                  (CX + 2.05, "MESA\nFramework")]:
    rbox(ax, xi, comp_y, 1.55, 0.75, label,
         fc=GRAY_MED, lw=0.8, fs=6.2, tc=BLACK, bold=True, radius=0.15)

# Evaluation axes — BLACK text
ev_y = comp_y - 0.55
ax.text(CX + CW/2, ev_y, "Evaluation Axes",
        ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=BLACK, zorder=5)
for rq in ["RQ1 Write  ·  RQ2 Read  ·  RQ3 Scale",
           "RQ4 Temporal  ·  RQ5 Hallucination",
           "RQ6 Portability"]:
    ev_y -= 0.34
    ax.text(CX + CW/2, ev_y, rq, ha="center", va="center",
            fontsize=6.0, color=BLACK, zorder=5)

# Center section title — BLACK
ax.text(CX + CW/2, 9.2, "Framework", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=BLACK)
ax.plot([5.0, 5.0], [8.9, 8.6], color=BLACK, lw=0.7, alpha=0.5, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Contributions
# ══════════════════════════════════════════════════════════════════════════════
contributions = [
    ("Unified\nTaxonomy (D1–D4)", 7.8),
    ("Write-Path\nBenchmarking",   5.6),
    ("MESA Safety\nEvaluation",    3.4),
]
for label, yc in contributions:
    rbox(ax, 7.4, yc - 0.6, 2.4, 1.2, label,
         fc=LT_BLUE, lw=1.0,
         fs=6.8, tc=BLACK, bold=False, radius=0.2)

ax.text(8.6, 9.2, "Contributions", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=BLACK)
ax.plot([8.6, 8.6], [8.9, 8.6], color=BLACK, lw=0.7, alpha=0.5, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — all BLACK
# ══════════════════════════════════════════════════════════════════════════════
for yc in [7.8, 5.6, 3.4]:
    harrow(ax, 2.6, yc, CX)           # challenges → center
    harrow(ax, CX + CW, yc, 7.4)      # center → contributions

fig.tight_layout(pad=0.3)
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig3_overview.{ext}",
                bbox_inches="tight", facecolor=WHITE)
print("✅ fig3_overview saved")
plt.close()
