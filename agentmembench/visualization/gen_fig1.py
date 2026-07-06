"""
Fig. 1 — Architecture comparison
STYLE: All text BLACK, all borders BLACK, 8pt unified font (IEEE body ~10pt, figures slightly smaller)
Architecture: nested-box structure matching reference Fig 2
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0,
    "text.color": "black",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

FIGURES = Path(__file__).resolve().parents[2] / "paper" / "figures"

# All fills light; ALL text/borders BLACK
HEADER_FILL = "#D6E4F0"   # light blue section header
INNER_FILL  = "#F2F2F2"   # light gray inner module
ACCENT_FILL = "#FDEBD0"   # light orange fill for Coupled side (structural only, text still black)
WHITE       = "white"
BLACK       = "black"

def sharp_box(ax, x, y, w, h, label="", fc=WHITE, ec=BLACK, lw=0.8,
              fs=8.0, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
        linewidth=lw, linestyle=ls, edgecolor=BLACK, facecolor=fc, zorder=3))
    if label:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=BLACK, zorder=5, multialignment="center")

def rounded_box(ax, x, y, w, h, label="", fc=WHITE, lw=0.8, fs=8.0,
                bold=False, ls="-", radius=0.12):
    style = f"round,pad=0,rounding_size={radius}"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
        linewidth=lw, linestyle=ls, edgecolor=BLACK, facecolor=fc, zorder=3))
    if label:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=BLACK, zorder=5, multialignment="center")

def arrow(ax, x1, y1, x2, y2, lw=1.2, two_head=False):
    if two_head:
        # Avoid matplotlib overlapping arrowheads in micro-gaps by using perfectly proportioned text
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.05, "$\\updownarrow$",
                ha="center", va="center", fontsize=12, fontweight="bold", color=BLACK, zorder=6)
    else:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=10,
                                    lw=lw, color=BLACK), zorder=6)

fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), facecolor=WHITE)

# ─── (a) Decoupled ───────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 11)
ax.axis("off")

# Outer dashed container
sharp_box(ax, 0.2, 0.8, 9.6, 9.6, fc="#F8FAFB", lw=1.0, ls="--")
# Section header: light blue
sharp_box(ax, 0.2, 8.5, 9.6, 1.7, "LLM Backbone", fc=HEADER_FILL, lw=0.8, bold=True, fs=9.0)
# Left rotated label
ax.text(0.55, 5.0, "Memory System", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=BLACK, rotation=90, zorder=6)
# Inner module section
sharp_box(ax, 0.9, 3.8, 8.2, 3.9, fc=INNER_FILL, lw=0.7)
ax.text(3.5, 7.35, "Query Processing", ha="center", va="center",
        fontsize=8.0, fontweight="bold", color=BLACK, zorder=5)
# Rounded component boxes
rounded_box(ax, 1.2, 5.7, 2.8, 1.4, "Query /\nUser Request", fc=WHITE, fs=7.5)
rounded_box(ax, 5.5, 5.7, 3.4, 1.4, "Vector Store\n(Simple Embedding)", fc=WHITE, fs=7.5)
rounded_box(ax, 5.5, 4.1, 3.4, 1.3, "Embedding Model", fc=INNER_FILL, fs=7.5)
# Bottom memory pool
sharp_box(ax, 0.2, 0.8, 9.6, 2.6,
          "Shared Vector Memory Pool\n(No LLM on write path)",
          fc=HEADER_FILL, lw=0.8, ls="--", bold=True, fs=7.5)
# Arrows
arrow(ax, 5.0, 8.5, 5.0, 7.7, two_head=True)
arrow(ax, 2.6, 5.7, 2.6, 3.8)
arrow(ax, 7.2, 5.7, 7.2, 5.4)
arrow(ax, 7.2, 4.1, 7.2, 3.8)
arrow(ax, 5.0, 3.8, 5.0, 3.4)
ax.text(5.0, 0.4, "(a) Decoupled  (e.g., Naive RAG)",
        ha="center", va="top", fontsize=8.0, color=BLACK)

# ─── (b) Coupled ─────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 11)
ax.axis("off")

# Outer container — slightly accented fill but BLACK border
sharp_box(ax, 0.2, 0.8, 9.6, 9.6, fc="#FFFDF8", lw=1.0, ls="--")
# Top header
sharp_box(ax, 0.2, 8.5, 9.6, 1.7, "LLM Backbone", fc=HEADER_FILL, lw=0.8, bold=True, fs=9.0)
# Left label
ax.text(0.55, 5.0, "Memory System", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=BLACK, rotation=90, zorder=6)
# LLM Extraction layer — accent fill, BLACK border/text
sharp_box(ax, 0.9, 3.8, 8.2, 4.2, fc=ACCENT_FILL, lw=1.2)
ax.text(3.5, 7.6, "LLM Extraction Layer (Write-Time)", ha="center",
        va="center", fontsize=8.0, fontweight="bold", color=BLACK, zorder=5)
# LLM Extractor box — rounded, thicker black border
rounded_box(ax, 1.1, 5.5, 3.4, 1.8, "LLM Extractor\n(Write-Time Call)",
            fc=ACCENT_FILL, lw=1.5, fs=8.0, bold=True, radius=0.12)
# Right boxes
rounded_box(ax, 5.7, 6.1, 3.2, 1.0, "Fact / Graph Store", fc=WHITE, fs=7.5)
rounded_box(ax, 5.7, 4.8, 3.2, 1.0, "Conflict Resolver\n(LLM-based)", fc=INNER_FILL, fs=6.5)
# Bottom memory store
sharp_box(ax, 0.2, 0.8, 9.6, 2.6,
          "Structured Memory Store (LLM-indexed facts)",
          fc=ACCENT_FILL, lw=1.0, ls="--", bold=True, fs=7.5)
# Arrows — all black
arrow(ax, 5.0, 8.5, 5.0, 8.0, two_head=True)
arrow(ax, 4.5, 6.4, 5.7, 6.6)
arrow(ax, 4.5, 5.8, 5.7, 5.3)
arrow(ax, 2.8, 5.5, 2.8, 3.8, lw=1.8)
ax.text(3.1, 4.65, "Synchronous LLM call", ha="left", va="center",
        fontsize=6.5, color=BLACK, style="italic", zorder=7)
ax.text(5.0, 0.4, "(b) Coupled  (e.g., Mem0, LangMem, Graphiti)",
        ha="center", va="top", fontsize=8.0, color=BLACK)

fig.subplots_adjust(bottom=0.08, top=0.99, left=0.01, right=0.99, wspace=0.04)
for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig1_arch_compare.{ext}", bbox_inches="tight", facecolor=WHITE)
print("✅ fig1 saved")
plt.close()
