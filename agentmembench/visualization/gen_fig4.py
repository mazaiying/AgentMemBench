"""
Fig. 4 — MemConflict example — Clean academic style (no boxes)
Uses left-side colored accent bars to distinguish phases.
No dashed boxes. Minimal, publication-ready.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[2] / "paper" / "figures"

matplotlib.rcParams.update({
    "font.family":   "serif",
    "font.serif":    ["Times New Roman", "DejaVu Serif"],
    "font.size":     8,
    "text.color":    "black",
    "figure.dpi":    300,
    "savefig.dpi":   300,
})

# ── Color palette ─────────────────────────────────────────────────────────────
BLACK     = "#1A1A1A"
GRAY_LN   = "#999999"   # line numbers
BLUE      = "#2E75B6"   # Round 1 accent
ORANGE    = "#E07B39"   # Round 5 (update) accent
RED       = "#C0392B"   # Round 8 (stale) accent
RESULT_BG = "#FFF3F3"   # faint red background for result line

# ── Content definition ────────────────────────────────────────────────────────
# (phase, ln, text, is_bold, color_override)
SECTIONS = [
    {
        "label":  "Round 1  —  Initial write",
        "color":  BLUE,
        "lines": [
            ("1",  'User: "My favorite city is Paris."',         False, None),
            ("2",  'Memory.write(key="city",  val="Paris")',     False, None),
            ("3",  '→  write_status = SUCCESS',                  False, GRAY_LN),
        ],
    },
    {
        "label":  "Round 5  —  Conflicting update",
        "color":  ORANGE,
        "lines": [
            ("4",  'User: "I moved to Tokyo last year."',        False, None),
            ("5",  'Memory.write(key="city",  val="Tokyo")',     False, None),
            ("6",  '→  write_status = SUCCESS  (reported)',      False, GRAY_LN),
        ],
    },
    {
        "label":  "Round 8  —  Retrieval",
        "color":  RED,
        "lines": [
            ("7",  'q = "What is the user\'s favorite city?"',   False, None),
            ("8",  'result = Memory.read(q)',                    False, None),
            ("9",  '→  result = "Paris"   ← STALE FACT',        True,  RED),
        ],
    },
]

# ── Layout constants ──────────────────────────────────────────────────────────
FIG_W    = 3.5
LINE_H   = 0.145   # height per content line (inches)
SEC_GAP  = 0.08    # gap between sections
LABEL_H  = 0.15    # height for section header label
PAD_T    = 0.08    # top padding
PAD_B    = 0.06    # bottom padding
BAR_W    = 0.04    # left accent bar width
BAR_X    = 0.05    # left accent bar x position
TEXT_X   = BAR_X + BAR_W + 0.06  # text x position
LN_X     = TEXT_X  # line number x
CODE_X   = TEXT_X + 0.18  # code text x

# Compute total height
def compute_height():
    h = PAD_T + PAD_B
    for i, sec in enumerate(SECTIONS):
        h += LABEL_H + len(sec["lines"]) * LINE_H
        if i < len(SECTIONS) - 1:
            h += SEC_GAP
    return h

FIG_H = compute_height()

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

def to_frac_y(y_in): return y_in / FIG_H
def to_frac_x(x_in): return x_in / FIG_W

# ── Draw sections ─────────────────────────────────────────────────────────────
y = FIG_H - PAD_T

for s_idx, sec in enumerate(SECTIONS):
    color = sec["color"]
    n_lines = len(sec["lines"])
    sec_h = LABEL_H + n_lines * LINE_H

    # Draw left accent bar
    ax.add_patch(mpatches.FancyBboxPatch(
        (BAR_X, y - sec_h),
        BAR_W, sec_h,
        boxstyle="square,pad=0",
        linewidth=0, facecolor=color, alpha=0.85,
        transform=ax.transData, zorder=2))

    # Section header label
    y_label = y - LABEL_H * 0.55
    ax.text(CODE_X, y_label, sec["label"],
            ha="left", va="center",
            fontsize=7.8, fontweight="bold", color=color,
            fontfamily="serif",
            transform=ax.transData, zorder=5)
    y -= LABEL_H

    # Code lines
    for (ln, text, bold, col) in sec["lines"]:
        y_mid = y - LINE_H * 0.60
        tc = col if col else BLACK
        fw = "bold" if bold else "normal"

        # Line number
        ax.text(LN_X + 0.14, y_mid, ln + ".",
                ha="right", va="center",
                fontsize=6.5, color=GRAY_LN,
                fontfamily="monospace",
                transform=ax.transData, zorder=5)

        # Code text
        ax.text(CODE_X, y_mid, text,
                ha="left", va="center",
                fontsize=7.0, color=tc, fontweight=fw,
                fontfamily="monospace",
                transform=ax.transData, zorder=5)
        y -= LINE_H

    if s_idx < len(SECTIONS) - 1:
        # Thin horizontal rule
        ax.plot([BAR_X, FIG_W - 0.05], [y - SEC_GAP * 0.5, y - SEC_GAP * 0.5],
                color="#E0E0E0", lw=0.5, transform=ax.transData, zorder=2)
        y -= SEC_GAP

# Thin outer border only (no fill)
ax.add_patch(mpatches.FancyBboxPatch(
    (0.03, 0.03), FIG_W - 0.06, FIG_H - 0.06,
    boxstyle="square,pad=0",
    linewidth=0.8, edgecolor="#CCCCCC", facecolor="none",
    transform=ax.transData, zorder=1))

fig.savefig(FIGURES / "fig4_memconflict_example.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGURES / "fig4_memconflict_example.png", bbox_inches="tight", facecolor="white", dpi=300)
print("✅ fig4_memconflict_example (clean style) saved")
plt.close()
