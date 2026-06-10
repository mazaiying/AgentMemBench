"""
Fig 10: Naive RAG Scale Experiment — write/read latency vs. scale
Academic style: black text, IEEE fonts, clean dual-axis line chart
Matches blue aesthetic of Fig 9.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size":   8.0,
    "text.color":  "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.dpi":  300,
    "savefig.dpi": 300,
})

RESULTS = Path("/Volumes/Elements SE/科研/icde agentmemory/MemSysBench/results")
FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")

# Load results
data = json.loads((RESULTS / "scale_naive_rag_result.json").read_text())
res  = data["results"]

scales       = [int(k) for k in res]
write_p50    = [res[str(s)]["write_p50_ms"]  for s in scales]
write_p95    = [res[str(s)]["write_p95_ms"]  for s in scales]
read_p50     = [res[str(s)]["read_p50_ms"]   for s in scales]
read_p95     = [res[str(s)]["read_p95_ms"]   for s in scales]
recall       = [res[str(s)]["recall_at_5"]   for s in scales]

# ── Figure layout: 1 row, 2 panels (compressed height) ───────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 1.8), facecolor="white")
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.28, top=0.93, wspace=0.42)

BLACK      = "black"
BLUE_DARK  = "#2E75B6"
BLUE_LIGHT = "#BDD7EE"
GRAY       = "#555555"

# ── Panel (a): Write Latency ─────────────────────────────────────────────────
ax1.plot(scales, write_p50, color=BLUE_DARK, lw=1.5, marker="o", ms=4,
         label="Write p50", zorder=5)
ax1.fill_between(scales, write_p50, write_p95, color=BLUE_LIGHT, alpha=0.5,
                 label="p50–p95 band")
ax1.set_xscale("log")
ax1.set_xlabel("Number of stored facts (log scale)\n(a) Write Latency vs. Scale",
               fontsize=7.5, color=BLACK)
ax1.set_ylabel("Write latency (ms)", fontsize=7.5, color=BLACK)
ax1.set_xticks(scales)
ax1.get_xaxis().set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax1.tick_params(axis="x", labelsize=7, rotation=30)
ax1.tick_params(axis="y", labelsize=7)
ax1.set_ylim(0, 2.0)
ax1.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax1.annotate("Scale-invariant\n(no LLM on write path)",
             xy=(1000, 0.5), xytext=(1000, 1.35),
             fontsize=7, color=BLACK, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.8, color=BLACK))
ax1.legend(fontsize=7, loc="upper right", framealpha=0.9,
           edgecolor=GRAY)
for sp in ax1.spines.values():
    sp.set_color(BLACK); sp.set_linewidth(0.8)

# ── Panel (b): Read Latency + Recall ────────────────────────────────────────
ax2b = ax2.twinx()

line_read, = ax2.plot(scales, read_p50, color=BLUE_DARK, lw=1.5, marker="s", ms=4,
                      label="Read p50 (ms)", zorder=5)
ax2.fill_between(scales, read_p50, read_p95, color=BLUE_LIGHT, alpha=0.5)
line_recall, = ax2b.plot(scales, [r*100 for r in recall], color=GRAY,
                          lw=1.2, ls="--", marker="^", ms=4,
                          label="Recall@5 (%)", zorder=4)

ax2.set_xscale("log")
ax2.set_xlabel("Number of stored facts (log scale)\n(b) Read Latency & Recall vs. Scale",
               fontsize=7.5, color=BLACK)
ax2.set_ylabel("Read latency p50 (ms)", fontsize=7.5, color=BLACK)
ax2b.set_ylabel("Recall@5 (%)", fontsize=7.5, color=GRAY)
ax2.set_xticks(scales)
ax2.get_xaxis().set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax2.tick_params(axis="x", labelsize=7, rotation=30)
ax2.tick_params(axis="y", labelsize=7)
ax2b.tick_params(axis="y", labelsize=7, colors=GRAY)
ax2b.set_ylim(0, 130)
ax2b.set_yticks([0, 25, 50, 75, 100])
ax2.set_ylim(0, 200)

lines  = [line_read, line_recall]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=7, loc="upper left",
           framealpha=0.9, edgecolor=GRAY)
for sp in ax2.spines.values():
    sp.set_color(BLACK); sp.set_linewidth(0.8)
for sp in ax2b.spines.values():
    sp.set_color(BLACK); sp.set_linewidth(0.8)

for ext in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig10_scale.{ext}", bbox_inches="tight", facecolor="white")
print("✅ fig10_scale (compressed) saved")
plt.close()
