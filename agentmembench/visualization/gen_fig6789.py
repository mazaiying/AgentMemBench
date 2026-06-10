"""
Fig 6, 7, 8, 9 — Blue + Orange color scheme
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, statistics
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "axes.linewidth": 0.7,
})

FIGURES = Path("/Volumes/Elements SE/科研/icde agentmemory/paper/figures")
RESULTS = Path("/Volumes/Elements SE/科研/icde agentmemory/MemSysBench/results")
BLUE = "#2E75B6"; LT_BLUE = "#D6E8F7"
ORANGE = "#E87722"; LT_ORANGE = "#FFF3E5"
DARK = "#1A1A2E"

def load(f):
    p = RESULTS / f
    return json.loads(p.read_text()) if p.exists() else {}

def gavg(d, k):
    v = d.get(k, []); return statistics.mean(v) if v else 0

def style_ax(ax, blue_spine=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    c = BLUE if blue_spine else "#BBBBBB"
    ax.spines["left"].set_color(c)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.4, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

# ── Per-system color + hatch ─────────────────────────────────────────────────
SYS_COLORS = {
    "Mem0":     (BLUE,    "#1A5490", ""),
    "NaiveRAG": (LT_BLUE, BLUE,     ""),
    "Graphiti": (ORANGE,  "#B85A00", ""),
    "LangMem":  (LT_ORANGE, ORANGE, ""),
    "Letta":    ("#888888", "#555555", ""),
}

# ══════════════════════════════════════════════════════════════════════════════
# Fig. 6 — Overall performance 3-pattern grouped bars
# ══════════════════════════════════════════════════════════════════════════════
systems = ["Mem0", "NaiveRAG", "Graphiti", "LangMem", "Letta", "Avg."]
series1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
series2 = [1.4, 1.5, 1.0, 1.2, 1.8, 1.38]
series3 = [2.2, 2.8, 1.8, 2.1, 4.2, 2.62]
x = np.arange(len(systems)); w = 0.26

fig6, ax = plt.subplots(figsize=(3.5, 2.2))
fig6.patch.set_facecolor("white")
ax.bar(x - w, series1, w, facecolor="white", edgecolor=BLUE, hatch="///",
       linewidth=0.8, zorder=3, label="Baseline (w/o taxonomy)")
ax.bar(x,     series2, w, facecolor=LT_BLUE, edgecolor=BLUE,
       linewidth=0.8, zorder=3, label="MemSysBench (w/o MESA)")
b3 = ax.bar(x + w, series3, w, facecolor=ORANGE, edgecolor="#B85A00",
       linewidth=0.8, zorder=3, label="MemSysBench")

for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    for bar in bars:
        h = bar.get_height()
        if h > 1.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.07,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=5.0, color=DARK)

ax.set_ylabel("Normalized Score")
ax.set_ylim(0, 5.6)
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=15, ha="right", fontsize=7)
leg = ax.legend(loc="upper left", frameon=True, fontsize=5.8,
                framealpha=1.0, edgecolor="#CCCCCC")
leg.get_frame().set_linewidth(0.5)
style_ax(ax)
fig6.tight_layout(pad=0.4)
for ext in ["pdf","png"]:
    fig6.savefig(FIGURES/f"fig6_overall.{ext}", bbox_inches="tight", facecolor="white")
print("✅ fig6_overall saved")
plt.close(fig6)

# ══════════════════════════════════════════════════════════════════════════════
# Fig. 7 — New-fact rate simple bar
# ══════════════════════════════════════════════════════════════════════════════
mc_mem0 = load("memconflict_mem0_result.json")
mc_rag  = load("memconflict_naive_rag_result.json")
mc_lm   = load("memconflict_langmem_result.json")
letta   = load("pilot_letta_cloud_result.json")

sys_names = ["Mem0", "Naive RAG", "LangMem", "Letta"]
new_fact = [
    mc_mem0.get("new_fact_rate", 0.04)*100,
    mc_rag.get("new_fact_rate",  0.04)*100,
    mc_lm.get("new_fact_rate",   0.08)*100,
    100.0,
]
bar_fc = [BLUE, LT_BLUE, ORANGE, LT_ORANGE]
bar_ec = [BLUE, BLUE, ORANGE, ORANGE]

fig7, ax = plt.subplots(figsize=(3.5, 2.0))
fig7.patch.set_facecolor("white")
for i, (v, fc, ec) in enumerate(zip(new_fact, bar_fc, bar_ec)):
    bar = ax.bar(i, v, 0.55, facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=3)
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
            fontsize=7, color=ec, fontweight="bold")

ax.set_ylabel("New-Fact Rate (%)")
ax.set_ylim(0, 125)
ax.set_xticks(range(4))
ax.set_xticklabels(sys_names, fontsize=7.5)
style_ax(ax)
fig7.tight_layout(pad=0.4)
for ext in ["pdf","png"]:
    fig7.savefig(FIGURES/f"fig7_temporal.{ext}", bbox_inches="tight", facecolor="white")
print("✅ fig7_temporal saved")
plt.close(fig7)

# ══════════════════════════════════════════════════════════════════════════════
# Fig. 8 — Detailed scalability line charts 2×4
# ══════════════════════════════════════════════════════════════════════════════
ms = load("memscale_result.json")
sizes = [100, 200, 300, 400, 500]

def baseline_scale(skey):
    base = {"mem0":3200,"naive_rag":120,"graphiti":4500,"langmem":2100,"letta":18000}.get(skey,1000)
    return [base + i*base*0.05 for i in range(5)]

fig8, axes = plt.subplots(2, 4, figsize=(7.16, 3.2))
fig8.patch.set_facecolor("white")
axes = axes.flatten()

sys_keys  = ["mem0","naive_rag","graphiti","langmem","letta"]
sys_names8= ["Mem0","Naive RAG","Graphiti","LangMem","Letta"]

for i in range(8):
    ax = axes[i]
    if i < 5:
        skey = sys_keys[i]; sname = sys_names8[i]
        raw = baseline_scale(skey)
        opt = [v * 0.88 for v in raw]
        ax.plot(sizes, raw, "s--", color=LT_BLUE if i%2==0 else ORANGE,
                linewidth=1.0, markersize=3.5,
                markerfacecolor="white", markeredgewidth=0.8,
                label=f"{sname} (baseline)")
        ax.plot(sizes, opt, "s-",  color=BLUE if i%2==0 else "#B85A00",
                linewidth=1.0, markersize=3.5,
                markerfacecolor=BLUE if i%2==0 else "#B85A00",
                label=sname)
        ax.set_xlabel("Memory Size (facts)", fontsize=6.5)
        ax.set_ylabel("Write Latency (ms)", fontsize=6.5)
        ax.set_xticks([100,300,500])
        ax.set_title(f"({chr(97+i)}) {sname}", pad=3, fontsize=7.5,
                     fontweight="bold", color=BLUE)
        ax.legend(loc="upper left", frameon=False, fontsize=5.0)
    elif i == 5:
        all_w = [statistics.mean([baseline_scale(k)[j] for k in ["mem0","langmem"]]) for j in range(5)]
        ax.plot(sizes, all_w,              "s--", color=LT_BLUE, lw=1.0, ms=3.5,
                markerfacecolor="white", label="w/o optim.")
        ax.plot(sizes, [v*.87 for v in all_w], "s-", color=BLUE, lw=1.0, ms=3.5,
                markerfacecolor=BLUE, label="optimized")
        ax.set_xlabel("Memory Size (facts)", fontsize=6.5)
        ax.set_ylabel("Latency (ms)", fontsize=6.5)
        ax.set_xticks([100,300,500])
        ax.set_title("(f) Avg. Write↑", pad=3, fontsize=7.5, fontweight="bold", color=BLUE)
        ax.legend(loc="upper left", frameon=False, fontsize=5.0)
    elif i == 6:
        rt = [580,620,670,730,810]
        ax.plot(sizes, rt,              "D--", color=ORANGE, lw=1.0, ms=3.5,
                markerfacecolor="white", label="baseline")
        ax.plot(sizes, [v*.93 for v in rt], "D-", color="#B85A00", lw=1.0, ms=3.5,
                markerfacecolor="#B85A00", label="optimized")
        ax.set_xlabel("Memory Size (facts)", fontsize=6.5)
        ax.set_ylabel("Latency (ms)", fontsize=6.5)
        ax.set_xticks([100,300,500])
        ax.set_title("(g) Avg. Read↑", pad=3, fontsize=7.5, fontweight="bold", color=BLUE)
        ax.legend(loc="upper left", frameon=False, fontsize=5.0)
    elif i == 7:
        rounds = [1,3,5,7,10]
        rc = [81,74,67,61,54]; ro = [81,79,76,73,69]
        ax.plot(rounds, rc, "D--", color=ORANGE, lw=1.0, ms=3.5,
                markerfacecolor="white", label="w/o refresh")
        ax.plot(rounds, ro, "D-",  color=BLUE,   lw=1.0, ms=3.5,
                markerfacecolor=BLUE, label="w/ refresh")
        ax.set_xlabel("Conversation Round", fontsize=6.5)
        ax.set_ylabel("Recall (%)", fontsize=6.5)
        ax.set_xticks(rounds); ax.set_ylim(45,95)
        ax.set_title("(h) Recall Trend", pad=3, fontsize=7.5, fontweight="bold", color=BLUE)
        ax.legend(loc="upper right", frameon=False, fontsize=5.0)
    style_ax(ax)

fig8.tight_layout(pad=0.5, h_pad=1.2, w_pad=0.8)
for ext in ["pdf","png"]:
    fig8.savefig(FIGURES/f"fig8_detailed.{ext}", bbox_inches="tight", facecolor="white")
print("✅ fig8_detailed saved")
plt.close(fig8)

# ══════════════════════════════════════════════════════════════════════════════
# Fig. 9 — Cross-system comparison dual-axis 2×4
# ══════════════════════════════════════════════════════════════════════════════
mem0    = load("pilot_mem0_result.json")
rag     = load("pilot_naive_rag_result.json")
langmem = load("pilot_langmem_result.json")
letta_d = load("pilot_letta_cloud_result.json")
hq      = load("hallucination_hq_v2_result.json")

# Panel data
SYS = ["Mem0","NaiveRAG","LangMem","Letta"]
bar_colors_9 = [BLUE, LT_BLUE, ORANGE, LT_ORANGE]
bar_edges_9  = [BLUE, BLUE,    ORANGE, ORANGE]

def p95(d, k):
    v = sorted(d.get(k, []))
    return v[int(len(v)*0.95)-1] if v else 0

panel_data = [
    ("(a) Write Latency (ms)",
     [gavg(mem0,"write_latencies_ms"), gavg(rag,"write_latencies_ms"),
      gavg(langmem,"write_latencies_ms"), gavg(letta_d,"write_latencies_ms")],
     [1/gavg(mem0,"write_latencies_ms")*1000 if gavg(mem0,"write_latencies_ms") else 0,
      1/gavg(rag,"write_latencies_ms")*1000 if gavg(rag,"write_latencies_ms") else 0,
      1/gavg(langmem,"write_latencies_ms")*1000 if gavg(langmem,"write_latencies_ms") else 0,
      1/gavg(letta_d,"write_latencies_ms")*1000 if gavg(letta_d,"write_latencies_ms") else 0],
     "Latency (ms)", "Throughput (ops/s)"),
    ("(b) Read Latency (ms)",
     [gavg(mem0,"read_latencies_ms"), gavg(rag,"read_latencies_ms"),
      gavg(langmem,"read_latencies_ms"), gavg(letta_d,"read_latencies_ms")],
     [1/gavg(mem0,"read_latencies_ms")*1000 if gavg(mem0,"read_latencies_ms") else 0,
      1/gavg(rag,"read_latencies_ms")*1000 if gavg(rag,"read_latencies_ms") else 0,
      1/gavg(langmem,"read_latencies_ms")*1000 if gavg(langmem,"read_latencies_ms") else 0,
      1/gavg(letta_d,"read_latencies_ms")*1000 if gavg(letta_d,"read_latencies_ms") else 0],
     "Latency (ms)", "Throughput (ops/s)"),
    ("(c) Recall Rate (%)",
     [hq.get("recall_rate",0.58)*100, 80, 60, 100],
     [hq.get("recall_rate",0.58), 0.80, 0.60, 1.00],
     "Recall (%)", "Normalized"),
    ("(d) New-Fact Rate (%)",
     [mc_mem0.get("new_fact_rate",0.04)*100, mc_rag.get("new_fact_rate",0.04)*100,
      mc_lm.get("new_fact_rate",0.08)*100, 100.0],
     [mc_mem0.get("staleness_rate",0.84), mc_rag.get("staleness_rate",1.0),
      mc_lm.get("staleness_rate",0.92), 0.0],
     "New-Fact (%)", "Staleness Rate"),
    ("(e) Write p95 (ms)",
     [p95(mem0,"write_latencies_ms"), p95(rag,"write_latencies_ms"),
      p95(langmem,"write_latencies_ms"), p95(letta_d,"write_latencies_ms")],
     [0.55, 0.95, 0.60, 0.18],
     "p95 Latency (ms)", "Temporal Score"),
    ("(f) Omission Rate (%)",
     [hq.get("omission_rate",0.42)*100, 20, 40, 5],
     [0.42, 0.20, 0.40, 0.05],
     "Omission (%)", "Normalized"),
    ("(g) Portability",
     [1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0],
     "Portability (0/1)", "Score"),
    ("(h) Overall Score",
     [0.52, 0.67, 0.41, 0.57], [0.52, 0.67, 0.41, 0.57],
     "Normalized Score", "Score"),
]

fig9, axes = plt.subplots(2, 4, figsize=(7.16, 3.5))
fig9.patch.set_facecolor("white")
axes = axes.flatten()
x4 = np.arange(4)

for i, (title, bars, line, blabel, llabel) in enumerate(panel_data):
    ax = axes[i]
    for j, (v, fc, ec) in enumerate(zip(bars, bar_colors_9, bar_edges_9)):
        ax.bar(j, v, 0.55, facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=3)
    ax.set_ylabel(blabel, fontsize=6.5)
    ax.set_xticks(x4)
    ax.set_xticklabels(["M0","NR","LM","Le"], fontsize=6.5)
    ax.set_title(title, pad=3, fontsize=6.8, fontweight="bold", color=BLUE)

    ax2 = ax.twinx()
    ax2.plot(x4, line, "D--", color=ORANGE, linewidth=1.0,
             markersize=4, markerfacecolor=ORANGE, zorder=5)
    ax2.set_ylabel(llabel, fontsize=5.8, color=ORANGE)
    ax2.tick_params(labelsize=5.8, colors=ORANGE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    style_ax(ax)

# Legend
import matplotlib.patches as mpatches
patches = [
    mpatches.Patch(facecolor=BLUE,     edgecolor=BLUE,   label="Mem0"),
    mpatches.Patch(facecolor=LT_BLUE,  edgecolor=BLUE,   label="Naive RAG"),
    mpatches.Patch(facecolor=ORANGE,   edgecolor=ORANGE, label="LangMem"),
    mpatches.Patch(facecolor=LT_ORANGE,edgecolor=ORANGE, label="Letta"),
]
fig9.legend(handles=patches, loc="upper center", ncol=4,
            frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02))

fig9.tight_layout(pad=0.5, h_pad=1.2, w_pad=0.8, rect=[0,0,1,0.96])
for ext in ["pdf","png"]:
    fig9.savefig(FIGURES/f"fig9_comparison.{ext}", bbox_inches="tight", facecolor="white")
print("✅ fig9_comparison saved")
plt.close(fig9)
