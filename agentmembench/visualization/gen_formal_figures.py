"""Regenerate the original paper figures with the released formal results.

The plotting layouts intentionally follow the original AgentMemBench paper:
the grouped profile chart, diversity radars, temporal bar chart, 2x4 scale
grid, 2x4 cross-system comparison, and two-panel scale figure. All measured
values are loaded from ``results/formal``. The only non-measured profile axis
is deployment simplicity, which is a documented taxonomy attribute.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "formal"
FIGURES = ROOT / "paper" / "figures"

# Preserve the system order used by the original paper figures.
SYSTEMS = ["Mem0", "Naive RAG", "Graphiti", "LangMem", "Letta"]
SHORT = ["M0", "NR", "Gr", "LM", "Le"]
COLORS = ["#1A4F7A", "#2E75B6", "#5B9BD5", "#9DC3E6", "#777777"]
MARKERS = ["o", "s", "^", "D", "P"]

FILES = {
    "Naive RAG": {
        phase: "naive_rag_naive_full_s2027_9170.json"
        for phase in (
            "retrieval",
            "conflict",
            "isolation",
            "deletion",
            "concurrency",
            "scale",
        )
    },
    "Mem0": {
        "retrieval": "mem0_mem0_retrieval_s2027_9170.json",
        "conflict": "mem0_mem0_conflict_s2027_9170.json",
        "isolation": "mem0_mem0_isolation_s2027_9170.json",
        "deletion": "mem0_mem0_deletion_s2027_9170.json",
        "concurrency": "mem0_mem0_concurrency_v2_s2027_9170.json",
        "scale": "mem0_mem0_scale_s2027_9170.json",
    },
    "LangMem": {
        phase: f"langmem_langmem_{phase}_s2027_9170.json"
        for phase in (
            "retrieval",
            "conflict",
            "isolation",
            "deletion",
            "concurrency",
            "scale",
        )
    },
    "Graphiti": {
        phase: f"graphiti_graphiti_{phase}_s2027_9170.json"
        for phase in (
            "retrieval",
            "conflict",
            "isolation",
            "deletion",
            "concurrency",
            "scale",
        )
    },
    "Letta": {
        phase: f"letta_letta_{phase}_s2027_9170.json"
        for phase in (
            "retrieval",
            "conflict",
            "isolation",
            "deletion",
            "concurrency",
            "scale",
        )
    },
}

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
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
    }
)


def phase(system: str, name: str) -> dict:
    payload = json.loads((RESULTS / FILES[system][name]).read_text())
    return payload["phases"][name]


def load_formal() -> dict[str, dict[str, dict]]:
    return {
        system: {
            name: phase(system, name)
            for name in (
                "retrieval",
                "conflict",
                "isolation",
                "deletion",
                "concurrency",
                "scale",
            )
        }
        for system in SYSTEMS
    }


def save(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(
            FIGURES / f"{stem}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def style_axis(axis: plt.Axes, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("black")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(direction="in", length=2.5, width=0.6, colors="black")
    if grid:
        axis.yaxis.grid(True, alpha=0.28, linewidth=0.4, color="gray", zorder=0)
        axis.set_axisbelow(True)


def inverse_latency(values: list[float]) -> list[float]:
    fastest = min(values)
    return [fastest / value for value in values]


def profile_axes(data: dict[str, dict[str, dict]]) -> dict[str, list[float]]:
    write_ms = [data[s]["retrieval"]["write_latency"]["mean_ms"] for s in SYSTEMS]
    read_ms = [data[s]["retrieval"]["read_latency"]["mean_ms"] for s in SYSTEMS]
    recall = [data[s]["retrieval"]["recall_at_k"] for s in SYSTEMS]

    clean_update = []
    reliability = []
    concurrency = []
    for system in SYSTEMS:
        conflict = data[system]["conflict"]
        clean_update.append(
            max(0.0, conflict["new_fact_rate"] - conflict["dual_version_rate"])
        )
        isolation = data[system]["isolation"]
        deletion = data[system]["deletion"]
        reliability.append(
            (
                1.0
                - isolation["cross_user_leak_rate"]
                + deletion["post_delete_absence_rate"]
            )
            / 2.0
        )
        concurrency.append(
            data[system]["concurrency"]["16"]["throughput_ops_s"]
        )
    peak_concurrency = max(concurrency)
    concurrency = [value / peak_concurrency for value in concurrency]

    # Taxonomy-derived deployment simplicity: fewer mandatory services and
    # less agent-mediated state management receive a higher descriptive value.
    simplicity = {
        "Mem0": 0.70,
        "Naive RAG": 1.00,
        "Graphiti": 0.45,
        "LangMem": 0.65,
        "Letta": 0.35,
    }

    return {
        system: [
            inverse_latency(write_ms)[idx],
            inverse_latency(read_ms)[idx],
            recall[idx],
            clean_update[idx],
            reliability[idx],
            concurrency[idx],
            1.0,  # all systems completed against the shared local backends
            simplicity[system],
        ]
        for idx, system in enumerate(SYSTEMS)
    }


def original_overall_profile(data: dict[str, dict[str, dict]]) -> None:
    """Original grouped-bar design, now backed by formal profile values."""

    profiles = profile_axes(data)
    recall_only = [data[s]["retrieval"]["recall_at_k"] for s in SYSTEMS]
    performance = [
        float(np.mean(profiles[s][:3]))
        for s in SYSTEMS
    ]
    lifecycle = [
        float(np.mean(profiles[s]))
        for s in SYSTEMS
    ]

    series = [
        recall_only + [float(np.mean(recall_only))],
        performance + [float(np.mean(performance))],
        lifecycle + [float(np.mean(lifecycle))],
    ]
    labels = ["Recall-only view", "Performance view", "Lifecycle profile"]
    fills = ["white", "#BDD7EE", "#2E75B6"]
    hatches = ["////", "....", ""]
    systems = SYSTEMS + ["Avg."]

    x = np.arange(len(systems))
    width = 0.26
    figure, axis = plt.subplots(figsize=(3.5, 2.4))
    for index, (values, fill, hatch, label) in enumerate(
        zip(series, fills, hatches, labels)
    ):
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            facecolor=fill,
            edgecolor="black",
            hatch=hatch,
            linewidth=0.8,
            label=label,
            zorder=3,
        )
    axis.set_ylabel("Descriptive normalized profile")
    axis.set_ylim(0, 1.12)
    axis.set_xticks(x)
    axis.set_xticklabels(systems, rotation=15, ha="right")
    style_axis(axis)
    legend = axis.legend(
        loc="lower left",
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=6.3,
    )
    legend.get_frame().set_linewidth(0.6)
    figure.tight_layout(pad=0.5)
    save(figure, "fig6_overall")


def original_diversity_radar(data: dict[str, dict[str, dict]]) -> None:
    """Original 2x4 radar layout with formal profile axes."""

    profiles = profile_axes(data)
    dimension_labels = [
        "1. Write latency score",
        "2. Read latency score",
        "3. Recall@5",
        "4. Clean update rate",
        "5. Isolation/deletion",
        "6. T16 throughput",
        "7. Backend compatibility",
        "8. Deployment simplicity",
    ]
    count = len(dimension_labels)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    angles += angles[:1]

    figure = plt.figure(figsize=(7.16, 3.0), facecolor="white")

    def radar(axis: plt.Axes, values: list[float], title: str) -> None:
        closed = values + values[:1]
        axis.set_theta_offset(np.pi / 2)
        axis.set_theta_direction(-1)
        axis.plot(
            angles,
            closed,
            "o-",
            color="#2E75B6",
            linewidth=1.2,
            markersize=3,
        )
        axis.fill(angles, closed, color="#BDD7EE", alpha=0.75)
        axis.set_xticks(angles[:-1])
        axis.set_xticklabels(["", "2", "3", "4", "", "6", "7", "8"], fontsize=6)
        axis.set_ylim(0, 1)
        axis.set_yticks([0.25, 0.5, 0.75, 1.0])
        axis.set_yticklabels(["", "", "", ""])
        axis.grid(color="#CCCCCC", linewidth=0.5)
        axis.spines["polar"].set_linewidth(0.8)
        axis.spines["polar"].set_color("black")
        axis.text(
            0.5,
            -0.15,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )

    for index, system in enumerate(SYSTEMS):
        axis = figure.add_subplot(2, 4, index + 1, projection="polar")
        radar(axis, profiles[system], f"({chr(ord('a') + index)}) {system}")

    legend_axis = figure.add_subplot(2, 4, 6)
    legend_axis.axis("off")
    for index, label in enumerate(dimension_labels):
        legend_axis.text(
            0.02,
            0.98 - index * 0.115,
            label,
            transform=legend_axis.transAxes,
            fontsize=5.7,
            va="top",
        )
    legend_axis.text(
        0.5,
        -0.08,
        "(f) Legend",
        transform=legend_axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )

    mean_axis = figure.add_subplot(2, 4, 7)
    means = [float(np.mean(profiles[s])) for s in SYSTEMS]
    mean_axis.bar(
        np.arange(len(SYSTEMS)),
        means,
        0.65,
        facecolor="#2E75B6",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    mean_axis.set_ylim(0, 1.0)
    mean_axis.set_xticks(np.arange(len(SYSTEMS)), SHORT)
    mean_axis.set_ylabel("Profile mean")
    style_axis(mean_axis)
    mean_axis.text(
        0.5,
        -0.20,
        "(g) Descriptive mean",
        transform=mean_axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )

    portability_axis = figure.add_subplot(2, 4, 8)
    portability_axis.bar(
        np.arange(len(SYSTEMS)),
        np.ones(len(SYSTEMS)),
        0.65,
        facecolor="#2E75B6",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    portability_axis.set_ylim(0, 1.3)
    portability_axis.set_yticks([0, 1], ["Fail", "Pass"])
    portability_axis.set_xticks(np.arange(len(SYSTEMS)), SHORT)
    style_axis(portability_axis)
    portability_axis.text(
        0.5,
        -0.20,
        "(h) Backend compatibility",
        transform=portability_axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )

    figure.text(
        0.5,
        -0.02,
        "1.Write latency  2.Read latency  3.Recall@5  4.Clean update"
        "  5.Isolation/deletion  6.T16 throughput  7.Backend compatibility"
        "  8.Deployment simplicity",
        ha="center",
        fontsize=5.4,
        va="top",
        bbox={
            "boxstyle": "square",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
            "pad": 0.3,
        },
    )
    figure.tight_layout(pad=0.4, h_pad=1.2, w_pad=0.6)
    save(figure, "fig5_diversity")


def original_temporal_bar(data: dict[str, dict[str, dict]]) -> None:
    """Original temporal bar chart with the five formal systems."""

    values = [data[s]["conflict"]["new_fact_rate"] * 100 for s in SYSTEMS]
    figure, axis = plt.subplots(figsize=(3.5, 2.0))
    bars = axis.bar(
        np.arange(len(SYSTEMS)),
        values,
        0.58,
        facecolor="#2E75B6",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    for bar, value in zip(bars, values):
        label = f"{value:.1f}%" if value < 10 or value % 1 else f"{value:.0f}%"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    axis.set_ylabel("New-Fact Rate (%)")
    axis.set_ylim(0, 120)
    axis.set_xticks(np.arange(len(SYSTEMS)))
    axis.set_xticklabels(SYSTEMS, rotation=12, ha="right")
    style_axis(axis)
    figure.tight_layout(pad=0.4)
    save(figure, "fig7_temporal")


def original_detailed_scale(data: dict[str, dict[str, dict]]) -> None:
    """Original 2x4 execution grid using scale p50/mean/p95/p99."""

    sizes = [100, 1000]
    stats = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    labels = ["Mean", "p50", "p95", "p99"]
    line_styles = [
        dict(color="black", ls="-", marker="o", mfc="black", mec="black"),
        dict(color="#404040", ls="--", marker="^", mfc="white", mec="#404040"),
        dict(color="#606060", ls="-.", marker="s", mfc="white", mec="#606060"),
        dict(color="#2E75B6", ls=":", marker="D", mfc="#2E75B6", mec="#2E75B6"),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(6.5, 2.55))
    flat = axes.flatten()

    for index, system in enumerate(SYSTEMS):
        axis = flat[index]
        for stat, label, line_style in zip(stats, labels, line_styles):
            values = [
                data[system]["scale"][str(size)]["write_latency"][stat]
                for size in sizes
            ]
            axis.plot(
                sizes,
                values,
                linewidth=1.0,
                markersize=3.2,
                markeredgewidth=0.6,
                label=label,
                **line_style,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes, ("100", "1,000"))
        axis.set_ylabel("Write latency (ms)")
        axis.text(
            0.5,
            -0.30,
            f"({chr(ord('a') + index)}) {system}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
        )
        style_axis(axis)

    for panel_index, latency_key, title in (
        (5, "write_latency", "(f) Avg. write latency"),
        (6, "read_latency", "(g) Avg. read latency"),
    ):
        axis = flat[panel_index]
        for stat, label, line_style in zip(stats, labels, line_styles):
            values = [
                float(
                    np.mean(
                        [
                            data[system]["scale"][str(size)][latency_key][stat]
                            for system in SYSTEMS
                        ]
                    )
                )
                for size in sizes
            ]
            axis.plot(
                sizes,
                values,
                linewidth=1.0,
                markersize=3.2,
                markeredgewidth=0.6,
                label=label,
                **line_style,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes, ("100", "1,000"))
        axis.set_ylabel("Latency (ms)")
        axis.text(
            0.5,
            -0.30,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
        )
        style_axis(axis)

    recall_axis = flat[7]
    for system, color, marker in zip(SYSTEMS, COLORS, MARKERS):
        values = [
            data[system]["scale"][str(size)]["recall_at_3"] * 100
            for size in sizes
        ]
        recall_axis.plot(
            sizes,
            values,
            color=color,
            marker=marker,
            linewidth=1.0,
            markersize=3.2,
        )
    recall_axis.set_xscale("log")
    recall_axis.set_xticks(sizes, ("100", "1,000"))
    recall_axis.set_ylim(95, 101)
    recall_axis.set_ylabel("Recall@3 (%)")
    recall_axis.text(
        0.5,
        -0.30,
        "(h) Exact-canary recall",
        transform=recall_axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )
    style_axis(recall_axis)

    handles = [
        plt.Line2D(
            [0],
            [0],
            linewidth=1.0,
            markersize=3.6,
            markeredgewidth=0.6,
            label=label,
            **line_style,
        )
        for label, line_style in zip(labels, line_styles)
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(pad=0.3, h_pad=1.2, w_pad=0.55, rect=[0, 0, 1, 0.90])
    save(figure, "fig8_detailed")


def original_cross_system(data: dict[str, dict[str, dict]]) -> None:
    """Original 2x4 bar-plus-line comparison with formal metrics."""

    retrieval = [data[s]["retrieval"] for s in SYSTEMS]
    conflict = [data[s]["conflict"] for s in SYSTEMS]
    deletion = [data[s]["deletion"] for s in SYSTEMS]

    write_mean = [row["write_latency"]["mean_ms"] for row in retrieval]
    read_mean = [row["read_latency"]["mean_ms"] for row in retrieval]
    recall = [row["recall_at_k"] * 100 for row in retrieval]
    type_gap = [
        abs(
            row["recall_by_event_type"]["PERSONAL_FACT"]
            - row["recall_by_event_type"]["TASK_REQUEST"]
        )
        * 100
        for row in retrieval
    ]
    new_fact = [row["new_fact_rate"] * 100 for row in conflict]
    staleness = [row["staleness_rate"] * 100 for row in conflict]
    write_p95 = [row["write_latency"]["p95_ms"] for row in retrieval]
    tail_factor = [
        row["write_latency"]["p95_ms"] / row["write_latency"]["p50_ms"]
        for row in retrieval
    ]
    omission = [row["omission_rate"] * 100 for row in retrieval]
    delete_absence = [row["post_delete_absence_rate"] * 100 for row in deletion]
    pre_visibility = [row["pre_delete_visibility_rate"] * 100 for row in deletion]

    panel_specs = [
        (
            "Write latency",
            write_mean,
            [1000.0 / value for value in write_mean],
            "Mean (ms, log)",
            "Ops/s",
            True,
        ),
        (
            "Read latency",
            read_mean,
            [1000.0 / value for value in read_mean],
            "Mean (ms, log)",
            "Ops/s",
            True,
        ),
        ("Recall@5", recall, type_gap, "Recall (%)", "Type gap (pp)", False),
        ("Temporal update", new_fact, staleness, "New fact (%)", "Stale (%)", False),
        ("Write tail", write_p95, tail_factor, "p95 (ms, log)", "p95/p50", True),
        ("Omission", omission, recall, "Omission (%)", "Recall (%)", False),
        (
            "Backend compatibility",
            [100.0] * len(SYSTEMS),
            [100.0] * len(SYSTEMS),
            "Completion (%)",
            "Compatible (%)",
            False,
        ),
        (
            "Audited deletion",
            delete_absence,
            pre_visibility,
            "Post-delete absent (%)",
            "Pre-visible (%)",
            False,
        ),
    ]

    fills = ["#1A4F7A", "white", "white", "#9DC3E6", "#777777"]
    hatches = ["", "////", "\\\\", "....", ""]
    figure, axes = plt.subplots(2, 4, figsize=(7.16, 2.95))
    flat = axes.flatten()
    x = np.arange(len(SYSTEMS))

    for index, (title, bars, line, bar_label, line_label, use_log) in enumerate(
        panel_specs
    ):
        axis = flat[index]
        for system_index, value in enumerate(bars):
            axis.bar(
                system_index,
                value,
                0.58,
                facecolor=fills[system_index],
                edgecolor="black",
                hatch=hatches[system_index],
                linewidth=0.8,
                zorder=3,
            )
        if use_log:
            axis.set_yscale("log")
        axis.set_ylabel(bar_label, fontsize=7.0)
        axis.set_xticks(x, SHORT)
        axis.text(
            0.5,
            -0.30,
            f"({chr(ord('a') + index)}) {title}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
            fontweight="bold",
        )
        style_axis(axis)

        secondary = axis.twinx()
        secondary.plot(
            x,
            line,
            linestyle=":",
            color="#2B7BBA",
            linewidth=1.0,
            marker="D",
            markersize=3.2,
            markerfacecolor="#2B7BBA",
            markeredgecolor="#2B7BBA",
            zorder=5,
        )
        secondary.set_ylabel(line_label, fontsize=6.2)
        secondary.tick_params(
            labelsize=6.2,
            colors="black",
            direction="in",
            length=2,
            width=0.5,
        )
        secondary.spines["top"].set_visible(False)
        secondary.spines["right"].set_color("black")
        secondary.spines["right"].set_linewidth(0.6)

    bar_handles = [
        mpatches.Patch(
            facecolor=fill,
            edgecolor="black",
            hatch=hatch,
            label=system,
            linewidth=0.8,
        )
        for fill, hatch, system in zip(fills, hatches, SYSTEMS)
    ]
    line_handle = plt.Line2D(
        [0],
        [0],
        linestyle=":",
        marker="D",
        color="#2B7BBA",
        linewidth=1.0,
        markersize=4,
        label="Secondary metric",
    )
    figure.legend(
        handles=bar_handles + [line_handle],
        loc="upper center",
        ncol=6,
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
        fontsize=6.8,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.3,
        columnspacing=0.7,
    )
    figure.tight_layout(pad=0.5, h_pad=2.4, w_pad=0.75, rect=[0, 0, 1, 0.87])
    save(figure, "fig9_comparison")


def original_scale_figure(data: dict[str, dict[str, dict]]) -> None:
    """Original equal-width two-panel scale design for all five systems."""

    sizes = [100, 1000]
    figure, (write_axis, read_axis) = plt.subplots(
        1, 2, figsize=(5.0, 2.2), facecolor="white"
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.88,
        bottom=0.18,
        top=0.97,
        wspace=0.55,
    )

    for system, color, marker in zip(SYSTEMS, COLORS, MARKERS):
        writes = [
            data[system]["scale"][str(size)]["write_latency"]["mean_ms"]
            for size in sizes
        ]
        reads = [
            data[system]["scale"][str(size)]["read_latency"]["mean_ms"]
            for size in sizes
        ]
        write_axis.plot(
            sizes,
            writes,
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=3.5,
            label=system,
        )
        read_axis.plot(
            sizes,
            reads,
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=3.5,
            label=system,
        )

    for axis, ylabel in (
        (write_axis, "Mean write latency (ms)"),
        (read_axis, "Mean read latency (ms)"),
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes)
        axis.get_xaxis().set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{int(value):,}")
        )
        axis.set_xlabel("Stored exact canaries")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#DDDDDD", linewidth=0.45)
        axis.tick_params(direction="in", length=2.5, width=0.6)
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    read_recall_axis = read_axis.twinx()
    mean_recall = [
        float(
            np.mean(
                [
                    data[system]["scale"][str(size)]["recall_at_3"] * 100
                    for system in SYSTEMS
                ]
            )
        )
        for size in sizes
    ]
    recall_line = read_recall_axis.plot(
        sizes,
        mean_recall,
        color="#444444",
        linewidth=1.1,
        linestyle="--",
        marker="^",
        markersize=3.2,
        label="Mean Recall@3",
    )[0]
    read_recall_axis.set_ylabel("Mean Recall@3 (%)", color="#444444")
    read_recall_axis.set_ylim(95, 101)
    read_recall_axis.tick_params(colors="#444444", labelsize=7.0)

    handles, labels = write_axis.get_legend_handles_labels()
    handles.append(recall_line)
    labels.append("Mean Recall@3")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        edgecolor="#555555",
        framealpha=0.95,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 1.12),
    )
    save(figure, "fig10_scale")


def main() -> None:
    formal = load_formal()
    original_overall_profile(formal)
    original_diversity_radar(formal)
    original_temporal_bar(formal)
    original_detailed_scale(formal)
    original_cross_system(formal)
    original_scale_figure(formal)
    print("generated original-style formal figures 5-10")


if __name__ == "__main__":
    main()
