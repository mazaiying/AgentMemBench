"""Generate paper result figures directly from the released formal JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "formal"
FIGURES = ROOT / "paper" / "figures"
DATA_META = ROOT / "data" / "memdialogue_v2_meta.json"
DATA_AUDIT = ROOT / "data" / "memdialogue_v2_audit.json"

SYSTEMS = ["Naive RAG", "Mem0", "LangMem", "Graphiti", "Letta"]
SHORT = ["Naive RAG", "Mem0", "LangMem", "Graphiti", "Letta"]
COLORS = ["#2E75B6", "#6FA8DC", "#A9C7E8", "#E69138", "#777777"]
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
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.linewidth": 0.7,
    }
)


def phase(system: str, name: str) -> dict:
    payload = json.loads((RESULTS / FILES[system][name]).read_text())
    return payload["phases"][name]


def style_axis(axis: plt.Axes, grid: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="in", width=0.6, length=2.5)
    if grid == "y":
        axis.yaxis.grid(True, color="#D9D9D9", linewidth=0.45, zorder=0)
    elif grid == "both":
        axis.grid(True, color="#E2E2E2", linewidth=0.4, zorder=0)
    axis.set_axisbelow(True)


def label_bars(axis: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.6,
        )


def save(figure: plt.Figure, stem: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES / f"{stem}.{extension}", facecolor="white")
    plt.close(figure)


def formal_summary() -> None:
    retrieval = {system: phase(system, "retrieval") for system in SYSTEMS}
    conflict = {system: phase(system, "conflict") for system in SYSTEMS}
    deletion = {system: phase(system, "deletion") for system in SYSTEMS}
    concurrency = {system: phase(system, "concurrency") for system in SYSTEMS}

    figure, axes = plt.subplots(2, 3, figsize=(7.16, 4.35))
    x = np.arange(len(SYSTEMS))

    for panel, key, title, ylabel in (
        (axes[0, 0], "write_latency", "(a) Mean write latency", "Latency (ms, log)"),
        (axes[0, 1], "read_latency", "(b) Mean read latency", "Latency (ms, log)"),
    ):
        values = [retrieval[system][key]["mean_ms"] for system in SYSTEMS]
        bars = panel.bar(
            x, values, color=COLORS, edgecolor="black", linewidth=0.55, zorder=3
        )
        panel.set_yscale("log")
        panel.set_ylabel(ylabel)
        panel.set_xticks(x, SHORT, rotation=20, ha="right")
        panel.set_title(title, fontweight="bold")
        for bar, value in zip(bars, values):
            panel.annotate(
                f"{value:,.0f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=5.4,
            )
        style_axis(panel)

    recall_groups = [
        [retrieval[system]["recall_at_k"] * 100 for system in SYSTEMS],
        [
            retrieval[system]["recall_by_event_type"]["PERSONAL_FACT"] * 100
            for system in SYSTEMS
        ],
        [
            retrieval[system]["recall_by_event_type"]["TASK_REQUEST"] * 100
            for system in SYSTEMS
        ],
    ]
    width = 0.23
    for offset, values, label, hatch in zip(
        (-width, 0, width),
        recall_groups,
        ("All", "Personal fact", "Task request"),
        ("", "///", "..."),
    ):
        axes[0, 2].bar(
            x + offset,
            values,
            width,
            label=label,
            color="#5B9BD5" if not hatch else "white",
            edgecolor="black",
            hatch=hatch,
            linewidth=0.55,
            zorder=3,
        )
    axes[0, 2].set_ylim(0, 125)
    axes[0, 2].set_ylabel("Recall@5 (%)")
    axes[0, 2].set_xticks(x, SHORT, rotation=20, ha="right")
    axes[0, 2].set_title("(c) Retrieval quality", fontweight="bold")
    axes[0, 2].legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axes[0, 2])

    categories = {
        "New only": [],
        "Dual version": [],
        "Stale only": [],
        "Unresolved": [],
    }
    for system in SYSTEMS:
        row = conflict[system]
        dual = row["dual_version_rate"]
        new_only = max(0.0, row["new_fact_rate"] - dual)
        stale = row["staleness_rate"]
        unresolved = max(0.0, 1.0 - new_only - dual - stale)
        for label, value in zip(categories, (new_only, dual, stale, unresolved)):
            categories[label].append(value * 100)
    bottom = np.zeros(len(SYSTEMS))
    fills = ("#2E75B6", "#A9C7E8", "#E69138", "#D9D9D9")
    hatches = ("", "///", "", "...")
    for (label, values), fill, hatch in zip(categories.items(), fills, hatches):
        axes[1, 0].bar(
            x,
            values,
            bottom=bottom,
            label=label,
            color=fill,
            edgecolor="black",
            hatch=hatch,
            linewidth=0.45,
            zorder=3,
        )
        bottom += np.array(values)
    axes[1, 0].set_ylim(0, 125)
    axes[1, 0].set_ylabel("Conflict outcomes (%)")
    axes[1, 0].set_xticks(x, SHORT, rotation=20, ha="right")
    axes[1, 0].set_title("(d) Temporal consistency", fontweight="bold")
    axes[1, 0].legend(frameon=False, ncol=2, loc="upper center")
    style_axis(axes[1, 0])

    pre = [deletion[system]["pre_delete_visibility_rate"] * 100 for system in SYSTEMS]
    post = [deletion[system]["post_delete_absence_rate"] * 100 for system in SYSTEMS]
    bars_pre = axes[1, 1].bar(
        x - 0.18,
        pre,
        0.36,
        color="white",
        edgecolor="black",
        hatch="///",
        linewidth=0.6,
        label="Pre-delete visible",
        zorder=3,
    )
    bars_post = axes[1, 1].bar(
        x + 0.18,
        post,
        0.36,
        color="#5B9BD5",
        edgecolor="black",
        linewidth=0.6,
        label="Post-delete absent",
        zorder=3,
    )
    axes[1, 1].set_ylim(0, 125)
    axes[1, 1].set_ylabel("Rate (%)")
    axes[1, 1].set_xticks(x, SHORT, rotation=20, ha="right")
    axes[1, 1].set_title("(e) Audited deletion", fontweight="bold")
    axes[1, 1].legend(frameon=False, ncol=2, loc="upper center")
    label_bars(axes[1, 1], bars_pre, "{:.0f}")
    label_bars(axes[1, 1], bars_post, "{:.0f}")
    style_axis(axes[1, 1])

    workers = [1, 4, 8, 16]
    for system, color, marker in zip(SYSTEMS, COLORS, MARKERS):
        values = [
            concurrency[system][str(worker)]["throughput_ops_s"]
            for worker in workers
        ]
        axes[1, 2].plot(
            workers,
            values,
            marker=marker,
            color=color,
            linewidth=1.1,
            markersize=4,
            label=system,
        )
    axes[1, 2].set_xlabel("Workers")
    axes[1, 2].set_ylabel("Throughput (ops/s)")
    axes[1, 2].set_xticks(workers)
    axes[1, 2].set_ylim(0, 33)
    axes[1, 2].set_title("(f) Concurrent writes", fontweight="bold")
    axes[1, 2].legend(frameon=False, ncol=2, loc="upper center")
    style_axis(axes[1, 2], grid="both")

    figure.tight_layout(pad=0.65, w_pad=0.75, h_pad=1.0)
    save(figure, "fig5_formal_summary")


def latency_distribution() -> None:
    retrieval = {system: phase(system, "retrieval") for system in SYSTEMS}
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    x = np.arange(len(SYSTEMS))
    width = 0.34
    for axis, key, title in (
        (axes[0], "write_latency", "(a) Write-latency distribution"),
        (axes[1], "read_latency", "(b) Read-latency distribution"),
    ):
        p50 = [retrieval[system][key]["p50_ms"] for system in SYSTEMS]
        p95 = [retrieval[system][key]["p95_ms"] for system in SYSTEMS]
        axis.bar(
            x - width / 2,
            p50,
            width,
            label="p50",
            color="#5B9BD5",
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
        axis.bar(
            x + width / 2,
            p95,
            width,
            label="p95",
            color="white",
            edgecolor="black",
            hatch="///",
            linewidth=0.55,
            zorder=3,
        )
        axis.set_yscale("log")
        axis.set_ylabel("Latency (ms, log)")
        axis.set_xticks(x, SHORT, rotation=18, ha="right")
        axis.set_title(title, fontweight="bold")
        axis.legend(frameon=False)
        style_axis(axis)
    figure.tight_layout(pad=0.5, w_pad=1.2)
    save(figure, "fig6_latency_distribution")


def scale_behavior() -> None:
    scale = {system: phase(system, "scale") for system in SYSTEMS}
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    sizes = [100, 1000]
    for axis, key, title in (
        (axes[0], "write_latency", "(a) Mean write latency"),
        (axes[1], "read_latency", "(b) Mean read latency"),
    ):
        for system, color, marker in zip(SYSTEMS, COLORS, MARKERS):
            values = [scale[system][str(size)][key]["mean_ms"] for size in sizes]
            axis.plot(
                sizes,
                values,
                marker=marker,
                color=color,
                linewidth=1.1,
                markersize=4.2,
                label=system,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes, ("100", "1,000"))
        axis.set_xlabel("Stored exact canaries")
        axis.set_ylabel("Latency (ms, log)")
        axis.set_title(title, fontweight="bold")
        style_axis(axis, grid="both")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    figure.tight_layout(pad=0.5, w_pad=1.2)
    save(figure, "fig7_scale_behavior")


def dataset_profile() -> None:
    metadata = json.loads(DATA_META.read_text())
    audit = json.loads(DATA_AUDIT.read_text())
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.15))

    event_labels = ["Task request", "Personal fact"]
    event_values = [
        metadata["event_type_counts"]["TASK_REQUEST"],
        metadata["event_type_counts"]["PERSONAL_FACT"],
    ]
    bars = axes[0].bar(
        event_labels,
        event_values,
        color=(COLORS[0], COLORS[3]),
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )
    axes[0].set_ylabel("Records")
    axes[0].set_title("(a) Event types", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=18)
    label_bars(axes[0], bars, "{:,.0f}")
    style_axis(axes[0])

    model_labels = ["Qwen2.5-7B", "Qwen2.5-14B"]
    model_values = [
        metadata["record_annotator_models"]["qwen2.5-7b-instruct"],
        metadata["record_annotator_models"]["qwen2.5-14b-instruct"],
    ]
    bars = axes[1].bar(
        model_labels,
        model_values,
        color=(COLORS[1], COLORS[2]),
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )
    axes[1].set_ylabel("Records")
    axes[1].set_title("(b) Extractor mixture", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=18)
    label_bars(axes[1], bars, "{:,.0f}")
    style_axis(axes[1])

    source_labels = ["Unique sources", "Multi-record sources"]
    source_values = [
        audit["unique_sources"],
        audit["sources_with_multiple_records"],
    ]
    bars = axes[2].bar(
        source_labels,
        source_values,
        color=(COLORS[0], COLORS[4]),
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )
    axes[2].set_ylabel("Source hashes")
    axes[2].set_title("(c) Source coverage", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=18)
    label_bars(axes[2], bars, "{:,.0f}")
    style_axis(axes[2])

    figure.tight_layout(pad=0.5, w_pad=1.0)
    save(figure, "fig8_dataset_profile")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    formal_summary()
    latency_distribution()
    scale_behavior()
    dataset_profile()
    print("generated formal-result figures 5-8")


if __name__ == "__main__":
    main()
