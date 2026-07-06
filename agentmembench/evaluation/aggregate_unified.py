"""Aggregate versioned unified benchmark JSON files across seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nested(data: dict[str, Any], *keys: str) -> float | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def flatten(result: dict[str, Any]) -> dict[str, float]:
    phases = result.get("phases", {})
    candidates = {
        "retrieval_recall_at_k": ("retrieval", "recall_at_k"),
        "retrieval_write_p50_ms": ("retrieval", "write_latency", "p50_ms"),
        "retrieval_write_p95_ms": ("retrieval", "write_latency", "p95_ms"),
        "retrieval_read_p50_ms": ("retrieval", "read_latency", "p50_ms"),
        "retrieval_read_p95_ms": ("retrieval", "read_latency", "p95_ms"),
        "conflict_new_fact_rate": ("conflict", "new_fact_rate"),
        "conflict_staleness_rate": ("conflict", "staleness_rate"),
        "isolation_leak_rate": ("isolation", "cross_user_leak_rate"),
        "deletion_audited_rate": ("deletion", "audited_deletion_rate"),
    }
    values: dict[str, float] = {}
    for name, path in candidates.items():
        value = nested(phases, *path)
        if value is not None:
            values[name] = value
    concurrency = phases.get("concurrency", {})
    for workers, row in concurrency.items():
        value = nested(row, "throughput_ops_s")
        if value is not None:
            values[f"throughput_w{workers}_ops_s"] = value
    scale = phases.get("scale", {})
    for size, row in scale.items():
        for metric, path in (
            ("recall_at_3", ("recall_at_3",)),
            ("write_p50_ms", ("write_latency", "p50_ms")),
            ("read_p50_ms", ("read_latency", "p50_ms")),
        ):
            value = nested(row, *path)
            if value is not None:
                values[f"scale_{size}_{metric}"] = value
    return values


def main() -> int:
    args = parse_args()
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    runs: list[dict[str, Any]] = []
    for path in args.inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        system = str(result["system"])
        metrics = flatten(result)
        grouped[system].append(metrics)
        runs.append(
            {
                "path": str(path),
                "system": system,
                "run_id": result.get("run_id"),
                "seed": result.get("config", {}).get("seed"),
            }
        )
    summary: dict[str, Any] = {}
    for system, rows in sorted(grouped.items()):
        names = sorted({name for row in rows for name in row})
        summary[system] = {}
        for name in names:
            values = [row[name] for row in rows if name in row]
            summary[system][name] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
    payload = {"runs": runs, "systems": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
