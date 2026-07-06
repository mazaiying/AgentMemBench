"""Audit a generated MemDialogue release without source conversation text."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agentmembench.data.build_memdialogue_wildchat import (
    SOURCE_DATASET,
    SOURCE_LICENSE,
    contains_direct_identifier,
    contains_injection,
    contains_sensitive_or_fictional_content,
    event_from_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def audit(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    models: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    violations: list[dict[str, Any]] = []

    def reject(line: int, record_id: str, reason: str) -> None:
        violations.append(
            {"line": line, "record_id": record_id, "reason": reason}
        )

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                event = event_from_record(record)
            except Exception as exc:
                reject(line_number, "", f"invalid_json_or_schema:{type(exc).__name__}")
                continue
            record_id = str(record.get("record_id", ""))
            if not record_id or record_id in seen_ids:
                reject(line_number, record_id, "missing_or_duplicate_record_id")
            seen_ids.add(record_id)
            if record.get("source_dataset") != SOURCE_DATASET:
                reject(line_number, record_id, "unexpected_source_dataset")
            if record.get("source_license") != SOURCE_LICENSE:
                reject(line_number, record_id, "unexpected_source_license")
            source = str(record.get("source_id", ""))
            if not source:
                reject(line_number, record_id, "missing_source_id")
            sources[source] += 1
            models[str(record.get("annotator_model", "unspecified"))] += 1
            event_type = str(event.get("event_type", ""))
            counts[event_type] += 1
            if event_type not in {"PERSONAL_FACT", "TASK_REQUEST", "UPDATE"}:
                reject(line_number, record_id, "invalid_event_type")
            fields = [
                str(event.get("raw_text", "")),
                str(event.get("query", "")),
                str(event.get("ground_truth", "")),
            ]
            if not fields[0].startswith("The user"):
                reject(line_number, record_id, "invalid_memory_prefix")
            if not re.match(r"^(what|which|where|when|how)\b", fields[1], re.I):
                reject(line_number, record_id, "invalid_query_form")
            if event_type == "TASK_REQUEST" and not (
                re.search(r"\buser\b", fields[1], re.I)
                and re.search(
                    r"\b(request|want|ask|need|require)\w*\b", fields[1], re.I
                )
            ):
                reject(line_number, record_id, "task_query_not_about_user_request")
            combined = " ".join(fields)
            if contains_direct_identifier(combined):
                reject(line_number, record_id, "direct_identifier")
            if contains_injection(combined):
                reject(line_number, record_id, "prompt_injection")
            if contains_sensitive_or_fictional_content(combined):
                reject(line_number, record_id, "sensitive_or_fictional_content")
            key = fields[0].casefold()
            if key in seen_texts:
                reject(line_number, record_id, "duplicate_normalized_memory")
            seen_texts.add(key)
            if event.get("release_verified") is not True:
                reject(line_number, record_id, "missing_release_verification")

    return {
        "records": sum(counts.values()),
        "event_type_counts": dict(counts),
        "annotator_models": dict(models),
        "unique_sources": len(sources),
        "sources_with_multiple_records": sum(value > 1 for value in sources.values()),
        "violations": violations,
        "passed": not violations,
    }


def main() -> int:
    args = parse_args()
    report = audit(args.data)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
