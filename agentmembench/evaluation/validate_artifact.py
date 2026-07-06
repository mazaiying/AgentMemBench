"""Validate the public AgentMemBench result artifact without external packages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "results" / "formal"
DATASET = ROOT / "data" / "memdialogue_v2.jsonl"
DATASET_META = ROOT / "data" / "memdialogue_v2_meta.json"
DATASET_AUDIT = ROOT / "data" / "memdialogue_v2_audit.json"
DATA_MANIFEST = ROOT / "data" / "SHA256SUMS"
RESULT_MANIFEST = FORMAL / "SHA256SUMS"

EXPECTED_JSON = {
    "graphiti_graphiti_concurrency_s2027_9170.json",
    "graphiti_graphiti_conflict_s2027_9170.json",
    "graphiti_graphiti_deletion_s2027_9170.json",
    "graphiti_graphiti_gate32k.json",
    "graphiti_graphiti_isolation_s2027_9170.json",
    "graphiti_graphiti_retrieval_s2027_9170.json",
    "graphiti_graphiti_scale_s2027_9170.json",
    "langmem_langmem_concurrency_s2027_9170.json",
    "langmem_langmem_conflict_s2027_9170.json",
    "langmem_langmem_deletion_s2027_9170.json",
    "langmem_langmem_isolation_s2027_9170.json",
    "langmem_langmem_retrieval_s2027_9170.json",
    "langmem_langmem_scale_s2027_9170.json",
    "letta_letta_concurrency_s2027_9170.json",
    "letta_letta_conflict_s2027_9170.json",
    "letta_letta_deletion_s2027_9170.json",
    "letta_letta_gate32k.json",
    "letta_letta_isolation_s2027_9170.json",
    "letta_letta_retrieval_s2027_9170.json",
    "letta_letta_scale_s2027_9170.json",
    "mem0_mem0_concurrency_v2_s2027_9170.json",
    "mem0_mem0_conflict_s2027_9170.json",
    "mem0_mem0_deletion_s2027_9170.json",
    "mem0_mem0_isolation_s2027_9170.json",
    "mem0_mem0_retrieval_s2027_9170.json",
    "mem0_mem0_retrieval_s2028_9170.json",
    "mem0_mem0_retrieval_s2029_9170.json",
    "mem0_mem0_scale_s2027_9170.json",
    "naive_rag_naive_full_s2027_9170.json",
}

PRIVATE_MARKERS = (
    "github" + "_pat_",
    "gh" + "p_",
    "/" + "home" + "/",
    "/" + "Users" + "/",
    "10" + ".77.",
    "ic" + "de",
)
PRIVATE_PATTERN = re.compile(
    "|".join(re.escape(marker) for marker in PRIVATE_MARKERS),
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        manifest[relative_path] = checksum
    return manifest


def validate_results() -> None:
    actual = {
        path.name for path in FORMAL.glob("*.json") if not path.name.startswith("._")
    }
    assert actual == EXPECTED_JSON, (
        f"formal JSON set mismatch: missing={sorted(EXPECTED_JSON - actual)}, "
        f"extra={sorted(actual - EXPECTED_JSON)}"
    )

    manifest = load_manifest(RESULT_MANIFEST)

    assert len(manifest) == len(EXPECTED_JSON)
    for name in sorted(EXPECTED_JSON):
        path = FORMAL / name
        relative = path.relative_to(ROOT).as_posix()
        assert manifest.get(relative) == sha256(path), f"checksum mismatch: {relative}"

        text = path.read_text(encoding="utf-8")
        assert not PRIVATE_PATTERN.search(text), f"private/stale marker: {relative}"
        result = json.loads(text)
        assert result["schema_version"] == "unified-benchmark-v2"
        assert result["system"] in {
            "naive_rag",
            "mem0",
            "langmem",
            "graphiti",
            "letta",
        }
        assert result["phases"], f"no completed phases: {relative}"


def validate_dataset() -> None:
    manifest = load_manifest(DATA_MANIFEST)
    for path in (DATASET, DATASET_META, DATASET_AUDIT):
        relative = path.relative_to(ROOT).as_posix()
        assert manifest.get(relative) == sha256(path), f"checksum mismatch: {relative}"

    metadata = json.loads(DATASET_META.read_text(encoding="utf-8"))
    audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    expected_records = metadata["records"]
    assert expected_records == sum(metadata["event_type_counts"].values())
    assert audit["passed"] is True and audit["violations"] == []
    assert audit["records"] == expected_records
    assert DATASET.exists(), f"missing released dataset: {DATASET.relative_to(ROOT)}"

    record_ids: set[str] = set()
    records = 0
    with DATASET.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            record_id = record["record_id"]
            assert record_id not in record_ids, f"duplicate record_id at line {line_number}"
            record_ids.add(record_id)
            assert record["source_license"] == "ODC-BY-1.0"
            assert record["memory_events"], f"no memory event at line {line_number}"
            assert all(
                event.get("release_verified") is True
                for event in record["memory_events"]
            ), f"unverified event at line {line_number}"
            records += 1
    assert records == expected_records, (
        f"dataset count mismatch: expected={expected_records}, actual={records}"
    )


def main() -> None:
    validate_results()
    validate_dataset()
    print(
        f"artifact validation passed: {len(EXPECTED_JSON)} result files, "
        f"{json.loads(DATASET_META.read_text())['records']} dataset records"
    )


if __name__ == "__main__":
    main()
