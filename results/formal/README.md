# Formal Result Artifact

This directory contains the unified raw outputs used for the paper. Every JSON
file records the workload arguments, local service endpoints, package versions,
hardware snapshot, latency distributions, and protocol-specific measurements.
The paper uses the seed-2027 files listed below. Older pilot outputs are not
included in the public artifact and are not cross-system evidence.

## Canonical files

| System | Retrieval | Conflict | Isolation | Deletion | Concurrency | Scale |
|---|---|---|---|---|---|---|
| Naive RAG | `naive_rag_naive_full_s2027_9170.json` | same combined file | same combined file | same combined file | same combined file | same combined file |
| Mem0 | `mem0_mem0_retrieval_s2027_9170.json` | `mem0_mem0_conflict_s2027_9170.json` | `mem0_mem0_isolation_s2027_9170.json` | `mem0_mem0_deletion_s2027_9170.json` | `mem0_mem0_concurrency_v2_s2027_9170.json` | `mem0_mem0_scale_s2027_9170.json` |
| LangMem | `langmem_langmem_retrieval_s2027_9170.json` | `langmem_langmem_conflict_s2027_9170.json` | `langmem_langmem_isolation_s2027_9170.json` | `langmem_langmem_deletion_s2027_9170.json` | `langmem_langmem_concurrency_s2027_9170.json` | `langmem_langmem_scale_s2027_9170.json` |
| Graphiti | `graphiti_graphiti_retrieval_s2027_9170.json` | `graphiti_graphiti_conflict_s2027_9170.json` | `graphiti_graphiti_isolation_s2027_9170.json` | `graphiti_graphiti_deletion_s2027_9170.json` | `graphiti_graphiti_concurrency_s2027_9170.json` | `graphiti_graphiti_scale_s2027_9170.json` |
| Letta | `letta_letta_retrieval_s2027_9170.json` | `letta_letta_conflict_s2027_9170.json` | `letta_letta_isolation_s2027_9170.json` | `letta_letta_deletion_s2027_9170.json` | `letta_letta_concurrency_s2027_9170.json` | `letta_letta_scale_s2027_9170.json` |

The Mem0 retrieval files with seeds 2028 and 2029 are additional-seed
robustness runs. `graphiti_graphiti_gate32k.json` and
`letta_letta_gate32k.json` are pre-flight context-window diagnostics; they are
not substituted for the canonical protocol files above.

## Headline seed-2027 measurements

| System | Recall@5 | New-fact rate | Cross-user leak | Audited deletion |
|---|---:|---:|---:|---:|
| Naive RAG | 0.966 | 1.000 | 0.000 | 1.000 |
| Mem0 | 0.818 | 0.900 | 0.000 | 1.000 |
| LangMem | 0.286 | 0.680 | 0.000 | 0.500 |
| Graphiti | 0.687 | 0.004 | 0.000 | 1.000 |
| Letta | 0.982 | 0.996 | 0.000 | 1.000 |

These values are convenience pointers, not separately edited results. The JSON
files remain the source of truth and include confidence intervals and
per-example retrieval decisions.

## Field semantics

- `operation_success_rate` is the fraction of `add` calls that returned without
  an exception.
- `materialization_rate` is the fraction of `add` calls that returned a
  non-empty backend memory-ID list. It is an adapter/backend return signal, not
  a retrieval validation and not the denominator used for throughput.
- `success_rate` in the concurrency records is retained as a compatibility
  alias for operation success. The Mem0 `v2` file is the corrected canonical
  concurrency output.
- Older retrieval records use `write_success_rate` for a non-empty returned-ID
  signal. Retrieval effectiveness is measured separately by `recall_at_k`.
- The observed ID-return materialization values for LLM-coupled systems are
  intentionally released without smoothing and can be non-monotonic across
  worker counts. They should be read as diagnostic behavior, not as a monotone
  scalability metric.

Naive RAG's combined output predates the additive
`operation_success_rate`/`materialization_rate` split. Its deterministic
adapter returns one record ID for every completed add, so its recorded
`success_rate` represents both conditions.

## Integrity check

From the repository root:

```bash
shasum -a 256 -c results/formal/SHA256SUMS
```

All formal runs use schema `unified-benchmark-v2`, Python 3.11.15,
Qwen2.5-14B-Instruct, bge-m3 embeddings, the package versions captured in each
JSON file, the same local model endpoints, and the 9,170-record MemDialogue
release metadata. Letta used the documented self-hosted 0.16.8 service with
the 1.12.1 Python client recorded in its result files.
