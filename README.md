# AgentMemBench

> **AgentMemBench: A Taxonomy-Driven Benchmark for Agent Memory Systems**
>
> *Open benchmark implementation, MemDialogue records, and raw evaluation results*

[![Dataset](https://img.shields.io/badge/dataset-MemDialogue%20v2-green)](DATASET_CARD.md)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## Overview

AgentMemBench is a taxonomy-driven benchmark for evaluating heterogeneous
agent-memory middleware under a common operational protocol. It complements
task- and trajectory-centric memory benchmarks by testing write paths, update
semantics, isolation, deletion, scalability, and backend portability. It
provides:

- **A four-dimensional taxonomy (D1–D4)** classifying memory systems along Storage Backend, LLM Coupling, Memory Hierarchy, and Target Workload
- **MESA** — Memory Evaluation Standard for Agents — covering six axes: write efficiency, retrieval quality, scalability, temporal consistency, isolation/privacy, and LLM portability
- **MemDialogue** — 9,170 release-verified memory events derived from the
  redistributable WildChat-4.8M corpus

We evaluate five representative open-source systems: **Mem0**, **Graphiti**, **Naive RAG**, **LangMem**, and **Letta**.

## Evaluation status

The released artifact combines pinned core-system versions, bounded utility
dependencies, and unified JSON files containing workload parameters, model
endpoints, per-example retrieval decisions, latency distributions, and
bootstrap confidence intervals.
Numerical claims in the paper are generated from `results/formal/`; older
pilot outputs are intentionally excluded from the public artifact. The
[formal-results index](results/formal/README.md) identifies the canonical files,
additional-seed runs, diagnostic runs, and metric semantics.

## Repository Structure

```
AgentMemBench/
├── agentmembench/
│   ├── check_env.py                  # Environment setup check
│   ├── data/
│   │   └── build_memdialogue_wildchat.py # Rebuild MemDialogue from WildChat-4.8M
│   ├── evaluation/
│   │   ├── unified_benchmark.py       # Versioned common workload harness
│   │   ├── aggregate_unified.py       # Aggregate versioned JSON outputs
│   │   └── validate_artifact.py       # Dependency-free release audit
│   └── visualization/
│       ├── gen_fig1.py               # Fig 1: Architecture comparison
│       ├── gen_fig2.py               # Fig 2: Motivation radar chart
│       ├── gen_fig3.py               # Fig 3: System overview
│       ├── gen_fig4.py               # Fig 4: MemConflict example
│       └── gen_formal_figures.py      # Figs 5–10: Original-style formal plots
├── data/
│   ├── SHA256SUMS                    # Dataset integrity manifest
│   ├── memdialogue_v2.jsonl          # Release-verified memory events
│   ├── memdialogue_v2_meta.json      # Provenance, filters, and statistics
│   └── memdialogue_v2_audit.json     # Deterministic release-audit report
├── results/
│   ├── formal/                       # Paper results for all five systems
│   │   ├── README.md                 # Result index and metric semantics
│   │   ├── SHA256SUMS                # Integrity manifest
│   │   └── *_s2027_9170.json         # Seed-2027 unified result files
├── DATASET_CARD.md
├── DATA_USE.md
└── paper/
    └── figures/                      # Ten paper figures (PDF)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure local services

```bash
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

# Expected defaults:
# LLM:       http://127.0.0.1:18000/v1
# Embedding: http://127.0.0.1:18002/v1
# Qdrant:    http://127.0.0.1:6333
# Neo4j:     bolt://127.0.0.1:17687
```

Use the same local LLM and embedding endpoints for every system. Do not mix
cloud latency with local latency in a single comparison. Port numbers are
deployment-local; the released JSON files record the exact endpoint arguments.

### 3. Run the environment check

```bash
python agentmembench/check_env.py
```

### 4. Validate the released artifact

```bash
python -m agentmembench.evaluation.validate_artifact
```

This dependency-free check verifies the exact formal-result set and checksums,
the absence of stale/private machine markers, the dataset record count, unique
record IDs, source license, and per-event release-verification flags.

### 5. Run the unified benchmark

```bash
python -m agentmembench.evaluation.unified_benchmark \
  --system naive_rag \
  --data data/memdialogue_v2.jsonl \
  --retrieval-records 1000 \
  --seed 2027 \
  --run-id formal_seed2027
```

Valid system names are `naive_rag`, `mem0`, `langmem`, `graphiti`, and
`letta`. Letta requires a separately running self-hosted server; the unified
adapter does not silently fall back to Letta Cloud.

### 6. Rebuild MemDialogue

```bash
python -m agentmembench.data.build_memdialogue_wildchat \
  --target 10000 \
  --revision c827c6df8fcf008219ffaffa4d1dd77491099367 \
  --output data/memdialogue_v2.jsonl \
  --meta data/memdialogue_v2_meta.json

python -m agentmembench.data.audit_memdialogue \
  data/memdialogue_v2.jsonl \
  --report data/memdialogue_v2_audit.json
```

See `DATASET_CARD.md` before rebuilding or redistributing the data.

### 7. Reproduce paper figures

```bash
cd agentmembench/visualization/
python gen_fig1.py && python gen_fig2.py && python gen_fig3.py
python gen_fig4.py && python gen_formal_figures.py
```

Figures 5–10 retain the layouts of the original paper (grouped profiles,
radars, temporal bars, scale grids, and cross-system panels) while regenerating
all measured values directly from `results/formal/`. The only non-measured
profile axis is the documented taxonomy attribute `deployment simplicity`.

## MemDialogue Dataset

**MemDialogue** is a 9,170-record benchmark dataset constructed from the
ODC-By-licensed WildChat-4.8M release via a reproducible quality pipeline:

1. English, multi-turn, non-toxic source filtering
2. Signal-based candidate selection and event-type quotas
3. Deterministic JSON extraction with a pinned local model
4. Direct-identifier, sensitive-content, and prompt-injection rejection
5. Independent release verification and exact-text deduplication

Only normalized model-generated records and source hashes are distributed;
upstream conversation turns are not copied into this repository. The pinned
source revision, per-record annotator model/backend, prompt version, filters,
rejection counts, and final type counts are recorded in
`data/memdialogue_v2_meta.json`.

**Format** (`data/memdialogue_v2.jsonl`):
```json
{
  "session_id": "wildchat_<source_hash>",
  "source_dataset": "allenai/WildChat-4.8M",
  "source_license": "ODC-BY-1.0",
  "annotator_model": "<local-model-id>",
  "prompt_version": "memdialogue-v2.x",
  "memory_events": [{
    "turn_idx": 2,
    "event_type": "PERSONAL_FACT",
    "raw_text": "The user prefers concise technical explanations.",
    "query": "How does the user prefer technical explanations?",
    "ground_truth": "concise",
    "evidence_turn_indices": [2],
    "release_verified": true
  }]
}
```

See `DATASET_CARD.md`, `DATA_USE.md`, and
`data/memdialogue_v2_meta.json` for construction details and release
statistics.

## Taxonomy

| System | D1 Storage | D2 LLM Coupling | D3 Hierarchy | D4 Target |
|--------|-----------|-----------------|--------------|-----------|
| Naive RAG | Vector | None | Flat | General |
| Mem0 | Vector | Strong | Flat | Personal |
| LangMem | Vector | Strong | Flat | Personal |
| Graphiti | Graph | Strong | Flat | Enterprise |
| Letta | Hierarchical | Strong | 2-Tier | MAS |

## MESA Evaluation Axes

| Axis | Name | Key Metric |
|------|------|-----------|
| M1 | Write Efficiency | Write/read latency (ms), QPS |
| M2 | Retrieval Quality | Recall rate, Omission rate (%) |
| M3 | Scalability | Latency and exact-canary recall at 100/1,000 facts |
| M4 | Temporal Consistency | New-fact rate, Staleness rate (%) |
| M5 | Isolation & Privacy | Cross-user leak rate, GDPR deletion completeness (%) |
| M6 | LLM Portability | Pass/Fail/Partial per LLM backend |

## Citation

```bibtex
@misc{ma2026agentmembench,
  title     = {AgentMemBench: A Taxonomy-Driven Benchmark for Agent Memory Systems},
  author    = {Zaiying Ma and Jiawei Guan and Xiaoyu Zhang and Xiaofeng Chen and Feng Zhang},
  year      = {2026},
  howpublished = {\url{https://github.com/mazaiying/AgentMemBench}}
}
```

## License

Code is released under the MIT License. MemDialogue is a derived database from
WildChat-4.8M and is distributed under ODC-By 1.0 with the attribution and use
conditions documented in [`DATA_USE.md`](DATA_USE.md).

---

*All results in `results/` are raw experiment outputs. See
[`results/formal/README.md`](results/formal/README.md) for the canonical
paper-result index and interpretation notes.*
