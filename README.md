# AgentMemBench

> **AgentMemBench: A Taxonomy-Driven Benchmark for Agent Memory Systems**
> 
> Zaiying Ma, Feng Zhang, Jiawei Guan, Xiaoyong Du
> 
> Renmin University of China
> 
> *IEEE ICDE 2026*

[![Paper](https://img.shields.io/badge/paper-ICDE2026-blue)](https://github.com/mazaiying/AgentMemBench)
[![Dataset](https://img.shields.io/badge/dataset-MemDialogue%2010k-green)](data/memdialogue.jsonl)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## Overview

AgentMemBench is the first taxonomy-driven benchmark for systematically evaluating heterogeneous agent memory systems. It provides:

- **A four-dimensional taxonomy (D1–D4)** classifying memory systems along Storage Backend, LLM Coupling, Memory Hierarchy, and Target Workload
- **MESA** — Memory Evaluation Standard for Agents — covering six axes: write efficiency, retrieval quality, scalability, temporal consistency, isolation/privacy, and LLM portability
- **MemDialogue** — a 10,000-sample dataset of real conversational memory events derived from LMSYS-Chat-1M

We evaluate five representative open-source systems: **Mem0**, **Graphiti**, **Naive RAG**, **LangMem**, and **Letta**.

## Key Findings

| Finding | Result |
|---------|--------|
| Write latency gap (LLM-coupled vs. uncoupled) | **11–17×** |
| Staleness rate in vector-store systems | **84–100%** |
| LLM-coupled systems failing on provider change | **3 out of 4** |
| Mem0: write-success with retrieval omission | **42% omission rate** despite 100% write-success signal |

## Repository Structure

```
AgentMemBench/
├── agentmembench/
│   ├── check_env.py                  # Environment setup check
│   ├── data/
│   │   ├── build_memdialogue.py      # Download & process LMSYS-Chat-1M
│   │   └── extend_to_10k.py          # Extend dataset to 10k samples
│   ├── evaluation/
│   │   ├── run_mem0.py               # Evaluate Mem0
│   │   ├── run_graphiti.py           # Evaluate Graphiti
│   │   ├── run_naive_rag.py          # Evaluate Naive RAG
│   │   ├── run_langmem.py            # Evaluate LangMem
│   │   ├── run_letta.py              # Evaluate Letta
│   │   ├── m1_write_efficiency.py    # M1: Write latency & throughput
│   │   ├── m1_throughput.py          # M1: QPS benchmark
│   │   ├── m2_retrieval_quality.py   # M2: Recall & omission rate
│   │   ├── m3_scalability.py         # M3: Scalability test (all systems)
│   │   ├── m3_scale_naive_rag.py     # M3: Scale experiment (Naive RAG)
│   │   ├── m4_temporal_consistency.py # M4: MemConflict protocol
│   │   ├── m4_semantic_drift.py      # M4: Semantic drift over rewrites
│   │   ├── m5_isolation.py           # M5: Cross-user isolation
│   │   ├── m5_deletion.py            # M5: GDPR deletion compliance
│   │   └── summarize.py              # Aggregate all results
│   └── visualization/
│       ├── gen_fig1.py               # Fig 1: Architecture comparison
│       ├── gen_fig2.py               # Fig 2: Motivation radar chart
│       ├── gen_fig3.py               # Fig 3: System overview
│       ├── gen_fig4.py               # Fig 4: MemConflict example
│       ├── gen_fig5.py               # Fig 5: Dataset diversity
│       ├── gen_fig6.py               # Fig 6: Write latency results
│       ├── gen_fig7.py               # Fig 7: Temporal consistency
│       ├── gen_fig8.py               # Fig 8: Detailed breakdown
│       ├── gen_fig9.py               # Fig 9: Cross-system comparison
│       └── gen_fig10.py              # Fig 10: Scale experiment
├── data/
│   ├── memdialogue.jsonl             # MemDialogue dataset (10,000 samples)
│   └── memdialogue_meta.json         # Dataset metadata & statistics
├── results/
│   ├── pilot_mem0_result.json
│   ├── pilot_graphiti_result.json
│   ├── pilot_naive_rag_result.json
│   ├── pilot_langmem_result.json
│   ├── pilot_letta_cloud_result.json
│   ├── memconflict_mem0_result.json
│   ├── memconflict_naive_rag_result.json
│   ├── memconflict_langmem_result.json
│   ├── scale_naive_rag_result.json
│   ├── isolation_mem0_result.json
│   ├── deletion_mem0_result.json
│   ├── hallucination_mem0_100_result.json
│   ├── qps_result.json
│   └── final_summary.md              # Human-readable findings summary
└── paper/
    └── figures/                      # All paper figures (PDF + PNG)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export DASHSCOPE_API_KEY="sk-..."    # Alibaba DashScope (for Qwen/embedding)
export MEM0_API_KEY="..."            # Mem0 (optional, for cloud API)
export LETTA_API_KEY="..."           # Letta Cloud (optional)
```

### 3. Check environment

```bash
python agentmembench/check_env.py
```

### 4. Run a single system evaluation

```bash
# Evaluate Mem0
python agentmembench/evaluation/run_mem0.py

# Evaluate Naive RAG
python agentmembench/evaluation/run_naive_rag.py

# Run temporal consistency test (MemConflict protocol)
python agentmembench/evaluation/m4_temporal_consistency.py

# Run isolation test
python agentmembench/evaluation/m5_isolation.py
```

### 5. Reproduce all figures

```bash
cd agentmembench/visualization/
python gen_fig1.py && python gen_fig2.py && python gen_fig3.py
python gen_fig4.py && python gen_fig5.py
python gen_fig6.py && python gen_fig7.py && python gen_fig8.py
python gen_fig9.py && python gen_fig10.py
```

## MemDialogue Dataset

**MemDialogue** is a 10,000-sample benchmark dataset constructed from LMSYS-Chat-1M via a five-stage quality filter:

1. Minimum-turn filter (≥6 turns, English)
2. Memory event classification (personal fact vs. task request)
3. Specificity check
4. Answerability verification
5. Cosine deduplication (threshold: 0.92)

**Format** (`data/memdialogue.jsonl`):
```json
{
  "id": "md_00001",
  "raw_text": "The user's name is Alice and she lives in Seattle.",
  "query": "Where does the user live?",
  "ground_truth": "Seattle",
  "event_type": "personal_fact",
  "category": "location"
}
```

Statistics: 6,800 personal facts (68%) + 3,200 task-request events (32%). Type–token ratio: 0.62 (vs. 0.41 for LoCoMo).

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
| M3 | Scalability | Latency at 100/500/1,000/5,000/10,000 facts |
| M4 | Temporal Consistency | New-fact rate, Staleness rate (%) |
| M5 | Isolation & Privacy | Cross-user leak rate, GDPR deletion completeness (%) |
| M6 | LLM Portability | Pass/Fail/Partial per LLM backend |

## Citation

```bibtex
@inproceedings{ma2026agentmembench,
  title     = {AgentMemBench: A Taxonomy-Driven Benchmark for Agent Memory Systems},
  author    = {Ma, Zaiying and Zhang, Feng and Guan, Jiawei and Du, Xiaoyong},
  booktitle = {Proceedings of the 42nd IEEE International Conference on Data Engineering (ICDE)},
  year      = {2026}
}
```

## License

Code: MIT License. Dataset (MemDialogue): CC BY 4.0.

---

*All results in `results/` are raw experiment outputs. See `results/final_summary.md` for a curated summary of the 13 key findings.*
