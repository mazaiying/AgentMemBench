"""Shared defaults for the local AgentMemBench reproducibility environment."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTMEMBENCH_LLM_URL", "http://127.0.0.1:18000/v1"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTMEMBENCH_LLM_MODEL", "qwen2.5-14b-instruct"
        )
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTMEMBENCH_EMBEDDING_URL", "http://127.0.0.1:18002/v1"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTMEMBENCH_EMBEDDING_MODEL", "bge-m3"
        )
    )
    embedding_dims: int = 1024
    qdrant_url: str = "http://127.0.0.1:6333"
    neo4j_uri: str = "bolt://127.0.0.1:17687"
    letta_url: str = "http://127.0.0.1:8283"
    memdialogue_path: str = "data/memdialogue_v2.jsonl"
    results_dir: str = "results/unified"
    recall_k: int = 5
    memconflict_pairs: int = 250
    isolation_users: int = 100
    scale_sizes: tuple[int, ...] = (100, 1000)
