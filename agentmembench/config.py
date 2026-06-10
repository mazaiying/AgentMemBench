"""
AgentMemBench Configuration
============================
Copy this file to config.py and fill in your API keys.
Never commit config.py to git (it is in .gitignore).

Usage:
    from agentmembench.config import Config
    cfg = Config()
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── LLM / Embedding (Alibaba DashScope, used for Qwen-based evaluation) ──
    dashscope_api_key: str = field(
        default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", "")
    )

    # ── OpenAI (optional, used by some system internals) ──
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )

    # ── Mem0 Cloud (optional) ──
    mem0_api_key: str = field(
        default_factory=lambda: os.environ.get("MEM0_API_KEY", "")
    )

    # ── Letta Cloud (optional) ──
    letta_api_key: str = field(
        default_factory=lambda: os.environ.get("LETTA_API_KEY", "")
    )

    # ── Qdrant (local, for Naive RAG) ──
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "agentmembench"

    # ── Models ──
    llm_model: str = "qwen-plus"            # LLM judge model
    embed_model: str = "text-embedding-v3"  # Embedding model

    # ── Benchmark parameters ──
    memdialogue_path: str = "data/memdialogue.jsonl"
    results_dir: str = "results"
    figures_dir: str = "paper/figures"

    # ── Evaluation defaults ──
    recall_k: int = 5
    memconflict_pairs: int = 25     # contradiction pairs per system
    isolation_users: int = 10       # namespaces for isolation test
    scale_sizes: tuple = (100, 500, 1000, 2000, 5000, 10000)

    def validate(self):
        """Check required keys are set."""
        errors = []
        if not self.dashscope_api_key:
            errors.append("DASHSCOPE_API_KEY not set")
        if errors:
            raise EnvironmentError(
                "Missing required configuration:\n  " + "\n  ".join(errors)
            )
        return True
