"""Check local dependencies and service endpoints without printing secrets."""

from __future__ import annotations

import importlib
import os
import sys

import httpx

from agentmembench.config import Config


def main() -> int:
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    config = Config()
    print(f"Python: {sys.version.split()[0]}")
    ok = True
    for module in (
        "openai",
        "mem0",
        "langmem",
        "graphiti_core",
        "letta_client",
        "qdrant_client",
        "neo4j",
    ):
        try:
            importlib.import_module(module)
            print(f"OK module: {module}")
        except ImportError:
            print(f"MISSING module: {module}")
            ok = False
    endpoints = {
        "LLM": config.llm_base_url.removesuffix("/v1") + "/health",
        "Embedding": config.embedding_base_url.removesuffix("/v1") + "/health",
        "Qdrant": config.qdrant_url + "/healthz",
    }
    with httpx.Client(timeout=5, trust_env=False) as client:
        for name, url in endpoints.items():
            try:
                response = client.get(url)
                response.raise_for_status()
                print(f"OK service: {name} ({url})")
            except Exception as exc:
                print(f"UNAVAILABLE service: {name} ({url}): {exc}")
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
