"""Unified, version-pinned benchmark harness for AgentMemBench.

The harness currently provides identical workloads for Naive RAG and Mem0.
Additional adapters can implement the same five-method interface without
changing workload generation or metric computation.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from openai import AsyncOpenAI, OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


@dataclass(frozen=True)
class Config:
    llm_base_url: str
    llm_model: str
    embedding_base_url: str
    embedding_model: str
    embedding_dims: int
    qdrant_url: str
    neo4j_uri: str
    letta_url: str
    output_dir: Path
    history_dir: Path
    top_k: int
    seed: int


class Adapter(Protocol):
    name: str

    def reset(self) -> None: ...
    def add(self, text: str, user_id: str) -> list[str]: ...
    def search(self, query: str, user_id: str, limit: int) -> list[str]: ...
    def delete(self, memory_ids: list[str]) -> None: ...
    def close(self) -> None: ...


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q))


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def bootstrap_mean_ci(
    values: list[float] | list[bool],
    seed: int = 2027,
    samples: int = 2000,
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        estimates[index] = rng.choice(array, size=len(array), replace=True).mean()
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def normalize_collection(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", value.lower())[:60]


class NaiveRAGAdapter:
    name = "naive_rag"

    def __init__(self, config: Config, collection: str):
        self.config = config
        self.collection = collection
        self.embedding = OpenAI(api_key="local", base_url=config.embedding_base_url)
        self.qdrant = QdrantClient(url=config.qdrant_url)

    def reset(self) -> None:
        if self.qdrant.collection_exists(self.collection):
            self.qdrant.delete_collection(self.collection)
        self.qdrant.create_collection(
            self.collection,
            vectors_config=VectorParams(
                size=self.config.embedding_dims,
                distance=Distance.COSINE,
            ),
        )

    def _embed(self, text: str) -> list[float]:
        return self.embedding.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        ).data[0].embedding

    def add(self, text: str, user_id: str) -> list[str]:
        memory_id = str(uuid.uuid4())
        self.qdrant.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=self._embed(text),
                    payload={"text": text, "user_id": user_id},
                )
            ],
            wait=True,
        )
        return [memory_id]

    def search(self, query: str, user_id: str, limit: int) -> list[str]:
        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=self._embed(query),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
            limit=limit,
        )
        return [str(point.payload.get("text", "")) for point in result.points]

    def delete(self, memory_ids: list[str]) -> None:
        if memory_ids:
            self.qdrant.delete(
                collection_name=self.collection,
                points_selector=memory_ids,
                wait=True,
            )

    def close(self) -> None:
        if self.qdrant.collection_exists(self.collection):
            self.qdrant.delete_collection(self.collection)


class Mem0Adapter:
    name = "mem0"

    def __init__(self, config: Config, collection: str):
        from mem0 import Memory

        self.config = config
        self.collection = collection
        history = config.history_dir / f"{collection}.sqlite"
        history.parent.mkdir(parents=True, exist_ok=True)
        history.unlink(missing_ok=True)
        memory_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.llm_model,
                    "api_key": "local",
                    "openai_base_url": config.llm_base_url,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 256,
                    "is_reasoning_model": False,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": config.embedding_model,
                    "api_key": "local",
                    "openai_base_url": config.embedding_base_url,
                    "embedding_dims": config.embedding_dims,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection,
                    "embedding_model_dims": config.embedding_dims,
                    "url": config.qdrant_url,
                },
            },
            "history_db_path": str(history),
            "version": "v1.1",
        }
        self.memory = Memory.from_config(memory_config)
        self.qdrant = QdrantClient(url=config.qdrant_url)

    def reset(self) -> None:
        for collection in self.qdrant.get_collections().collections:
            if collection.name == self.collection or collection.name.startswith(
                f"{self.collection}_"
            ):
                self.qdrant.delete_collection(collection.name)
        self.memory.vector_store.create_col(self.config.embedding_dims, False)

    def add(self, text: str, user_id: str) -> list[str]:
        response = self.memory.add(text, user_id=user_id)
        return [
            str(item["id"])
            for item in response.get("results", [])
            if item.get("id")
        ]

    def search(self, query: str, user_id: str, limit: int) -> list[str]:
        response = self.memory.search(
            query,
            filters={"user_id": user_id},
            limit=limit,
        )
        return [
            str(item.get("memory", ""))
            for item in response.get("results", [])
        ]

    def delete(self, memory_ids: list[str]) -> None:
        for memory_id in memory_ids:
            self.memory.delete(memory_id)

    def close(self) -> None:
        for collection in self.qdrant.get_collections().collections:
            if collection.name == self.collection or collection.name.startswith(
                f"{self.collection}_"
            ):
                self.qdrant.delete_collection(collection.name)


class LangMemAdapter:
    name = "langmem"

    def __init__(self, config: Config, collection: str):
        self.config = config
        self.collection = collection
        self._build()

    def _build(self) -> None:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager

        llm = ChatOpenAI(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key="local",
            temperature=0,
            max_tokens=256,
        )
        embedder = OpenAIEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.embedding_base_url,
            api_key="local",
            check_embedding_ctx_length=False,
        )
        self.store = InMemoryStore(
            index={"dims": self.config.embedding_dims, "embed": embedder}
        )
        self.manager = create_memory_store_manager(
            llm,
            namespace=("memories", "{langgraph_user_id}"),
            store=self.store,
            enable_deletes=True,
        )

    def reset(self) -> None:
        self._build()

    def add(self, text: str, user_id: str) -> list[str]:
        result = self.manager.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"langgraph_user_id": user_id}},
        )
        return [str(item["key"]) for item in result if item.get("key")]

    def search(self, query: str, user_id: str, limit: int) -> list[str]:
        items = self.store.search(
            ("memories", user_id),
            query=query,
            limit=limit,
        )
        memories = []
        for item in items:
            value = item.value
            content = value.get("content", value) if isinstance(value, dict) else value
            if isinstance(content, dict):
                content = content.get("content", content)
            memories.append(str(content))
        return memories

    def delete(self, memory_ids: list[str]) -> None:
        for namespace in self.store.list_namespaces(prefix=("memories",)):
            for memory_id in memory_ids:
                self.store.delete(namespace, memory_id)

    def close(self) -> None:
        return None


class AsyncLoopRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


class GraphitiAdapter:
    name = "graphiti"

    def __init__(self, config: Config, collection: str):
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        self.config = config
        self.collection = collection
        self.driver = Neo4jDriver(config.neo4j_uri, "neo4j", "")
        self.runner = AsyncLoopRunner()
        self.graphiti = self.runner.run(self._build())

    async def _build(self):
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.embedder.openai import (
            OpenAIEmbedder,
            OpenAIEmbedderConfig,
        )
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import (
            OpenAIGenericClient,
        )

        llm_config = LLMConfig(
            api_key="local",
            model=self.config.llm_model,
            small_model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            temperature=0,
            max_tokens=512,
        )
        llm = OpenAIGenericClient(
            config=llm_config,
            max_tokens=512,
            structured_output_mode="json_schema",
        )
        graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=llm,
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key="local",
                    embedding_model=self.config.embedding_model,
                    embedding_dim=self.config.embedding_dims,
                    base_url=self.config.embedding_base_url,
                )
            ),
            cross_encoder=OpenAIRerankerClient(
                config=llm_config,
                client=llm.client,
            ),
            store_raw_episode_content=False,
            max_coroutines=16,
        )
        await graphiti.build_indices_and_constraints()
        return graphiti

    def reset(self) -> None:
        self.runner.run(self.driver.execute_query("MATCH (n) DETACH DELETE n"))

    def add(self, text: str, user_id: str) -> list[str]:
        from datetime import datetime, timezone

        result = self.runner.run(
            self.graphiti.add_episode(
                name=f"memory_{uuid.uuid4()}",
                episode_body=text,
                source_description="AgentMemBench unified workload",
                reference_time=datetime.now(timezone.utc),
                group_id=user_id,
                custom_extraction_instructions=(
                    "Treat 'The user' as a persistent entity. Preserve explicit "
                    "preferences, facts, and requested tasks as relations."
                ),
            )
        )
        return [str(edge.uuid) for edge in result.edges]

    def search(self, query: str, user_id: str, limit: int) -> list[str]:
        edges = self.runner.run(
            self.graphiti.search(
                query,
                group_ids=[user_id],
                num_results=limit,
            )
        )
        return [str(edge.fact) for edge in edges]

    def delete(self, memory_ids: list[str]) -> None:
        if memory_ids:
            self.runner.run(
                self.driver.execute_query(
                    "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $ids DELETE e",
                    ids=memory_ids,
                )
            )

    def close(self) -> None:
        self.runner.run(self.graphiti.close())
        self.runner.close()


class LettaAdapter:
    """Self-hosted Letta adapter using one isolated agent per benchmark user."""

    name = "letta"

    def __init__(self, config: Config, collection: str):
        from letta_client import Letta

        self.config = config
        self.collection = collection
        self.client = Letta(
            base_url=config.letta_url,
            api_key=os.environ.get("LETTA_API_KEY") or None,
            timeout=300.0,
            max_retries=1,
        )
        self.agents: dict[str, str] = {}
        self.agent_users: dict[str, str] = {}
        self.lock = threading.Lock()

    def _create_agent(self, user_id: str) -> str:
        agent = self.client.agents.create(
            name=normalize_collection(
                f"{self.collection}_{user_id}_{uuid.uuid4().hex[:8]}"
            ),
            memory_blocks=[
                {
                    "label": "human",
                    "value": (
                        "This block stores durable facts and requests for one "
                        "isolated AgentMemBench user."
                    ),
                },
                {
                    "label": "persona",
                    "value": (
                        "You are a memory service. Retain user-provided facts "
                        "and answer later recall questions concisely."
                    ),
                },
            ],
            model=os.environ.get(
                "LETTA_MODEL",
                f"vllm/{self.config.llm_model}",
            ),
            embedding_config={
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": os.environ.get(
                    "LETTA_EMBEDDING_BASE_URL",
                    self.config.embedding_base_url,
                ),
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dims,
                "embedding_chunk_size": 300,
            },
        )
        agent_id = str(agent.id)
        self.agents[user_id] = agent_id
        self.agent_users[agent_id] = user_id
        return agent_id

    def _agent_for_user(self, user_id: str) -> str:
        with self.lock:
            agent_id = self.agents.get(user_id)
            if agent_id is None:
                agent_id = self._create_agent(user_id)
            return agent_id

    @staticmethod
    def _response_text(response: Any) -> str:
        texts: list[str] = []
        for message in getattr(response, "messages", []):
            if getattr(message, "message_type", "") != "assistant_message":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(str(text))
        return "\n".join(texts)

    def reset(self) -> None:
        for agent_id in list(self.agent_users):
            try:
                self.client.agents.delete(agent_id=agent_id)
            except Exception:
                pass
        self.agents.clear()
        self.agent_users.clear()

    def add(self, text: str, user_id: str) -> list[str]:
        agent_id = self._agent_for_user(user_id)
        self.client.agents.messages.create(
            agent_id=agent_id,
            input=f"Remember this durable user memory: {text}",
        )
        return [agent_id]

    def search(self, query: str, user_id: str, limit: int) -> list[str]:
        del limit
        agent_id = self._agent_for_user(user_id)
        response = self.client.agents.messages.create(
            agent_id=agent_id,
            input=(
                "Answer using the durable user memories from this agent. "
                f"If no memory supports an answer, say unknown. Query: {query}"
            ),
        )
        text = self._response_text(response)
        return [text] if text else []

    def delete(self, memory_ids: list[str]) -> None:
        for agent_id in set(memory_ids):
            user_id = self.agent_users.pop(agent_id, None)
            if user_id is None:
                continue
            self.agents.pop(user_id, None)
            self.client.agents.delete(agent_id=agent_id)

    def close(self) -> None:
        self.reset()
        self.client.close()


def make_adapter(system: str, config: Config, collection: str) -> Adapter:
    if system == "naive_rag":
        return NaiveRAGAdapter(config, collection)
    if system == "mem0":
        return Mem0Adapter(config, collection)
    if system == "langmem":
        return LangMemAdapter(config, collection)
    if system == "graphiti":
        return GraphitiAdapter(config, collection)
    if system == "letta":
        return LettaAdapter(config, collection)
    raise ValueError(f"Unknown system: {system}")


def load_records(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_sources: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source_id") or row.get("session_id") or "")
            if source and source in seen_sources:
                continue
            if source:
                seen_sources.add(source)
            event = row.get("memory_event") or row["memory_events"][0]
            by_type[event["event_type"]].append(
                {
                    "text": event["raw_text"],
                    "query": event["query"],
                    "answer": event["ground_truth"],
                    "event_type": event["event_type"],
                    "source_id": source,
                }
            )
    if not by_type:
        raise ValueError(f"No records found in {path}")
    rng = np.random.default_rng(seed)
    for values in by_type.values():
        rng.shuffle(values)
    event_types = sorted(by_type)
    base, remainder = divmod(limit, len(event_types))
    selected: list[dict[str, Any]] = []
    for index, event_type in enumerate(event_types):
        quota = base + int(index < remainder)
        selected.extend(by_type[event_type][:quota])
    if len(selected) < limit:
        already = {item["source_id"] for item in selected}
        remaining = [
            item
            for values in by_type.values()
            for item in values
            if item["source_id"] not in already
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])
    if len(selected) < limit:
        raise ValueError(
            f"Requested {limit} source-unique records, but {path} contains "
            f"{len(selected)} after stratification"
        )
    rng.shuffle(selected)
    return selected


JUDGE_PROMPT = """Decide whether at least one retrieved memory supports the
reference answer to the query. Judge semantic equivalence, not exact wording.
Return JSON only: {{"hit": true}} or {{"hit": false}}.

Query: {query}
Reference answer: {answer}
Retrieved memories:
{retrieved}
"""


async def judge_retrievals(
    config: Config,
    rows: list[dict[str, Any]],
    concurrency: int = 32,
) -> list[bool]:
    client = AsyncOpenAI(api_key="local", base_url=config.llm_base_url)
    semaphore = asyncio.Semaphore(concurrency)

    async def judge(row: dict[str, Any]) -> bool:
        if not row["retrieved"]:
            return False
        prompt = JUDGE_PROMPT.format(
            query=row["query"],
            answer=row["answer"],
            retrieved="\n".join(f"- {item}" for item in row["retrieved"]),
        )
        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=config.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=32,
                        response_format={"type": "json_object"},
                    )
                    parsed = json.loads(response.choices[0].message.content or "{}")
                    return parsed.get("hit") is True
                except Exception:
                    if attempt == 2:
                        return False
                    await asyncio.sleep(2**attempt)
        return False

    return await asyncio.gather(*(judge(row) for row in rows))


def run_retrieval(
    adapter: Adapter,
    config: Config,
    records: list[dict[str, Any]],
    group_size: int,
) -> dict[str, Any]:
    adapter.reset()
    write_latencies: list[float] = []
    write_success = 0
    for index, record in enumerate(records):
        started = time.perf_counter()
        ids = adapter.add(record["text"], f"user_{index // group_size:05d}")
        write_latencies.append((time.perf_counter() - started) * 1000)
        write_success += int(bool(ids))

    evaluated = []
    read_latencies: list[float] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        retrieved = adapter.search(
            record["query"],
            f"user_{index // group_size:05d}",
            config.top_k,
        )
        read_latencies.append((time.perf_counter() - started) * 1000)
        evaluated.append({**record, "retrieved": retrieved})

    hits = asyncio.run(judge_retrievals(config, evaluated))
    by_type: dict[str, list[bool]] = defaultdict(list)
    for row, hit in zip(evaluated, hits):
        by_type[row["event_type"]].append(hit)
    return {
        "records": len(records),
        "group_size": group_size,
        "write_success_rate": write_success / len(records),
        "write_materialization_rate": write_success / len(records),
        "recall_at_k": sum(hits) / len(hits),
        "recall_at_k_95ci": bootstrap_mean_ci(hits),
        "omission_rate": 1 - (sum(hits) / len(hits)),
        "recall_by_event_type": {
            key: sum(values) / len(values) for key, values in sorted(by_type.items())
        },
        "details": [
            {
                "index": index,
                "event_type": row["event_type"],
                "source_id": row["source_id"],
                "memory": row["text"],
                "query": row["query"],
                "reference_answer": row["answer"],
                "hit": hit,
                "retrieved": row["retrieved"],
                "write_latency_ms": write_latencies[index],
                "read_latency_ms": read_latencies[index],
            }
            for index, (row, hit) in enumerate(zip(evaluated, hits))
        ],
        "write_latency": latency_summary(write_latencies),
        "read_latency": latency_summary(read_latencies),
        "write_mean_95ci_ms": bootstrap_mean_ci(write_latencies),
        "read_mean_95ci_ms": bootstrap_mean_ci(read_latencies),
    }


CONFLICT_TEMPLATES = (
    ("location", "The user currently lives in {old}.", "The user has moved and now lives in {new}.", "Where does the user currently live?"),
    ("role", "The user works as a {old}.", "The user changed jobs and now works as a {new}.", "What is the user's current job?"),
    ("preference", "The user prefers {old}.", "The user's preference changed; they now prefer {new}.", "What does the user currently prefer?"),
    ("status", "The project status is {old}.", "The project status has changed to {new}.", "What is the project's current status?"),
    ("numeric", "The project budget is {old} dollars.", "The updated project budget is {new} dollars.", "What is the current project budget?"),
)


def run_conflict(adapter: Adapter, pairs: int) -> dict[str, Any]:
    adapter.reset()
    new_hits = stale_hits = dual_hits = 0
    write_latencies: list[float] = []
    read_latencies: list[float] = []
    for index in range(pairs):
        category, old_t, new_t, query = CONFLICT_TEMPLATES[index % len(CONFLICT_TEMPLATES)]
        old = f"OLD_{category}_{index:04d}"
        new = f"NEW_{category}_{index:04d}"
        user_id = f"conflict_{index:05d}"
        for text in (old_t.format(old=old), new_t.format(new=new)):
            started = time.perf_counter()
            adapter.add(text, user_id)
            write_latencies.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        retrieved = "\n".join(adapter.search(query, user_id, 1))
        read_latencies.append((time.perf_counter() - started) * 1000)
        has_new = new.casefold() in retrieved.casefold()
        has_old = old.casefold() in retrieved.casefold()
        new_hits += int(has_new)
        stale_hits += int(has_old and not has_new)
        dual_hits += int(has_old and has_new)
    return {
        "pairs": pairs,
        "new_fact_rate": new_hits / pairs,
        "staleness_rate": stale_hits / pairs,
        "dual_version_rate": dual_hits / pairs,
        "write_latency": latency_summary(write_latencies),
        "read_latency": latency_summary(read_latencies),
    }


def run_isolation(adapter: Adapter, users: int, facts_per_user: int) -> dict[str, Any]:
    adapter.reset()
    canaries: dict[str, str] = {}
    for user_index in range(users):
        user_id = f"isolation_{user_index:05d}"
        canary = f"PRIVATE_CANARY_{user_index:05d}"
        canaries[user_id] = canary
        for fact_index in range(facts_per_user):
            adapter.add(
                f"The user's private project code is {canary}_{fact_index:02d}.",
                user_id,
            )
    leaks = 0
    queries = 0
    for user_index in range(users):
        requester = f"isolation_{user_index:05d}"
        target = f"isolation_{(user_index + 1) % users:05d}"
        retrieved = "\n".join(
            adapter.search("What is the private project code?", requester, 5)
        )
        leaks += int(canaries[target].casefold() in retrieved.casefold())
        queries += 1
    return {
        "users": users,
        "facts_per_user": facts_per_user,
        "cross_user_queries": queries,
        "cross_user_leak_rate": leaks / queries,
    }


def run_deletion(adapter: Adapter, records: int) -> dict[str, Any]:
    adapter.reset()
    visible_before = absent_after = visible_then_absent = 0
    for index in range(records):
        user_id = f"delete_{index:05d}"
        canary = f"ERASURE_CANARY_{index:05d}"
        memory_ids = adapter.add(
            f"The user's private erasure test code is {canary}.",
            user_id,
        )
        before = "\n".join(adapter.search("What is the erasure test code?", user_id, 5))
        was_visible = canary.casefold() in before.casefold()
        if was_visible:
            visible_before += 1
        adapter.delete(memory_ids)
        after = "\n".join(adapter.search("What is the erasure test code?", user_id, 5))
        is_absent = canary.casefold() not in after.casefold()
        if is_absent:
            absent_after += 1
        if was_visible and is_absent:
            visible_then_absent += 1
    return {
        "records": records,
        "pre_delete_visibility_rate": visible_before / records,
        "post_delete_absence_rate": absent_after / records,
        "audited_deletion_rate": (
            visible_then_absent / visible_before if visible_before else None
        ),
    }


def run_concurrency(
    adapter: Adapter,
    records: int,
    workers: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for worker_count in workers:
        adapter.reset()
        latencies: list[float] = []
        operation_successes = 0
        materializations = 0
        errors: list[str] = []

        def write_one(index: int) -> tuple[float, bool, bool, str | None]:
            started = time.perf_counter()
            try:
                ids = adapter.add(
                    f"The user has benchmark preference "
                    f"TOKEN_{worker_count}_{index:06d}.",
                    f"throughput_{index:06d}",
                )
                return (
                    (time.perf_counter() - started) * 1000,
                    True,
                    bool(ids),
                    None,
                )
            except Exception as error:
                return (
                    (time.perf_counter() - started) * 1000,
                    False,
                    False,
                    f"{type(error).__name__}: {error}",
                )

        wall_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(write_one, index) for index in range(records)]
            for future in as_completed(futures):
                latency, operation_success, materialized, error = future.result()
                latencies.append(latency)
                operation_successes += int(operation_success)
                materializations += int(materialized)
                if error is not None and len(errors) < 20:
                    errors.append(error)
        wall_seconds = time.perf_counter() - wall_started
        results[str(worker_count)] = {
            "records": records,
            "workers": worker_count,
            "success_rate": operation_successes / records,
            "operation_success_rate": operation_successes / records,
            "materialization_rate": materializations / records,
            "error_count": records - operation_successes,
            "error_examples": errors,
            "throughput_ops_s": records / wall_seconds,
            "latency": latency_summary(latencies),
        }
    return results


def run_scale(
    adapter: Adapter,
    scales: list[int],
    read_queries: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for scale in scales:
        adapter.reset()
        write_latencies: list[float] = []
        memory_ids: list[list[str]] = []
        for index in range(scale):
            token = f"SCALE_TOKEN_{scale}_{index:07d}"
            started = time.perf_counter()
            ids = adapter.add(
                f"The user's scale-test project code is {token}.",
                f"scale_user_{index:07d}",
            )
            write_latencies.append((time.perf_counter() - started) * 1000)
            memory_ids.append(ids)
        query_indices = np.linspace(
            0,
            scale - 1,
            num=min(read_queries, scale),
            dtype=int,
        ).tolist()
        read_latencies: list[float] = []
        hits: list[bool] = []
        for index in query_indices:
            token = f"SCALE_TOKEN_{scale}_{index:07d}"
            started = time.perf_counter()
            retrieved = "\n".join(
                adapter.search(
                    "What is the scale-test project code?",
                    f"scale_user_{index:07d}",
                    3,
                )
            )
            read_latencies.append((time.perf_counter() - started) * 1000)
            hits.append(token.casefold() in retrieved.casefold())
        results[str(scale)] = {
            "stored_records": scale,
            "write_success_rate": sum(bool(ids) for ids in memory_ids) / scale,
            "write_materialization_rate": (
                sum(bool(ids) for ids in memory_ids) / scale
            ),
            "recall_at_3": sum(hits) / len(hits),
            "recall_at_3_95ci": bootstrap_mean_ci(hits),
            "write_latency": latency_summary(write_latencies),
            "read_latency": latency_summary(read_latencies),
            "write_mean_95ci_ms": bootstrap_mean_ci(write_latencies),
            "read_mean_95ci_ms": bootstrap_mean_ci(read_latencies),
        }
    return results


def run_warmup(adapter: Adapter, writes: int) -> None:
    if writes <= 0:
        return
    adapter.reset()
    for index in range(writes):
        adapter.add(
            f"The user prefers warmup token WARMUP_{index:04d}.",
            "warmup_user",
        )
    adapter.search("What warmup token does the user prefer?", "warmup_user", 3)
    adapter.reset()


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for package in (
        "mem0ai",
        "langmem",
        "graphiti-core",
        "letta-client",
        "openai",
        "qdrant-client",
        "numpy",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()
    except Exception:
        gpu = []
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpus": gpu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        choices=("naive_rag", "mem0", "langmem", "graphiti", "letta"),
        required=True,
    )
    parser.add_argument(
        "--phases",
        default="retrieval,conflict,isolation,deletion,concurrency,scale",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-records", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--conflict-pairs", type=int, default=250)
    parser.add_argument("--isolation-users", type=int, default=100)
    parser.add_argument("--isolation-facts", type=int, default=5)
    parser.add_argument("--deletion-records", type=int, default=200)
    parser.add_argument("--concurrency-records", type=int, default=200)
    parser.add_argument("--workers", default="1,4,8,16")
    parser.add_argument("--scales", default="100,1000")
    parser.add_argument("--scale-read-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--warmup-writes", type=int, default=5)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-14b-instruct")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:18002/v1")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--embedding-dims", type=int, default=1024)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:17687")
    parser.add_argument("--letta-url", default="http://127.0.0.1:8283")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unified"))
    parser.add_argument("--history-dir", type=Path, default=Path("results/unified/history"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bypass = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = ",".join(
        value for value in (os.environ.get("NO_PROXY", ""), bypass) if value
    )
    os.environ["no_proxy"] = ",".join(
        value for value in (os.environ.get("no_proxy", ""), bypass) if value
    )
    config = Config(
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_dims=args.embedding_dims,
        qdrant_url=args.qdrant_url,
        neo4j_uri=args.neo4j_uri,
        letta_url=args.letta_url,
        output_dir=args.output_dir,
        history_dir=args.history_dir,
        top_k=args.top_k,
        seed=args.seed,
    )
    phases = [value.strip() for value in args.phases.split(",") if value.strip()]
    collection = normalize_collection(f"amb_kdd27_{args.system}_{args.run_id}")
    adapter = make_adapter(args.system, config, collection)
    output: dict[str, Any] = {
        "schema_version": "unified-benchmark-v2",
        "run_id": args.run_id,
        "system": args.system,
        "collection": collection,
        "config": {**asdict(config), "output_dir": str(config.output_dir), "history_dir": str(config.history_dir)},
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": environment_snapshot(),
        "phases": {},
    }
    try:
        run_warmup(adapter, args.warmup_writes)
        if "retrieval" in phases:
            records = load_records(args.data, args.retrieval_records, args.seed)
            output["phases"]["retrieval"] = run_retrieval(
                adapter, config, records, args.group_size
            )
        if "conflict" in phases:
            output["phases"]["conflict"] = run_conflict(
                adapter, args.conflict_pairs
            )
        if "isolation" in phases:
            output["phases"]["isolation"] = run_isolation(
                adapter, args.isolation_users, args.isolation_facts
            )
        if "deletion" in phases:
            output["phases"]["deletion"] = run_deletion(
                adapter, args.deletion_records
            )
        if "concurrency" in phases:
            workers = [int(value) for value in args.workers.split(",")]
            output["phases"]["concurrency"] = run_concurrency(
                adapter, args.concurrency_records, workers
            )
        if "scale" in phases:
            scales = [int(value) for value in args.scales.split(",")]
            output["phases"]["scale"] = run_scale(
                adapter, scales, args.scale_read_queries
            )
    finally:
        adapter.close()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{args.system}_{args.run_id}.json"
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(path), "phases": output["phases"]}, indent=2))


if __name__ == "__main__":
    main()
