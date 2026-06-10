"""
Step 6: Naive RAG Baseline Pilot
完全自行实现，不依赖任何 Memory 框架。

架构：
  写入：Qwen text-embedding-v3 向量化 → 存入 Qdrant (in-memory)
  检索：向量化 query → cosine similarity 检索 Top-K

这是最简单的 baseline，代表"没有专门 Memory 系统"的情况。
"""

import os, time, json, statistics
from pathlib import Path
from datetime import datetime

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 与所有 pilot 完全相同的测试数据
SAMPLE_MEMORIES = [
    {"id": "m01", "user_id": "user_001", "text": "Ma Zaiying is a doctoral student at Renmin University of China, researching large language models."},
    {"id": "m02", "user_id": "user_001", "text": "User prefers writing code in Python and dislikes Java."},
    {"id": "m03", "user_id": "user_001", "text": "User's doctoral advisor is Professor Zhang, with group meetings every Thursday afternoon."},
    {"id": "m04", "user_id": "user_001", "text": "User is writing a paper for ICDE 2027 on Agent Memory Benchmarking."},
    {"id": "m05", "user_id": "user_001", "text": "User lives in Beijing, likes hotpot but cannot eat spicy food."},
    {"id": "m06", "user_id": "user_002", "text": "User A is a product manager based in Shanghai, responsible for AI product line."},
    {"id": "m07", "user_id": "user_002", "text": "User A mentioned their company plans to launch a new feature in Q2 next year."},
    {"id": "m08", "user_id": "user_002", "text": "UPDATE: User A has transferred from product manager to technical director."},
    {"id": "m09", "user_id": "user_003", "text": "This conversation discusses KV Cache optimization involving vLLM and SGLang frameworks."},
    {"id": "m10", "user_id": "user_003", "text": "The largest problem in KV Cache is the inability to cache hits for semantically similar prompts."},
]

SAMPLE_QUERIES = [
    {"user_id": "user_001", "query": "What is this user's research direction?",
     "expected_en": "large language model", "expected_cn": "大语言模型"},
    {"user_id": "user_001", "query": "What programming language does the user prefer?",
     "expected_en": "python", "expected_cn": "Python"},
    {"user_id": "user_001", "query": "Which conference is the user submitting a paper to?",
     "expected_en": "icde", "expected_cn": "ICDE"},
    {"user_id": "user_002", "query": "What is User A's current job title?",
     "expected_en": "technical director", "expected_cn": "技术总监",
     "old_en": "product manager", "old_cn": "产品经理"},
    {"user_id": "user_003", "query": "What is the core problem with KV Cache?",
     "expected_en": "semantic", "expected_cn": "语义"},
]

def get_embedding(text: str, client) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-v3", input=text)
    return resp.data[0].embedding

def run_naive_rag():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

    client    = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)
    qdrant    = QdrantClient(":memory:")
    DIM       = 1024
    COL       = "naive_rag_pilot"

    if qdrant.collection_exists(COL):
        qdrant.delete_collection(COL)
    qdrant.create_collection(COL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    print("=" * 60)
    print("MemSysBench Mini Pilot — Naive RAG Baseline")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("架构: Qwen text-embedding-v3 + Qdrant (in-memory)")
    print("=" * 60)

    # 写入
    print(f"\n[1/2] 写入 {len(SAMPLE_MEMORIES)} 条记忆...")
    write_latencies = []
    points = []
    for i, mem in enumerate(SAMPLE_MEMORIES):
        t0 = time.perf_counter()
        vec = get_embedding(mem["text"], client)
        qdrant.upsert(COL, points=[PointStruct(
            id=i, vector=vec,
            payload={"text": mem["text"], "user_id": mem["user_id"], "mem_id": mem["id"]}
        )])
        lat = (time.perf_counter() - t0) * 1000
        write_latencies.append(lat)
        print(f"  [{i+1:2d}] ✅ {lat:6.0f}ms  {mem['text'][:50]}...")

    # 检索
    print(f"\n[2/2] 执行 {len(SAMPLE_QUERIES)} 次检索...")
    read_latencies, results_log = [], []
    for i, q in enumerate(SAMPLE_QUERIES):
        t0 = time.perf_counter()
        q_vec = get_embedding(q["query"], client)
        from qdrant_client.models import QueryRequest
        hits = qdrant.query_points(
            collection_name=COL,
            query=q_vec,
            limit=3,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=q["user_id"]))])
        ).points
        lat = (time.perf_counter() - t0) * 1000
        read_latencies.append(lat)

        top_text  = hits[0].payload["text"] if hits else "（无结果）"
        top_lower = top_text.lower()
        hit   = q["expected_en"].lower() in top_lower or q["expected_cn"].lower() in top_lower
        stale = False
        if "old_en" in q:
            stale = q["old_en"].lower() in top_lower or q["old_cn"].lower() in top_lower

        status = "✅" if hit else ("🕰️ 过时" if stale else "⚠️ ")
        print(f"  [{i+1}] {status} {q['query']}")
        print(f"       返回: {top_text[:90]}")
        if stale: print(f"       ⚠️  返回了旧事实！")
        print(f"       延迟: {lat:.0f}ms | 命中: {'是' if hit else '否'} | 过时: {'是' if stale else '否'}")
        results_log.append({"query": q["query"], "retrieved": top_text,
                            "hit": hit, "stale": stale, "latency_ms": lat})

    # 汇总
    print("\n" + "=" * 60)
    print("📊 Naive RAG Baseline 结果")
    print("=" * 60)
    print(f"写入延迟: 平均 {statistics.mean(write_latencies):.0f}ms / 最小 {min(write_latencies):.0f}ms / 最大 {max(write_latencies):.0f}ms")
    print(f"检索延迟: 平均 {statistics.mean(read_latencies):.0f}ms / 最小 {min(read_latencies):.0f}ms / 最大 {max(read_latencies):.0f}ms")
    hit_n   = sum(r["hit"]   for r in results_log)
    stale_n = sum(r["stale"] for r in results_log)
    print(f"命中率: {hit_n/len(results_log):.0%} ({hit_n}/{len(results_log)})")
    print(f"过时返回: {stale_n} 条")

    out = {
        "system": "Naive RAG",
        "timestamp": datetime.now().isoformat(),
        "config": {"embedder": "text-embedding-v3", "vector_store": "qdrant (in-memory)", "llm": "none"},
        "write_latencies_ms": write_latencies,
        "read_latencies_ms":  read_latencies,
        "query_results": results_log,
    }
    out_path = Path("../results/pilot_naive_rag_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")

if __name__ == "__main__":
    run_naive_rag()
