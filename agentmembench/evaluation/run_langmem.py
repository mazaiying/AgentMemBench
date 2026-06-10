"""
Step 7: LangMem Pilot
LangMem 是 LangGraph 原生的 Memory 库。

注意：LangMem 需要 LangChain 生态，通过 ChatOpenAI 使用千问。
"""

import os, time, json, statistics, asyncio
from pathlib import Path
from datetime import datetime

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SAMPLE_MEMORIES = [
    {"user_id": "user_001", "text": "Ma Zaiying is a doctoral student at Renmin University of China, researching large language models."},
    {"user_id": "user_001", "text": "User prefers writing code in Python and dislikes Java."},
    {"user_id": "user_001", "text": "User's doctoral advisor is Professor Zhang, with group meetings every Thursday afternoon."},
    {"user_id": "user_001", "text": "User is writing a paper for ICDE 2027 on Agent Memory Benchmarking."},
    {"user_id": "user_001", "text": "User lives in Beijing, likes hotpot but cannot eat spicy food."},
    {"user_id": "user_002", "text": "User A is a product manager based in Shanghai, responsible for AI product line."},
    {"user_id": "user_002", "text": "User A mentioned their company plans to launch a new feature in Q2 next year."},
    {"user_id": "user_002", "text": "UPDATE: User A has transferred from product manager to technical director."},
    {"user_id": "user_003", "text": "This conversation discusses KV Cache optimization involving vLLM and SGLang frameworks."},
    {"user_id": "user_003", "text": "The largest problem in KV Cache is the inability to cache hits for semantically similar prompts."},
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


async def run_langmem():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    print("=" * 60)
    print("MemSysBench Mini Pilot — LangMem")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        from openai import AsyncOpenAI

        # 千问 LLM（通过 LangChain OpenAI 兼容接口）
        llm = ChatOpenAI(
            model="qwen-plus",
            api_key=QWEN_API_KEY,
            base_url=DASHSCOPE_URL,
            temperature=0,
        )

        # 直接用 OpenAI 客户端做 embedding（LangChain 的 OpenAIEmbeddings 格式与千问不兼容）
        embed_client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)

        async def embed_texts(texts: list[str]) -> list[list[float]]:
            resp = await embed_client.embeddings.create(
                model="text-embedding-v3", input=texts
            )
            return [d.embedding for d in resp.data]

        print("✅ LangMem 初始化成功（千问 LLM + 直接 Embedding）")
    except Exception as e:
        print(f"❌ LangMem 初始化失败: {e}")
        import traceback; traceback.print_exc()
        return

    # 写入
    print(f"\n[1/2] 写入 {len(SAMPLE_MEMORIES)} 条记忆...")
    write_latencies = []
    
    # LangMem 使用 in-memory store
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore(
        index={"embed": embed_texts, "dims": 1024}
    )

    for i, mem in enumerate(SAMPLE_MEMORIES):
        t0 = time.perf_counter()
        try:
            namespace = (mem["user_id"], "memories")
            await store.aput(namespace, f"mem_{i}", {"text": mem["text"]})
            lat = (time.perf_counter() - t0) * 1000
            write_latencies.append(lat)
            print(f"  [{i+1:2d}] ✅ {lat:6.0f}ms  {mem['text'][:50]}...")
        except Exception as e:
            print(f"  [{i+1:2d}] ❌ 写入失败: {e}")

    # 检索
    print(f"\n[2/2] 执行 {len(SAMPLE_QUERIES)} 次检索...")
    read_latencies, results_log = [], []
    for i, q in enumerate(SAMPLE_QUERIES):
        t0 = time.perf_counter()
        try:
            namespace = (q["user_id"], "memories")
            results = await store.asearch(namespace, query=q["query"], limit=3)
            lat = (time.perf_counter() - t0) * 1000
            read_latencies.append(lat)

            top_text  = results[0].value.get("text", "（无结果）") if results else "（无结果）"
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
        except Exception as e:
            print(f"  [{i+1}] ❌ 检索失败: {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("📊 LangMem 结果汇总")
    print("=" * 60)
    if write_latencies:
        print(f"写入延迟: 平均 {statistics.mean(write_latencies):.0f}ms / 最小 {min(write_latencies):.0f}ms / 最大 {max(write_latencies):.0f}ms")
    if read_latencies:
        print(f"检索延迟: 平均 {statistics.mean(read_latencies):.0f}ms / 最小 {min(read_latencies):.0f}ms / 最大 {max(read_latencies):.0f}ms")
    if results_log:
        hit_n = sum(r["hit"] for r in results_log)
        print(f"命中率: {hit_n/len(results_log):.0%} ({hit_n}/{len(results_log)})")

    out = {
        "system": "LangMem",
        "timestamp": datetime.now().isoformat(),
        "config": {"llm": "qwen-plus", "embedder": "text-embedding-v3", "store": "InMemoryStore"},
        "write_latencies_ms": write_latencies,
        "read_latencies_ms":  read_latencies,
        "query_results": results_log,
    }
    out_path = Path("../results/pilot_langmem_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_langmem())
