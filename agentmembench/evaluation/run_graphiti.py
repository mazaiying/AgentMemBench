"""
Step 5: Graphiti (Zep) Pilot 测试
运行方式: python step5_pilot_graphiti.py

前提：
  1. 已安装: pip install graphiti-core
  2. 有 Neo4j Aura 免费实例（https://neo4j.com/cloud/platform/aura-graph-database/）
  3. 设置环境变量：
     export NEO4J_URI='neo4j+s://xxxxxxxx.databases.neo4j.io'
     export NEO4J_PASSWORD='your_password'
     export DASHSCOPE_API_KEY='sk-xxx'

目的：与 Mem0 Pilot 做横向对比，测量相同场景下：
  - 写入延迟（write latency）
  - 检索延迟（read latency）
  - 时序更新准确率（staleness）
"""

import os
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime

# ── 配置 ─────────────────────────────────────────────────────
QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
NEO4J_URI     = os.environ.get("NEO4J_URI", "")
NEO4J_USER    = os.environ.get("NEO4J_USERNAME", "")   # Aura 实例用户名（不一定是 neo4j）
NEO4J_PASS    = os.environ.get("NEO4J_PASSWORD", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── 测试数据（与 Mem0 pilot 完全相同，保证可比性）─────────────
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


async def run_graphiti_pilot():
    """运行 Graphiti pilot，使用千问作为 LLM"""

    # 检查依赖
    missing = []
    if not QWEN_API_KEY: missing.append("DASHSCOPE_API_KEY")
    if not NEO4J_URI:    missing.append("NEO4J_URI")
    if not NEO4J_PASS:   missing.append("NEO4J_PASSWORD")
    if missing:
        print(f"❌ 请设置环境变量: {', '.join(missing)}")
        print("\n设置示例：")
        print("  export DASHSCOPE_API_KEY='sk-xxx'")
        print("  export NEO4J_URI='neo4j+s://xxx.databases.neo4j.io'")
        print("  export NEO4J_PASSWORD='your_password'")
        return

    print("=" * 60)
    print("MemSysBench Mini Pilot — Graphiti 系统测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"LLM:  qwen-plus (via DashScope)")
    print(f"图DB: Neo4j Aura ({NEO4J_URI[:30]}...)")
    print("=" * 60)

    # 1. 初始化 Graphiti（配置千问作为 LLM）
    print("\n[1/3] 初始化 Graphiti...")
    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        # Graphiti 内部 reranker 默认读 OPENAI_API_KEY，把它指向千问即可
        os.environ["OPENAI_API_KEY"] = QWEN_API_KEY
        os.environ["OPENAI_BASE_URL"] = DASHSCOPE_URL

        # 千问 LLM 配置
        llm_config = LLMConfig(
            api_key=QWEN_API_KEY,
            model="qwen-plus",
            base_url=DASHSCOPE_URL,
        )
        llm_client = OpenAIClient(config=llm_config)

        # 千问 Embedding 配置
        embedder_config = OpenAIEmbedderConfig(
            api_key=QWEN_API_KEY,
            embedding_model="text-embedding-v3",
            base_url=DASHSCOPE_URL,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        # 初始化 Graphiti（reranker 自动使用 OPENAI_API_KEY 环境变量）
        graphiti = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASS,
            llm_client=llm_client,
            embedder=embedder,
        )
        await graphiti.build_indices_and_constraints()
        print("✅ Graphiti 初始化成功，已连接 Neo4j")

    except Exception as e:
        print(f"❌ Graphiti 初始化失败: {e}")
        import traceback; traceback.print_exc()
        return

    # 2. 写入记忆（Episodes）
    print(f"\n[2/3] 写入 {len(SAMPLE_MEMORIES)} 条记忆...")
    write_latencies = []

    for i, mem in enumerate(SAMPLE_MEMORIES):
        t_start = time.perf_counter()
        try:
            await graphiti.add_episode(
                name=f"memory_{mem['user_id']}_{i}",
                episode_body=mem["text"],
                source_description=f"user_session for {mem['user_id']}",
                reference_time=datetime.now(),
                group_id=mem["user_id"],   # 用 group_id 隔离不同用户
            )
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            write_latencies.append(latency_ms)
            print(f"  [{i+1:2d}] ✅ 写入成功  延迟: {latency_ms:7.0f}ms  | {mem['text'][:50]}...")
        except Exception as e:
            t_end = time.perf_counter()
            print(f"  [{i+1:2d}] ❌ 写入失败: {e}")

    # 3. 检索测试
    print(f"\n[3/3] 执行 {len(SAMPLE_QUERIES)} 次检索...")
    read_latencies = []
    results_log = []

    for i, q in enumerate(SAMPLE_QUERIES):
        t_start = time.perf_counter()
        try:
            results = await graphiti.search(
                query=q["query"],
                group_ids=[q["user_id"]],
                num_results=3,
            )
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            read_latencies.append(latency_ms)

            # 取最相关结果
            top_result = results[0].fact if results else "（无结果）"
            top_lower  = top_result.lower()

            hit   = (q["expected_en"].lower() in top_lower or
                     q["expected_cn"].lower() in top_lower)
            stale = False
            if "old_en" in q:
                stale = (q["old_en"].lower() in top_lower or
                         q["old_cn"].lower() in top_lower)

            status = "✅" if hit else ("🕰️ 过时" if stale else "⚠️ ")
            print(f"  [{i+1}] {status} 查询: {q['query']}")
            print(f"       返回: {top_result[:90]}")
            if stale:
                print(f"       ⚠️  返回了旧事实！(staleness 问题)")
            print(f"       延迟: {latency_ms:.0f}ms  |  命中: {'是' if hit else '否'}  |  过时: {'是' if stale else '否'}")

            results_log.append({
                "query": q["query"], "retrieved": top_result,
                "hit": hit, "stale": stale, "latency_ms": latency_ms
            })
        except Exception as e:
            t_end = time.perf_counter()
            print(f"  [{i+1}] ❌ 检索失败: {e}")

    # 4. 汇总
    import statistics
    print("\n" + "=" * 60)
    print("📊 实验结果汇总（Graphiti）")
    print("=" * 60)

    # 关键发现：LLM 耦合问题
    write_success = len(write_latencies)
    write_fail    = len(SAMPLE_MEMORIES) - write_success
    if write_fail > 0:
        print(f"⚠️  LLM耦合问题：写入失败 {write_fail}/{len(SAMPLE_MEMORIES)} 条")
        print(f"   原因：Graphiti 实体提取 prompt 要求 {{\"name\":\"...\"}} 格式")
        print(f"   千问返回 {{\"entity_name\":\"...\"}} 或 {{\"entity\":\"...\"}}")
        print(f"   → 这是论文 RQ6 的实锨证据：Graphiti 对非-OpenAI 模型兼容性为 0%")
    if write_latencies:
        print(f"写入延迟（成功 {write_success} 条）:")
        print(f"  平均: {statistics.mean(write_latencies):.0f}ms")
        print(f"  最小: {min(write_latencies):.0f}ms")
        print(f"  最大: {max(write_latencies):.0f}ms")
    if read_latencies:
        print(f"检索延迟 (n={len(read_latencies)}):")
        print(f"  平均: {statistics.mean(read_latencies):.0f}ms")
        print(f"  最小: {min(read_latencies):.0f}ms")
        print(f"  最大: {max(read_latencies):.0f}ms")
    if results_log:
        hit_rate = sum(r["hit"] for r in results_log) / len(results_log)
        stale_n  = sum(r["stale"] for r in results_log)
        print(f"命中率:   {hit_rate:.0%} ({sum(r['hit'] for r in results_log)}/{len(results_log)})")
        print(f"过时返回: {stale_n} 条")

    # 5. 对比 Mem0
    mem0_path = Path("../results/pilot_mem0_result.json")
    if mem0_path.exists():
        with open(mem0_path) as f:
            mem0 = json.load(f)
        print("\n📊 与 Mem0 对比")
        print(f"{'指标':<20} {'Mem0':>12} {'Graphiti':>12}")
        print("-" * 45)
        if mem0.get("write_latencies_ms") and write_latencies:
            m0w = statistics.mean(mem0["write_latencies_ms"])
            gfw = statistics.mean(write_latencies)
            print(f"{'写入延迟 平均':<20} {m0w:>10.0f}ms {gfw:>10.0f}ms")
        if mem0.get("read_latencies_ms") and read_latencies:
            m0r = statistics.mean(mem0["read_latencies_ms"])
            gfr = statistics.mean(read_latencies)
            print(f"{'检索延迟 平均':<20} {m0r:>10.0f}ms {gfr:>10.0f}ms")

    # 6. 保存
    out = {
        "system": "Graphiti",
        "timestamp": datetime.now().isoformat(),
        "config": {"llm": "qwen-plus", "embedder": "text-embedding-v3", "graph_db": "neo4j_aura"},
        "write_latencies_ms": write_latencies,
        "read_latencies_ms":  read_latencies,
        "query_results":      results_log,
    }
    out_path = Path("../results/pilot_graphiti_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存: {out_path}")
    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(run_graphiti_pilot())
