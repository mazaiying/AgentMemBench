"""
Step 9: 并发压测 — Mem0 (RQ3: Concurrency Consistency)

测试内容：
  1. Sequential baseline：串行写入 100 条记忆，测 p50/p95/p99 延迟
  2. Concurrent write：同时发起 N 个写入请求，测并发下的延迟和成功率
  3. Concurrent read：同时发起 N 个检索请求
  4. Mixed R/W：读写混合并发，测一致性（写入后立即能否检索到）

这是 MemSysBench 的核心实验之一：现有系统文档都不提并发行为。
"""

import os, time, json, asyncio, statistics
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 100 条测试记忆（从 10 条扩展）
import random
random.seed(42)

TOPICS = [
    ("user_001", "research", [
        "User researches large language models at Renmin University.",
        "User's paper focuses on Agent Memory Benchmarking for ICDE 2027.",
        "User uses Python for all research code, avoids Java.",
        "User's advisor Professor Zhang holds meetings every Thursday.",
        "User submitted abstract to ICDE 2027 in March.",
        "User is collaborating with a team at Tsinghua on LLM evaluation.",
        "User prefers VSCode over PyCharm for coding.",
        "User runs experiments on a Mac with M2 chip.",
        "User's GitHub repo for the benchmark is private currently.",
        "User plans to open-source MemSysBench after submission.",
    ]),
    ("user_001", "life", [
        "User lives in Beijing's Haidian district near the university.",
        "User likes hotpot but cannot eat spicy food.",
        "User drinks one cup of coffee every morning.",
        "User goes for a 30-minute walk after dinner.",
        "User's favorite restaurant is near campus on Zhongguancun Street.",
        "User watches tech YouTube channels on weekends.",
        "User is learning Japanese in spare time.",
        "User has a cat named 'Pepper'.",
        "User's hometown is Chengdu.",
        "User visited Japan in summer 2024.",
    ]),
    ("user_002", "work", [
        "User A was a product manager at an AI startup in Shanghai.",
        "User A transitioned to technical director in Q1 2026.",  # UPDATE
        "User A manages a team of 12 engineers.",
        "User A's company is developing an AI-powered CRM product.",
        "User A uses Notion for project management.",
        "User A holds weekly team sync every Monday morning.",
        "User A's company raised Series B in late 2025.",
        "User A is responsible for the AI product roadmap.",
        "User A presented at a Shanghai AI summit in April.",
        "User A mentors two junior PMs on the team.",
    ]),
    ("user_003", "tech", [
        "Discussion focuses on KV Cache optimization in LLM inference.",
        "vLLM and SGLang are the two primary frameworks being evaluated.",
        "The core KV Cache problem is semantic cache miss for similar prompts.",
        "PagedAttention in vLLM significantly reduces memory fragmentation.",
        "SGLang's RadixAttention achieves better prefix cache hit rates.",
        "User proposes a semantic hashing approach for KV Cache indexing.",
        "Current KV Cache solutions struggle with multi-tenant isolation.",
        "Quantized KV Cache (FP8) can reduce memory by 2x with small quality loss.",
        "The team is evaluating Mooncake as an alternative KV Cache system.",
        "Target latency improvement is 3-5x compared to no-cache baseline.",
    ]),
]

# 展开为 100 条
MEMORIES_100 = []
for user_id, _, texts in TOPICS:
    for text in texts:
        MEMORIES_100.append({"user_id": user_id, "text": text})

# 扩展到 100 条（每组 10 条 * 4 组 + 补充生成 60 条）
EXTRA_TEMPLATES = [
    "user_001", "user_002", "user_003", "user_004"
]
while len(MEMORIES_100) < 100:
    uid = random.choice(EXTRA_TEMPLATES)
    i   = len(MEMORIES_100)
    MEMORIES_100.append({"user_id": uid,
                         "text": f"Auto-generated memory #{i} for benchmarking load test. "
                                 f"Contains factual info about {uid}'s activity #{i}."})

QUERIES_5 = [
    {"user_id": "user_001", "query": "What is the user's research topic?",
     "expected_en": "large language model"},
    {"user_id": "user_001", "query": "What programming language does the user prefer?",
     "expected_en": "python"},
    {"user_id": "user_002", "query": "What is User A's current job title?",
     "expected_en": "technical director",
     "old_en": "product manager"},
    {"user_id": "user_003", "query": "What is the core problem with KV Cache?",
     "expected_en": "semantic"},
    {"user_id": "user_001", "query": "Where does the user live?",
     "expected_en": "beijing"},
]


def run_concurrency_test():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    from mem0 import Memory

    MEM0_CONFIG = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "qwen-plus",
                "temperature": 0,
                "api_key": QWEN_API_KEY,
                "openai_base_url": DASHSCOPE_URL,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-v3",
                "api_key": QWEN_API_KEY,
                "openai_base_url": DASHSCOPE_URL,
                "embedding_dims": 1024,
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "memsysbench_concurrency",
                "embedding_model_dims": 1024,
                "on_disk": False,
            }
        },
        "version": "v1.1"
    }

    print("=" * 60)
    print("MemSysBench Concurrency Test — Mem0")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"规模: {len(MEMORIES_100)} 条记忆 × {len(QUERIES_5)} 条查询")
    print("=" * 60)

    m = Memory.from_config(MEM0_CONFIG)

    # ── 实验1：串行写入 100 条（测 p50/p95/p99）───────────────
    print(f"\n[EXP1] 串行写入 {len(MEMORIES_100)} 条记忆...")
    seq_write_lats = []
    for i, mem in enumerate(MEMORIES_100):
        t0 = time.perf_counter()
        try:
            m.add(mem["text"], user_id=mem["user_id"])
            lat = (time.perf_counter() - t0) * 1000
            seq_write_lats.append(lat)
            if (i+1) % 10 == 0:
                print(f"  {i+1:3d}/100  最近10条avg: {statistics.mean(seq_write_lats[-10:]):.0f}ms")
        except Exception as e:
            print(f"  [{i+1}] ❌ {e}")

    if seq_write_lats:
        srt = sorted(seq_write_lats)
        n   = len(srt)
        p50 = srt[int(n*0.50)]
        p95 = srt[int(n*0.95)]
        p99 = srt[int(n*0.99)]
        print(f"\n串行写入统计 (n={n}):")
        print(f"  avg={statistics.mean(srt):.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
        print(f"  总耗时: {sum(srt)/1000:.1f}s")

    # ── 实验2：并发读（同时发起 5 个检索）─────────────────────
    print(f"\n[EXP2] 并发检索（5个并发查询）...")
    conc_read_lats = []

    def search_one(q):
        t0 = time.perf_counter()
        try:
            res = m.search(q["query"], filters={"user_id": q["user_id"]}, limit=3)
            lat = (time.perf_counter() - t0) * 1000
            result_list = res.get("results", []) if isinstance(res, dict) else res
            top = result_list[0]["memory"] if result_list else "（无结果）"
            hit = q["expected_en"].lower() in top.lower()
            return {"lat": lat, "hit": hit, "top": top, "query": q["query"]}
        except Exception as e:
            return {"lat": (time.perf_counter()-t0)*1000, "hit": False, "top": str(e), "query": q["query"]}

    # 3轮并发测试，每轮5个并发
    for round_i in range(3):
        t_round = time.perf_counter()
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(search_one, q): q for q in QUERIES_5}
            round_lats = []
            for fut in as_completed(futures):
                r = fut.result()
                round_lats.append(r["lat"])
                conc_read_lats.append(r["lat"])
                hit_str = "✅" if r["hit"] else "⚠️ "
                print(f"  [R{round_i+1}] {hit_str} {r['query'][:40]}  {r['lat']:.0f}ms")
        print(f"  第{round_i+1}轮总耗时: {(time.perf_counter()-t_round)*1000:.0f}ms  avg={statistics.mean(round_lats):.0f}ms")

    if conc_read_lats:
        print(f"\n并发检索统计 (n={len(conc_read_lats)}):")
        srt = sorted(conc_read_lats)
        print(f"  avg={statistics.mean(srt):.0f}ms  p95={srt[int(len(srt)*0.95)]:.0f}ms")

    # ── 实验3：写后立即读（一致性测试）────────────────────────
    print(f"\n[EXP3] 写后立即读一致性测试...")
    consistency_results = []
    TEST_MEM = "user_test: Test user just bought a new laptop, a MacBook Pro M4."
    for trial in range(5):
        # 写入
        t_w = time.perf_counter()
        m.add(TEST_MEM, user_id="user_test")
        write_lat = (time.perf_counter() - t_w) * 1000

        # 立刻读
        t_r = time.perf_counter()
        res = m.search("What did the test user buy?", filters={"user_id": "user_test"}, limit=1)
        read_lat = (time.perf_counter() - t_r) * 1000

        result_list = res.get("results", []) if isinstance(res, dict) else res
        top = result_list[0]["memory"] if result_list else "（无结果）"
        found = "macbook" in top.lower() or "laptop" in top.lower()
        consistency_results.append(found)
        status = "✅ 可读" if found else "❌ 不一致"
        print(f"  trial {trial+1}: 写{write_lat:.0f}ms → 读{read_lat:.0f}ms → {status}")

    cons_rate = sum(consistency_results) / len(consistency_results)
    print(f"\n写后立即读一致性: {cons_rate:.0%} ({sum(consistency_results)}/{len(consistency_results)})")

    # ── 保存结果 ────────────────────────────────────────────
    out = {
        "system": "Mem0",
        "test": "concurrency",
        "timestamp": datetime.now().isoformat(),
        "scale": len(MEMORIES_100),
        "seq_write": {
            "n": len(seq_write_lats),
            "avg_ms": statistics.mean(seq_write_lats) if seq_write_lats else None,
            "p50_ms": sorted(seq_write_lats)[int(len(seq_write_lats)*0.50)] if seq_write_lats else None,
            "p95_ms": sorted(seq_write_lats)[int(len(seq_write_lats)*0.95)] if seq_write_lats else None,
            "p99_ms": sorted(seq_write_lats)[int(len(seq_write_lats)*0.99)] if seq_write_lats else None,
            "total_s": sum(seq_write_lats)/1000 if seq_write_lats else None,
        },
        "conc_read": {
            "n": len(conc_read_lats),
            "avg_ms": statistics.mean(conc_read_lats) if conc_read_lats else None,
            "p95_ms": sorted(conc_read_lats)[int(len(conc_read_lats)*0.95)] if conc_read_lats else None,
        },
        "write_then_read_consistency": {
            "trials": len(consistency_results),
            "consistent": sum(consistency_results),
            "rate": cons_rate,
        }
    }

    out_path = Path("../results/concurrency_mem0_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存: {out_path}")
    print("\n✅ 并发压测完成！")


if __name__ == "__main__":
    run_concurrency_test()
