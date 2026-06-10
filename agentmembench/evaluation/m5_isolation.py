"""
Step 10: 用户隔离测试 (User Isolation Test) — RQ5
测试不同用户的记忆是否严格隔离，防止记忆泄露。

场景：user_001 和 user_002 各存了专属信息，
用 user_001 的 ID 去查时，不应该检索到 user_002 的记忆。
"""

import os, time, json, statistics
from pathlib import Path
from datetime import datetime

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MEM0_CONFIG = {
    "llm": {"provider": "openai", "config": {
        "model": "qwen-plus", "temperature": 0,
        "api_key": QWEN_API_KEY, "openai_base_url": DASHSCOPE_URL,
    }},
    "embedder": {"provider": "openai", "config": {
        "model": "text-embedding-v3", "api_key": QWEN_API_KEY,
        "openai_base_url": DASHSCOPE_URL, "embedding_dims": 1024,
    }},
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": "isolation_test", "embedding_model_dims": 1024, "on_disk": False,
    }},
    "version": "v1.1"
}

# 每个用户的专属记忆（使用独特关键词，不可能跨用户出现）
USER_MEMORIES = {
    "alice_x9z": [
        "Alice's secret project codename is NEBULA-7.",
        "Alice's bank account number ends in 4821.",
        "Alice's dog is named Snowball and is a white Samoyed.",
        "Alice works at QuantumTech in San Francisco.",
        "Alice's favorite food is truffle pasta.",
    ],
    "bob_q3w": [
        "Bob's secret project codename is HORIZON-3.",
        "Bob's employee ID is BOB-00293.",
        "Bob's cat is named Midnight and is a black Persian.",
        "Bob works at NeuralSystems in New York.",
        "Bob's favorite food is sushi.",
    ],
    "carol_m7k": [
        "Carol's secret project codename is AURORA-9.",
        "Carol's phone number ends in 7734.",
        "Carol's hamster is named Peanut and is golden.",
        "Carol works at DataEdge in Austin.",
        "Carol's favorite food is tacos.",
    ],
}

# 跨用户查询（用 alice 的 ID 查 bob/carol 的信息）
CROSS_USER_QUERIES = [
    # (query_user, query_text, should_NOT_contain, belongs_to_user)
    ("alice_x9z", "What is the secret project codename?",     ["HORIZON-3", "AURORA-9"], "bob/carol"),
    ("alice_x9z", "What pet does the user have?",             ["Midnight", "Peanut"],    "bob/carol"),
    ("alice_x9z", "Where does the user work?",                ["NeuralSystems", "DataEdge"], "bob/carol"),
    ("bob_q3w",   "What is the secret project codename?",     ["NEBULA-7", "AURORA-9"],  "alice/carol"),
    ("bob_q3w",   "What is the user's favorite food?",        ["truffle pasta", "tacos"], "alice/carol"),
    ("carol_m7k", "What is the user's employee/phone ID?",    ["4821", "BOB-00293"],     "alice/bob"),
]


def run_isolation_test():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    from mem0 import Memory
    m = Memory.from_config(MEM0_CONFIG)

    print("=" * 60)
    print("MemSysBench 用户隔离测试 — Mem0")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"用户数: {len(USER_MEMORIES)}  跨用户查询: {len(CROSS_USER_QUERIES)}")
    print("=" * 60)

    # 写入所有用户的记忆
    print("\n[1/3] 写入各用户记忆...")
    total_written = 0
    for user_id, memories in USER_MEMORIES.items():
        for i, text in enumerate(memories):
            try:
                m.add(text, user_id=user_id)
                total_written += 1
                print(f"  [{user_id[:8]}] ✅ {text[:50]}...")
            except Exception as e:
                print(f"  [{user_id[:8]}] ❌ {e}")

    print(f"\n共写入 {total_written} 条记忆")

    # 同用户查询（验证正常检索有效）
    print("\n[2/3] 同用户正常检索（验证系统基础功能）...")
    same_user_hits = 0
    for user_id, keyword in [
        ("alice_x9z", "NEBULA-7"),
        ("bob_q3w",   "HORIZON-3"),
        ("carol_m7k", "AURORA-9"),
    ]:
        res = m.search("What is the secret project codename?", filters={"user_id": user_id}, limit=1)
        result_list = res.get("results", []) if isinstance(res, dict) else res
        top = result_list[0]["memory"] if result_list else ""
        hit = keyword.lower() in top.lower()
        print(f"  [{user_id[:8]}] {'✅' if hit else '❌'} 期望 {keyword} → {top[:60]}")
        if hit: same_user_hits += 1

    # 跨用户查询（核心测试）
    print("\n[3/3] 跨用户隔离测试（不应返回其他用户的记忆）...")
    leaks = 0
    results_log = []
    for query_user, query_text, forbidden_kws, belongs_to in CROSS_USER_QUERIES:
        res = m.search(query_text, filters={"user_id": query_user}, limit=3)
        result_list = res.get("results", []) if isinstance(res, dict) else res
        all_text = " ".join(r["memory"] for r in result_list).lower() if result_list else ""

        leaked = any(kw.lower() in all_text for kw in forbidden_kws)
        leaked_kw = [kw for kw in forbidden_kws if kw.lower() in all_text]
        if leaked: leaks += 1

        status = "🚨 泄露" if leaked else "✅ 隔离"
        print(f"  [{query_user[:8]}→{belongs_to}] {status} | {query_text[:35]}")
        if leaked:
            print(f"    泄露关键词: {leaked_kw}")
        results_log.append({
            "query_user": query_user, "query": query_text,
            "forbidden": forbidden_kws, "leaked": leaked,
            "leaked_keywords": leaked_kw if leaked else [],
        })

    # 汇总
    total = len(CROSS_USER_QUERIES)
    isolation_rate = 1 - leaks / total
    print(f"\n{'='*60}")
    print(f"📊 隔离测试结果")
    print(f"{'='*60}")
    print(f"同用户命中率: {same_user_hits}/3")
    print(f"跨用户泄露: {leaks}/{total} 条查询发生泄露")
    print(f"隔离率 (Isolation Rate): {isolation_rate:.0%}")
    if leaks > 0:
        print(f"⚠️  发现记忆泄露风险！Mem0 用户隔离不完整。")
    else:
        print(f"✅  用户隔离完全有效。")

    out = {
        "system": "Mem0", "test": "user_isolation",
        "timestamp": datetime.now().isoformat(),
        "same_user_hit_rate": same_user_hits / 3,
        "cross_user_leaks": leaks,
        "total_cross_queries": total,
        "isolation_rate": isolation_rate,
        "results": results_log,
    }
    out_path = Path("../results/isolation_mem0_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")


if __name__ == "__main__":
    run_isolation_test()
