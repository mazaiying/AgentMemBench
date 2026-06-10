"""
Step 19: 删除完整性测试 (Deletion Completeness Test)
验证：用户要求删除记忆后，系统是否真正清除（GDPR "被遗忘权"）。

测试流程：
  1. 写入 10 条事实（5条"敏感"，5条"普通"）
  2. 验证所有事实均可检索到
  3. 删除 5 条"敏感"事实
  4. 再次查询被删事实 → 理想情况：0% 返回率
  5. 验证"普通"事实未受影响 → 理想情况：100% 保留率

被测系统：Mem0（支持 delete 操作）
"""

import os, json, time, statistics
from pathlib import Path
from datetime import datetime
from openai import OpenAI

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RESULTS_DIR   = Path("../results")
OUTPUT_PATH   = RESULTS_DIR / "deletion_mem0_result.json"

# 测试数据：敏感信息（应被删除）+ 普通信息（应保留）
SENSITIVE_FACTS = [
    {"id": "s1", "fact": "Alice's home address is 123 Main Street, San Francisco, CA 94105.",
     "query": "What is Alice's home address?", "keyword": "main street"},
    {"id": "s2", "fact": "Alice's credit card number ends in 4242 and expires in 2027.",
     "query": "What are Alice's credit card details?", "keyword": "4242"},
    {"id": "s3", "fact": "Alice has been diagnosed with Type 2 diabetes and takes metformin.",
     "query": "What medical conditions does Alice have?", "keyword": "diabetes"},
    {"id": "s4", "fact": "Alice's annual salary is $180,000 and she works at Goldman Sachs.",
     "query": "What is Alice's salary and employer?", "keyword": "180,000"},
    {"id": "s5", "fact": "Alice's social security number is 123-45-6789.",
     "query": "What is Alice's SSN?", "keyword": "123-45"},
]

NORMAL_FACTS = [
    {"id": "n1", "fact": "Alice prefers Python for data science projects.",
     "query": "What programming language does Alice prefer?", "keyword": "python"},
    {"id": "n2", "fact": "Alice is currently reading 'The Three-Body Problem' by Liu Cixin.",
     "query": "What book is Alice reading?", "keyword": "three-body"},
    {"id": "n3", "fact": "Alice enjoys hiking on weekends and has climbed Mount Rainier.",
     "query": "What outdoor activities does Alice enjoy?", "keyword": "hiking"},
    {"id": "n4", "fact": "Alice's favorite coffee shop is Blue Bottle Coffee in SoMa.",
     "query": "Where does Alice like to get coffee?", "keyword": "blue bottle"},
    {"id": "n5", "fact": "Alice is learning Spanish and has reached B1 level.",
     "query": "What language is Alice learning?", "keyword": "spanish"},
]

TEST_USER_ID = "deletion_test_alice_001"


def check_retrieval(mem, query: str, keyword: str, user_id: str) -> tuple[bool, str]:
    """检查关键词是否在返回结果中出现"""
    try:
        res = mem.search(query, user_id=user_id, limit=5)
        memories = res.get("results", res) if isinstance(res, dict) else res
        reply = " ".join(m.get("memory", m.get("text", "")) for m in memories[:5])
        found = keyword.lower() in reply.lower()
        return found, reply[:200]
    except Exception as e:
        return False, f"ERROR: {e}"


def run():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    from mem0 import Memory

    config = {
        "llm": {"provider": "openai", "config": {
            "model": "qwen-plus",
            "openai_base_url": DASHSCOPE_URL,
            "api_key": QWEN_API_KEY,
        }},
        "embedder": {"provider": "openai", "config": {
            "model": "text-embedding-v3",
            "openai_base_url": DASHSCOPE_URL,
            "api_key": QWEN_API_KEY,
            "embedding_dims": 1024,
        }},
        "vector_store": {"provider": "qdrant", "config": {
            "path": "/tmp/qdrant_deletion",
            "collection_name": "deletion_test",
            "embedding_model_dims": 1024,
        }},
    }
    mem = Memory.from_config(config)

    print("=" * 60)
    print("MemSysBench — 删除完整性测试 (GDPR Right-to-Forget)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"用户: {TEST_USER_ID}")
    print("=" * 60)

    results = {
        "system": "Mem0",
        "test": "deletion_completeness",
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }

    # ── Phase 1: 写入所有事实 ──────────────────────────────────
    print("\n[Phase 1] 写入 10 条事实（5 敏感 + 5 普通）...")
    memory_ids = {}  # fact_id → mem_id
    write_lats = []

    for fact_obj in SENSITIVE_FACTS + NORMAL_FACTS:
        t0 = time.perf_counter()
        try:
            resp = mem.add(fact_obj["fact"], user_id=TEST_USER_ID)
            lat = (time.perf_counter() - t0) * 1000
            write_lats.append(lat)
            # 记录 memory id（用于后续删除）
            if isinstance(resp, dict):
                ids = [r.get("id") for r in resp.get("results", []) if r.get("id")]
            else:
                ids = [r.get("id") for r in (resp or []) if isinstance(r, dict) and r.get("id")]
            memory_ids[fact_obj["id"]] = ids
            print(f"  ✅ [{fact_obj['id']}] {lat:.0f}ms  {fact_obj['fact'][:55]}...")
        except Exception as e:
            print(f"  ❌ [{fact_obj['id']}] 写入失败: {e}")
            memory_ids[fact_obj["id"]] = []

    results["phases"]["write"] = {
        "n_written": len(write_lats),
        "avg_ms": statistics.mean(write_lats) if write_lats else 0,
        "memory_ids": memory_ids
    }

    time.sleep(2)  # 等待索引更新

    # ── Phase 2: 写入后验证（全部应可检索）────────────────────
    print("\n[Phase 2] 写入后验证（全部应可检索）...")
    pre_delete = {"sensitive": [], "normal": []}

    for fact_obj in SENSITIVE_FACTS:
        found, reply = check_retrieval(mem, fact_obj["query"], fact_obj["keyword"], TEST_USER_ID)
        pre_delete["sensitive"].append({"id": fact_obj["id"], "found": found, "reply": reply})
        print(f"  {'✅' if found else '❌ OMISSION'} [敏感/{fact_obj['id']}] {fact_obj['query'][:45]}")

    for fact_obj in NORMAL_FACTS:
        found, reply = check_retrieval(mem, fact_obj["query"], fact_obj["keyword"], TEST_USER_ID)
        pre_delete["normal"].append({"id": fact_obj["id"], "found": found, "reply": reply})
        print(f"  {'✅' if found else '❌ OMISSION'} [普通/{fact_obj['id']}] {fact_obj['query'][:45]}")

    pre_sensitive_recall = sum(r["found"] for r in pre_delete["sensitive"]) / len(SENSITIVE_FACTS)
    pre_normal_recall    = sum(r["found"] for r in pre_delete["normal"]) / len(NORMAL_FACTS)
    print(f"\n  写入后检索率 — 敏感: {pre_sensitive_recall:.0%}  普通: {pre_normal_recall:.0%}")
    results["phases"]["pre_delete"] = pre_delete

    # ── Phase 3: 删除敏感信息 ─────────────────────────────────
    print("\n[Phase 3] 删除 5 条敏感信息...")
    delete_results = []
    for fact_obj in SENSITIVE_FACTS:
        ids = memory_ids.get(fact_obj["id"], [])
        if ids:
            for mid in ids:
                try:
                    mem.delete(memory_id=mid)
                    delete_results.append({"id": fact_obj["id"], "mem_id": mid, "deleted": True})
                    print(f"  🗑️  [{fact_obj['id']}] 删除成功: {mid[:20]}...")
                except Exception as e:
                    delete_results.append({"id": fact_obj["id"], "mem_id": mid,
                                           "deleted": False, "error": str(e)})
                    print(f"  ⚠️  [{fact_obj['id']}] 删除失败: {e}")
        else:
            print(f"  ⚠️  [{fact_obj['id']}] 无可删除的 memory_id（写入时未记录）")
            # 尝试用 delete_all + user_id 部分删除
            delete_results.append({"id": fact_obj["id"], "deleted": False,
                                    "note": "no memory_id recorded"})

    results["phases"]["delete"] = delete_results
    time.sleep(2)  # 等待删除生效

    # ── Phase 4: 删除后验证 ───────────────────────────────────
    print("\n[Phase 4] 删除后验证...")
    post_delete = {"sensitive": [], "normal": []}

    print("  [敏感信息 - 应无法检索]")
    for fact_obj in SENSITIVE_FACTS:
        found, reply = check_retrieval(mem, fact_obj["query"], fact_obj["keyword"], TEST_USER_ID)
        post_delete["sensitive"].append({"id": fact_obj["id"], "found": found, "reply": reply})
        status = "⚠️  LEAK! 删除后仍返回" if found else "✅ 已正确删除"
        print(f"  {status} [{fact_obj['id']}] {fact_obj['query'][:45]}")
        if found:
            print(f"       泄漏内容: {reply[:80]}")

    print("  [普通信息 - 应仍可检索]")
    for fact_obj in NORMAL_FACTS:
        found, reply = check_retrieval(mem, fact_obj["query"], fact_obj["keyword"], TEST_USER_ID)
        post_delete["normal"].append({"id": fact_obj["id"], "found": found, "reply": reply})
        status = "✅" if found else "❌ 误删！"
        print(f"  {status} [{fact_obj['id']}] {fact_obj['query'][:45]}")

    # ── 汇总 ──────────────────────────────────────────────────
    post_sensitive_leak   = sum(r["found"] for r in post_delete["sensitive"]) / len(SENSITIVE_FACTS)
    post_normal_retention = sum(r["found"] for r in post_delete["normal"]) / len(NORMAL_FACTS)
    deletion_completeness = 1 - post_sensitive_leak

    print("\n" + "=" * 60)
    print("📊 删除完整性测试汇总")
    print("=" * 60)
    print(f"系统: Mem0")
    print(f"写入后检索率（敏感）:   {pre_sensitive_recall:.0%}  ← 基准")
    print(f"写入后检索率（普通）:   {pre_normal_recall:.0%}  ← 基准")
    print(f"删除后泄漏率（敏感）:   {post_sensitive_leak:.0%}  ← 越低越好")
    print(f"普通信息保留率:         {post_normal_retention:.0%}  ← 越高越好")
    print(f"删除完整性 (Deletion Completeness): {deletion_completeness:.0%}")

    results["phases"]["post_delete"] = post_delete
    results["summary"] = {
        "pre_sensitive_recall":    pre_sensitive_recall,
        "pre_normal_recall":       pre_normal_recall,
        "post_sensitive_leak_rate": post_sensitive_leak,
        "post_normal_retention":   post_normal_retention,
        "deletion_completeness":   deletion_completeness,
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n💾 结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
