"""
Step 11: MemConflict 数据集构建 + 测试 — RQ4 深化
合成含矛盾事实的更新序列，系统化测试时序更新准确率。

数据集格式：
  每条记录包含：旧事实 → 新事实 → 查询 → 期望答案
  测试系统是否能在 UPDATE 后返回新事实而非旧事实。
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
        "collection_name": "memconflict_test", "embedding_model_dims": 1024, "on_disk": False,
    }},
    "version": "v1.1"
}

# MemConflict 数据集：30条更新场景，覆盖5种更新类型
MEMCONFLICT_DATASET = [
    # Type 1: 职位变更 (Job Change) — 10条
    {"id": "mc001", "user": "u_job_01", "type": "job_change",
     "old": "Sarah is a junior software engineer at TechCorp.",
     "new": "Sarah has been promoted to senior software engineer at TechCorp.",
     "query": "What is Sarah's job title?",
     "expected_new": "senior software engineer", "expected_old": "junior software engineer"},
    {"id": "mc002", "user": "u_job_02", "type": "job_change",
     "old": "Mike works as a data analyst.",
     "new": "Mike switched careers and is now a machine learning engineer.",
     "query": "What does Mike do for work?",
     "expected_new": "machine learning engineer", "expected_old": "data analyst"},
    {"id": "mc003", "user": "u_job_03", "type": "job_change",
     "old": "Lisa is the CEO of StartupX.",
     "new": "Lisa resigned as CEO and is now an independent consultant.",
     "query": "What is Lisa's current role?",
     "expected_new": "consultant", "expected_old": "ceo"},
    {"id": "mc004", "user": "u_job_04", "type": "job_change",
     "old": "Tom is a product manager at BigCo.",
     "new": "Tom left BigCo and founded his own startup called NovaTech.",
     "query": "Where does Tom work?",
     "expected_new": "novatech", "expected_old": "bigco"},
    {"id": "mc005", "user": "u_job_05", "type": "job_change",
     "old": "Emma is a PhD student at MIT.",
     "new": "Emma graduated and joined Google as a research scientist.",
     "query": "What is Emma's current occupation?",
     "expected_new": "research scientist", "expected_old": "phd student"},

    # Type 2: 住址变更 (Location Change) — 5条
    {"id": "mc011", "user": "u_loc_01", "type": "location_change",
     "old": "David lives in New York City.",
     "new": "David moved to San Francisco for a new job.",
     "query": "Where does David live?",
     "expected_new": "san francisco", "expected_old": "new york"},
    {"id": "mc012", "user": "u_loc_02", "type": "location_change",
     "old": "The team's office is in London.",
     "new": "The team relocated their office to Berlin.",
     "query": "Where is the team's office?",
     "expected_new": "berlin", "expected_old": "london"},
    {"id": "mc013", "user": "u_loc_03", "type": "location_change",
     "old": "Anna attends Stanford University.",
     "new": "Anna transferred to Harvard University.",
     "query": "Which university does Anna attend?",
     "expected_new": "harvard", "expected_old": "stanford"},
    {"id": "mc014", "user": "u_loc_04", "type": "location_change",
     "old": "The conference will be held in Tokyo.",
     "new": "The conference venue changed to Seoul.",
     "query": "Where is the conference being held?",
     "expected_new": "seoul", "expected_old": "tokyo"},
    {"id": "mc015", "user": "u_loc_05", "type": "location_change",
     "old": "Mark's lab is on the 3rd floor.",
     "new": "Mark's lab moved to the 7th floor after renovation.",
     "query": "Which floor is Mark's lab on?",
     "expected_new": "7th", "expected_old": "3rd"},

    # Type 3: 数值更新 (Numeric Update) — 5条
    {"id": "mc021", "user": "u_num_01", "type": "numeric_update",
     "old": "The project has a budget of $1 million.",
     "new": "The project budget was increased to $2.5 million.",
     "query": "What is the project budget?",
     "expected_new": "2.5 million", "expected_old": "1 million"},
    {"id": "mc022", "user": "u_num_02", "type": "numeric_update",
     "old": "The team has 8 members.",
     "new": "The team expanded and now has 15 members.",
     "query": "How many members are on the team?",
     "expected_new": "15", "expected_old": "8"},
    {"id": "mc023", "user": "u_num_03", "type": "numeric_update",
     "old": "The app has 10,000 daily active users.",
     "new": "After the viral campaign, the app now has 500,000 daily active users.",
     "query": "How many daily active users does the app have?",
     "expected_new": "500,000", "expected_old": "10,000"},
    {"id": "mc024", "user": "u_num_04", "type": "numeric_update",
     "old": "The model achieves 78% accuracy on the benchmark.",
     "new": "After fine-tuning, the model now achieves 91% accuracy.",
     "query": "What accuracy does the model achieve?",
     "expected_new": "91", "expected_old": "78"},
    {"id": "mc025", "user": "u_num_05", "type": "numeric_update",
     "old": "The deadline is March 15th.",
     "new": "The deadline was extended to April 30th.",
     "query": "What is the deadline?",
     "expected_new": "april", "expected_old": "march"},

    # Type 4: 偏好更新 (Preference Change) — 5条
    {"id": "mc031", "user": "u_pref_01", "type": "preference_change",
     "old": "User prefers dark mode in all applications.",
     "new": "User switched to light mode after eye strain issues.",
     "query": "What display mode does the user prefer?",
     "expected_new": "light mode", "expected_old": "dark mode"},
    {"id": "mc032", "user": "u_pref_02", "type": "preference_change",
     "old": "User's preferred programming language is JavaScript.",
     "new": "User transitioned to TypeScript as their primary language.",
     "query": "What is the user's preferred programming language?",
     "expected_new": "typescript", "expected_old": "javascript"},
    {"id": "mc033", "user": "u_pref_03", "type": "preference_change",
     "old": "User is vegetarian and doesn't eat meat.",
     "new": "User started eating fish again and is now pescatarian.",
     "query": "What is the user's dietary preference?",
     "expected_new": "pescatarian", "expected_old": "vegetarian"},
    {"id": "mc034", "user": "u_pref_04", "type": "preference_change",
     "old": "User uses VS Code as their primary IDE.",
     "new": "User switched to Cursor IDE for AI-assisted coding.",
     "query": "What IDE does the user use?",
     "expected_new": "cursor", "expected_old": "vs code"},
    {"id": "mc035", "user": "u_pref_05", "type": "preference_change",
     "old": "User drinks 3 cups of coffee per day.",
     "new": "User quit coffee completely and switched to green tea.",
     "query": "What does the user drink?",
     "expected_new": "green tea", "expected_old": "coffee"},

    # Type 5: 状态更新 (Status Change) — 5条
    {"id": "mc041", "user": "u_stat_01", "type": "status_change",
     "old": "The paper is under review at ICDE 2027.",
     "new": "The paper was accepted at ICDE 2027.",
     "query": "What is the status of the paper?",
     "expected_new": "accepted", "expected_old": "under review"},
    {"id": "mc042", "user": "u_stat_02", "type": "status_change",
     "old": "The project is in the planning phase.",
     "new": "The project moved to the development phase.",
     "query": "What phase is the project in?",
     "expected_new": "development", "expected_old": "planning"},
    {"id": "mc043", "user": "u_stat_03", "type": "status_change",
     "old": "User is single.",
     "new": "User got married last month.",
     "query": "What is the user's relationship status?",
     "expected_new": "married", "expected_old": "single"},
    {"id": "mc044", "user": "u_stat_04", "type": "status_change",
     "old": "The product is in beta testing.",
     "new": "The product launched publicly and is now generally available.",
     "query": "What is the product's availability status?",
     "expected_new": "generally available", "expected_old": "beta"},
    {"id": "mc045", "user": "u_stat_05", "type": "status_change",
     "old": "The server is running on AWS.",
     "new": "The team migrated the server to Google Cloud.",
     "query": "Where is the server hosted?",
     "expected_new": "google cloud", "expected_old": "aws"},
]


def run_memconflict_test():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    from mem0 import Memory
    m = Memory.from_config(MEM0_CONFIG)

    print("=" * 60)
    print("MemSysBench MemConflict 数据集测试 — Mem0")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试条数: {len(MEMCONFLICT_DATASET)}  类型: 5种更新场景")
    print("=" * 60)

    results_by_type = {}
    all_results = []

    for i, item in enumerate(MEMCONFLICT_DATASET):
        uid = item["user"]
        print(f"\n[{i+1:2d}/{len(MEMCONFLICT_DATASET)}] {item['id']} ({item['type']})")

        # 步骤1：写入旧事实
        try:
            m.add(item["old"], user_id=uid)
            print(f"  旧: {item['old'][:60]}")
        except Exception as e:
            print(f"  ❌ 旧事实写入失败: {e}")
            continue

        # 步骤2：写入新事实（UPDATE）
        try:
            m.add(item["new"], user_id=uid)
            print(f"  新: {item['new'][:60]}")
        except Exception as e:
            print(f"  ❌ 新事实写入失败: {e}")
            continue

        # 步骤3：查询
        t0 = time.perf_counter()
        try:
            res = m.search(item["query"], filters={"user_id": uid}, limit=1)
            lat = (time.perf_counter() - t0) * 1000
            result_list = res.get("results", []) if isinstance(res, dict) else res
            top = result_list[0]["memory"] if result_list else ""
            top_lower = top.lower()

            got_new   = item["expected_new"].lower() in top_lower
            got_old   = item["expected_old"].lower() in top_lower
            status = "✅ 新" if got_new else ("🕰️ 旧(Stale)" if got_old else "⚠️ 无关")
            print(f"  查: {item['query']}")
            print(f"  返: {top[:80]}")
            print(f"  → {status}  ({lat:.0f}ms)")

            result = {
                "id": item["id"], "type": item["type"],
                "got_new": got_new, "got_old": got_old,
                "retrieved": top, "latency_ms": lat
            }
            all_results.append(result)
            results_by_type.setdefault(item["type"], []).append(result)
        except Exception as e:
            print(f"  ❌ 查询失败: {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("📊 MemConflict 结果汇总")
    print("=" * 60)
    print(f"\n{'更新类型':<25} {'新事实率':>8} {'旧事实率':>8} {'无关':>6}")
    print("-" * 52)
    for update_type, res_list in results_by_type.items():
        new_rate   = sum(r["got_new"] for r in res_list) / len(res_list)
        old_rate   = sum(r["got_old"] for r in res_list) / len(res_list)
        other_rate = 1 - new_rate - old_rate
        print(f"  {update_type:<23} {new_rate:>7.0%} {old_rate:>8.0%} {other_rate:>6.0%}")

    total_new = sum(r["got_new"] for r in all_results)
    total_old = sum(r["got_old"] for r in all_results)
    n = len(all_results)
    print(f"\n{'总计':<25} {total_new/n:>7.0%} {total_old/n:>8.0%}")
    print(f"\n  ✅ 正确返回新事实: {total_new}/{n} ({total_new/n:.0%})")
    print(f"  🕰️  错误返回旧事实: {total_old}/{n} ({total_old/n:.0%}) ← Staleness Rate")

    out = {
        "system": "Mem0", "test": "memconflict",
        "timestamp": datetime.now().isoformat(),
        "n_total": n,
        "new_fact_rate": total_new / n,
        "staleness_rate": total_old / n,
        "by_type": {t: {
            "n": len(r), "new_rate": sum(x["got_new"] for x in r)/len(r),
            "old_rate": sum(x["got_old"] for x in r)/len(r),
        } for t, r in results_by_type.items()},
        "results": all_results,
    }
    out_path = Path("../results/memconflict_mem0_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")


if __name__ == "__main__":
    run_memconflict_test()
