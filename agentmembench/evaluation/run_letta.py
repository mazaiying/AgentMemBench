"""
Step 17: Letta Cloud Pilot + MemConflict 测试（最终版）
API Key: 通过 LETTA_API_KEY 环境变量传入
"""
import os, time, json, statistics
from pathlib import Path
from datetime import datetime

LETTA_API_KEY = os.environ.get("LETTA_API_KEY", "")

PILOT_MEMORIES = [
    "Ma Zaiying is a doctoral student at Renmin University researching LLM agent memory systems.",
    "User prefers Python and dislikes JavaScript.",
    "User is writing a benchmark paper for ICDE 2027 called MemSysBench.",
    "User lives in Beijing and enjoys hotpot.",
    "The core research question is how memory systems handle temporal updates.",
]
PILOT_QUERIES = [
    ("What is the user's research topic?",         "memory"),
    ("What programming language does user prefer?", "python"),
    ("What paper is the user writing?",             "memsysbench"),
]

CONFLICT_SUBSET = [
    {"id":"mc001","type":"job_change",
     "old":"Sarah is a junior software engineer at TechCorp.",
     "new":"Sarah has been promoted to senior software engineer at TechCorp.",
     "query":"What is Sarah's job title?",
     "expected_new":"senior software engineer","expected_old":"junior software engineer"},
    {"id":"mc011","type":"location_change",
     "old":"David lives in New York City.",
     "new":"David moved to San Francisco for a new job.",
     "query":"Where does David live?",
     "expected_new":"san francisco","expected_old":"new york"},
    {"id":"mc021","type":"numeric_update",
     "old":"The project has a budget of $1 million.",
     "new":"The project budget was increased to $2.5 million.",
     "query":"What is the project budget?",
     "expected_new":"2.5 million","expected_old":"1 million"},
    {"id":"mc031","type":"preference_change",
     "old":"User prefers dark mode in all applications.",
     "new":"User switched to light mode after eye strain issues.",
     "query":"What display mode does the user prefer?",
     "expected_new":"light mode","expected_old":"dark mode"},
    {"id":"mc041","type":"status_change",
     "old":"The paper is under review at ICDE 2027.",
     "new":"The paper was accepted at ICDE 2027.",
     "query":"What is the status of the paper?",
     "expected_new":"accepted","expected_old":"under review"},
]


def get_reply(response) -> str:
    for msg in getattr(response, "messages", []):
        for attr in ["text", "content"]:
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list):
                for block in val:
                    t = getattr(block, "text", None)
                    if t: return t.strip()
    return ""


def run():
    if not LETTA_API_KEY:
        print("❌ LETTA_API_KEY not set"); return

    from letta_client import Letta
    client = Letta(api_key=LETTA_API_KEY)

    print("=" * 60)
    print("MemSysBench — Letta Cloud Pilot")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ─── PART 1: Pilot ───────────────────────────────────────
    print("\n[PART 1] 基础 Pilot（5条记忆 + 3条检索）")
    agent = client.agents.create(name=f"memsys_pilot_{int(time.time())}", model="letta/auto")
    aid = agent.id
    print(f"✅ Agent: {aid[:20]}...")

    write_lats = []
    for mem in PILOT_MEMORIES:
        t0 = time.perf_counter()
        client.agents.messages.create(
            agent_id=aid,
            messages=[{"role": "user", "content": mem}]
        )
        lat = (time.perf_counter() - t0) * 1000
        write_lats.append(lat)
        print(f"  ✅ {lat:.0f}ms  {mem[:55]}...")

    read_lats, hits = [], 0
    for q, kw in PILOT_QUERIES:
        t0 = time.perf_counter()
        resp = client.agents.messages.create(
            agent_id=aid,
            messages=[{"role": "user", "content": q}]
        )
        lat = (time.perf_counter() - t0) * 1000
        read_lats.append(lat)
        reply = get_reply(resp)
        hit = kw.lower() in reply.lower()
        if hit: hits += 1
        print(f"  {'✅' if hit else '⚠️ '} {lat:.0f}ms  {q}")
        print(f"       → {reply[:90]}")

    # 删除 Pilot Agent，释放配额
    client.agents.delete(agent_id=aid)
    print(f"  🗑️  Pilot agent 已删除，释放配额")

    # ─── PART 2: MemConflict ─────────────────────────────────
    print(f"\n[PART 2] MemConflict（{len(CONFLICT_SUBSET)}条更新场景）")
    print("Letta MemGPT架构有显式记忆管理，预期 Staleness 低于其他系统")
    print("注意：检测逻辑 → 有新事实关键词=正确（即使也提及旧事实作为历史背景）")
    conflict_results = []

    for item in CONFLICT_SUBSET:
        print(f"\n[{item['id']}] {item['type']}")
        # 每个场景建一个独立 Agent，用完立刻删除
        ag = client.agents.create(name=f"mc_{item['id']}_{int(time.time())}", model="letta/auto")
        try:
            # 写旧事实
            client.agents.messages.create(
                agent_id=ag.id,
                messages=[{"role": "user", "content": item["old"]}]
            )
            print(f"  旧: {item['old'][:60]}")
            # 写新事实
            client.agents.messages.create(
                agent_id=ag.id,
                messages=[{"role": "user", "content": item["new"]}]
            )
            print(f"  新: {item['new'][:60]}")
            # 查询
            t0 = time.perf_counter()
            resp = client.agents.messages.create(
                agent_id=ag.id,
                messages=[{"role": "user", "content": item["query"]}]
            )
            lat = (time.perf_counter() - t0) * 1000
            reply = get_reply(resp)
            rl = reply.lower()
            got_new = item["expected_new"].lower() in rl
            # Staleness = 新事实没有出现 且 旧事实出现了
            # Letta 经常同时提及新旧（"原来是A，现在是B"），有新就算正确
            got_old_only = (not got_new) and (item["expected_old"].lower() in rl)
            got_old = item["expected_old"].lower() in rl  # 记录是否出现旧事实（不代表stale）
            status = "✅ 新" if got_new else ("🕰️ 旧(Stale)" if got_old_only else "⚠️ 无关")
            print(f"  → {status}  ({lat:.0f}ms)")
            print(f"     回答: {reply[:120]}")
            conflict_results.append({
                "id": item["id"], "type": item["type"],
                "got_new": got_new, "stale": got_old_only, "got_old_mentioned": got_old,
                "latency_ms": lat, "reply": reply[:200]
            })
        finally:
            client.agents.delete(agent_id=ag.id)
            print(f"  🗑️  Agent 已删除")

    # ─── 汇总 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Letta Cloud 结果汇总")
    print("=" * 60)
    print(f"写入延迟: avg={statistics.mean(write_lats):.0f}ms  p95={sorted(write_lats)[int(len(write_lats)*0.95)-1]:.0f}ms")
    print(f"检索延迟: avg={statistics.mean(read_lats):.0f}ms")
    print(f"命中率:   {hits}/{len(PILOT_QUERIES)} = {hits/len(PILOT_QUERIES):.0%}")
    n = len(conflict_results) if conflict_results else 1
    new_n  = sum(r["got_new"] for r in conflict_results)
    stale_n = sum(r["stale"]  for r in conflict_results)
    print(f"MemConflict ({len(conflict_results)}条): 新事实率={new_n/n:.0%}  真正Stale={stale_n/n:.0%}")

    out = {
        "system": "Letta Cloud", "timestamp": datetime.now().isoformat(),
        "llm": "letta/auto (Letta managed)", "note": "Letta Cloud free tier; detection: got_new=True if new kw in reply",
        "write_latencies_ms": write_lats, "read_latencies_ms": read_lats,
        "pilot_hit_rate": hits / len(PILOT_QUERIES),
        "memconflict": {
            "n": len(conflict_results),
            "new_fact_rate": new_n / n,
            "staleness_rate": stale_n / n,
            "results": conflict_results
        },
    }
    out_path = Path("../results/pilot_letta_cloud_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")


if __name__ == "__main__":
    run()
