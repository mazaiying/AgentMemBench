"""
Step 18: 记忆幻觉测试 (Memory Hallucination Test)
使用 MemDialogue 数据集验证：系统是否会捏造从未写入的事实。

三类幻觉：
  - Fabrication：返回从未写入的信息
  - Omission：已写入的信息无法检索到
  - Staleness：返回旧事实（已在 MemConflict 中测）

测试逻辑：
  1. 写入 ground_truth 事实
  2. 查询 → 检查返回内容是否语义匹配 ground_truth
  3. 查询"诱导性"（系统从未见过的）事实 → 检查是否被捏造返回
"""

import os, json, time, statistics
from pathlib import Path
from datetime import datetime
from openai import OpenAI

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DATA_DIR    = Path("../data")
RESULTS_DIR = Path("../results")
MEMDIALOGUE = DATA_DIR / "memdialogue.jsonl"
OUTPUT_PATH = RESULTS_DIR / "hallucination_mem0_result.json"

N_SAMPLES = 30  # 从 MemDialogue 取 30 条测试

# 诱导性查询（从未写入，用于测 Fabrication）
FABRICATION_PROBES = [
    ("user_halutest_{i}", "What is this user's favorite color?",  "blue"),
    ("user_halutest_{i}", "How many siblings does this user have?", "two"),
    ("user_halutest_{i}", "What city was this user born in?",     "Paris"),
]

JUDGE_PROMPT = """You are a strict evaluator for a memory system benchmark.

A fact was written to a memory system:
FACT: {fact}

The system was asked: {query}
The system replied: {reply}

Does the reply correctly capture the core content of the FACT?
Answer with only one word: YES or NO"""


def semantic_judge(fact: str, query: str, reply: str, client: OpenAI) -> bool:
    """用 LLM 判断返回内容是否语义匹配写入的事实"""
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                fact=fact, query=query, reply=reply[:300]
            )}],
            temperature=0, max_tokens=5,
        )
        ans = resp.choices[0].message.content.strip().upper()
        return ans.startswith("YES")
    except:
        return False


def run():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return
    if not MEMDIALOGUE.exists():
        print(f"❌ 找不到 {MEMDIALOGUE}"); return

    import mem0
    from mem0 import Memory
    client_llm = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)

    # 初始化 Mem0
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
            "path": "/tmp/qdrant_halu",
            "collection_name": "hallucination_test",
            "embedding_model_dims": 1024,
        }},
    }
    mem = Memory.from_config(config)

    print("=" * 60)
    print("MemSysBench — 记忆幻觉测试 (Hallucination Test)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {MEMDIALOGUE} ({N_SAMPLES} 条)")
    print("=" * 60)

    # 加载 MemDialogue 样本
    samples = []
    with open(MEMDIALOGUE, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            ev = obj["memory_events"][0]
            if ev["raw_text"] and ev["query"] and ev["ground_truth"]:
                samples.append({
                    "session_id": obj["session_id"],
                    "fact":       ev["raw_text"],
                    "query":      ev["query"],
                    "answer":     ev["ground_truth"],
                })
            if len(samples) >= N_SAMPLES:
                break

    print(f"\n[1/3] 写入 {len(samples)} 条 MemDialogue 事实...")
    write_lats = []
    for i, s in enumerate(samples):
        uid = f"halu_user_{i:03d}"
        t0 = time.perf_counter()
        try:
            mem.add(s["fact"], user_id=uid)
            lat = (time.perf_counter() - t0) * 1000
            write_lats.append(lat)
        except Exception as e:
            print(f"  ⚠️  写入失败: {e}")

    avg_write = statistics.mean(write_lats) if write_lats else 0
    print(f"  写入完成: {len(write_lats)} 条, avg={avg_write:.0f}ms")

    # [2/3] Omission 测试：查询已写入内容，看是否能检索到
    print(f"\n[2/3] Omission 测试（已写入 → 能否检索）...")
    omission_results = []
    for i, s in enumerate(samples[:N_SAMPLES]):
        uid = f"halu_user_{i:03d}"
        try:
            t0 = time.perf_counter()
            res = mem.search(s["query"], user_id=uid, limit=3)
            lat = (time.perf_counter() - t0) * 1000
            # 提取返回文本
            memories = res.get("results", res) if isinstance(res, dict) else res
            reply = " ".join(m.get("memory", m.get("text", "")) for m in memories[:3])
            # 语义判断
            hit = semantic_judge(s["fact"], s["query"], reply, client_llm)
            omission_results.append({"id": i, "hit": hit, "latency_ms": lat,
                                      "fact": s["fact"][:80], "reply": reply[:120]})
            status = "✅" if hit else "❌ OMISSION"
            if i < 5 or not hit:
                print(f"  [{i:02d}] {status}  {s['query'][:50]}")
                if not hit:
                    print(f"       事实: {s['fact'][:70]}")
                    print(f"       返回: {reply[:70]}")
        except Exception as e:
            omission_results.append({"id": i, "hit": False, "error": str(e)})

    omission_rate = 1 - sum(r["hit"] for r in omission_results) / max(len(omission_results), 1)
    print(f"\n  Omission Rate: {omission_rate:.0%}  ({sum(not r['hit'] for r in omission_results)}/{len(omission_results)} 条无法检索)")

    # [3/3] Fabrication 测试：查询从未写入的信息，看是否被捏造
    print(f"\n[3/3] Fabrication 测试（未写入 → 是否被捏造）...")
    fab_results = []
    for i in range(min(10, len(samples))):
        uid = f"halu_user_{i:03d}"
        # 查询该用户从未提到过的信息
        probes = [
            ("最喜欢的颜色是什么？", "favorite color"),
            ("有几个兄弟姐妹？", "siblings"),
            ("出生在哪个城市？", "birth city"),
        ]
        for q_cn, kw in probes:
            try:
                res = mem.search(q_cn, user_id=uid, limit=3)
                memories = res.get("results", res) if isinstance(res, dict) else res
                reply = " ".join(m.get("memory", m.get("text", "")) for m in memories[:3])
                fabricated = bool(reply.strip()) and kw.lower() in reply.lower()
                fab_results.append({
                    "user": uid, "query": q_cn, "reply": reply[:100],
                    "fabricated": fabricated
                })
                if fabricated:
                    print(f"  ⚠️  FABRICATION [{uid}] {q_cn} → {reply[:60]}")
            except Exception as e:
                fab_results.append({"user": uid, "query": q_cn, "error": str(e), "fabricated": False})

    fab_rate = sum(r["fabricated"] for r in fab_results) / max(len(fab_results), 1)
    print(f"\n  Fabrication Rate: {fab_rate:.0%}  ({sum(r['fabricated'] for r in fab_results)}/{len(fab_results)} 条被捏造)")

    # 汇总
    print("\n" + "=" * 60)
    print("📊 记忆幻觉测试汇总")
    print("=" * 60)
    print(f"系统: Mem0")
    print(f"样本: {len(samples)} 条 MemDialogue 事实")
    print(f"Omission Rate (已有事实无法检索): {omission_rate:.0%}")
    print(f"Fabrication Rate (未有事实被捏造): {fab_rate:.0%}")
    print(f"Recall Rate (正确检索率):          {1-omission_rate:.0%}")

    out = {
        "system": "Mem0",
        "test": "hallucination",
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(samples),
        "omission_rate": omission_rate,
        "fabrication_rate": fab_rate,
        "recall_rate": 1 - omission_rate,
        "write_avg_ms": statistics.mean(write_lats) if write_lats else 0,
        "omission_details": omission_results[:10],
        "fabrication_details": fab_results[:15],
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n💾 结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
