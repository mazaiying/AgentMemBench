"""
Step 20: 语义漂移测试 (Semantic Drift Test)
测试：反复"读→改写→写回"循环后，记忆内容的保真度衰减情况。

模拟真实 Agent 行为：
  Round 0: 写入原始事实 F0
  Round k: 读取当前记忆 → LLM 改写（模拟 Agent 更新记忆） → 写回 Fk
  最终：比较 Fk 与 F0 的语义相似度，绘制衰减曲线

关键指标：
  - Semantic Fidelity per Round（每轮语义保真度，LLM-Judge 0-10分）
  - Drift Rate（信息关键字在第K轮的残留率）
  - First Corruption Round（首次出现信息失真的轮次）
"""

import os, json, time, statistics
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from mem0 import Memory

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RESULTS_DIR   = Path("../results")
OUTPUT_PATH   = RESULTS_DIR / "semantic_drift_result.json"

N_ROUNDS = 8   # 迭代轮次
TEST_USER = "drift_test_user_001"

# 测试用原始事实（包含多个具体可验证的信息点）
ORIGINAL_FACTS = [
    {
        "id": "drift_f1",
        "fact": "Alice is a 32-year-old software engineer at Google Brain in Mountain View. She has a PhD in Computer Science from MIT, specializes in reinforcement learning, and earns $250,000 annually.",
        "keywords": ["32", "google brain", "mountain view", "phd", "mit", "reinforcement learning", "250,000"],
        "query": "Tell me everything you know about Alice's professional background."
    },
    {
        "id": "drift_f2",
        "fact": "Bob's project deadline is March 15, 2027. The project budget is exactly $1.35 million, the team has 8 members, and the client is Samsung Electronics Korea.",
        "keywords": ["march 15", "2027", "1.35 million", "8 members", "samsung electronics"],
        "query": "What are the details of Bob's project?"
    },
    {
        "id": "drift_f3",
        "fact": "Carol takes metformin 500mg twice daily for Type 2 diabetes, diagnosed in 2021. Her A1C level is 6.8%, and her next checkup is scheduled for July 10, 2026.",
        "keywords": ["metformin", "500mg", "diabetes", "2021", "6.8", "july 10"],
        "query": "What medical information do you have about Carol?"
    },
]

REWRITE_PROMPT = """You are an AI assistant updating your memory notes.

Current memory note: {current_memory}

Please rewrite this as a concise updated memory note, keeping all important facts accurate.
Just output the rewritten note, nothing else."""

FIDELITY_PROMPT = """Compare these two texts and rate how well Text B preserves the factual content of Text A.

Text A (Original): {original}
Text B (Current):  {current}

Rate on a scale of 0-10:
- 10: All facts perfectly preserved
- 7-9: Most facts preserved, minor wording changes
- 4-6: Some facts lost or changed
- 1-3: Major information loss
- 0: Completely different content

Output only a number (0-10)."""


def get_fidelity_score(original: str, current: str, client: OpenAI) -> float:
    try:
        r = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": FIDELITY_PROMPT.format(
                original=original, current=current[:500]
            )}],
            temperature=0, max_tokens=5,
        )
        return float(r.choices[0].message.content.strip())
    except:
        return 5.0


def keyword_retention(text: str, keywords: list) -> float:
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def rewrite_memory(text: str, client: OpenAI) -> str:
    try:
        r = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                current_memory=text
            )}],
            temperature=0.3, max_tokens=200,
        )
        return r.choices[0].message.content.strip()
    except:
        return text


def run():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    client = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)

    mem_config = {
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
            "path": "/tmp/qdrant_drift",
            "collection_name": "drift_test",
            "embedding_model_dims": 1024,
        }},
    }
    mem = Memory.from_config(mem_config)

    print("=" * 62)
    print("MemSysBench — 语义漂移测试 (Semantic Drift)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"系统: Mem0  |  轮次: {N_ROUNDS}  |  事实数: {len(ORIGINAL_FACTS)}")
    print("=" * 62)

    all_results = []

    for fact_obj in ORIGINAL_FACTS:
        fid       = fact_obj["id"]
        original  = fact_obj["fact"]
        keywords  = fact_obj["keywords"]
        query     = fact_obj["query"]
        uid       = f"{TEST_USER}_{fid}"

        print(f"\n▶ [{fid}] 原始事实:")
        print(f"  {original[:80]}...")

        # Round 0: 写入原始事实
        try:
            mem.add(original, user_id=uid)
            print(f"  [R0] ✅ 写入成功")
        except Exception as e:
            print(f"  [R0] ❌ 写入失败: {e}")
            continue

        rounds_data = []
        current_text = original

        for rnd in range(1, N_ROUNDS + 1):
            time.sleep(0.5)

            # Step 1: 读取当前记忆
            try:
                res = mem.search(query, filters={"user_id": uid}, limit=3)
                memories = res.get("results", [])
                retrieved = " ".join(m.get("memory", "") for m in memories[:3])
            except Exception as e:
                retrieved = current_text
                print(f"  [R{rnd}] ⚠️ 检索失败: {e}")

            if not retrieved.strip():
                retrieved = current_text

            # Step 2: LLM 改写（模拟 Agent 更新记忆）
            rewritten = rewrite_memory(retrieved, client)

            # Step 3: 写回改写后的版本（新增，不覆盖）
            try:
                mem.add(rewritten, user_id=uid)
            except Exception as e:
                print(f"  [R{rnd}] ⚠️ 写回失败: {e}")

            # Step 4: 评估保真度
            fidelity  = get_fidelity_score(original, rewritten, client)
            kw_retain = keyword_retention(rewritten, keywords)

            rounds_data.append({
                "round":     rnd,
                "retrieved": retrieved[:200],
                "rewritten": rewritten[:200],
                "fidelity":  fidelity,
                "kw_retention": kw_retain,
            })

            current_text = rewritten
            print(f"  [R{rnd}] 保真度={fidelity:.1f}/10  关键词保留={kw_retain:.0%}  "
                  f"内容={rewritten[:45]}...")

        # 汇总该事实的漂移曲线
        fidelities = [r["fidelity"] for r in rounds_data]
        kw_retains = [r["kw_retention"] for r in rounds_data]

        # 首次腐化轮次（保真度首次低于 7）
        first_corrupt = next(
            (r["round"] for r in rounds_data if r["fidelity"] < 7), None
        )

        all_results.append({
            "fact_id":           fid,
            "original":          original,
            "keywords":          keywords,
            "rounds":            rounds_data,
            "final_fidelity":    fidelities[-1] if fidelities else 0,
            "avg_fidelity":      statistics.mean(fidelities) if fidelities else 0,
            "final_kw_retention": kw_retains[-1] if kw_retains else 0,
            "first_corrupt_round": first_corrupt,
        })

        print(f"\n  → 最终保真度: {fidelities[-1]:.1f}/10  "
              f"首次腐化: R{first_corrupt}  "
              f"最终关键词保留: {kw_retains[-1]:.0%}")

    # ── 全局汇总 ──────────────────────────────────────────────
    avg_final_fidelity = statistics.mean(r["final_fidelity"] for r in all_results)
    avg_final_kw       = statistics.mean(r["final_kw_retention"] for r in all_results)
    corrupt_rounds     = [r["first_corrupt_round"] for r in all_results if r["first_corrupt_round"]]

    print("\n" + "=" * 62)
    print("📊 语义漂移测试汇总")
    print("=" * 62)
    print(f"事实数:            {len(all_results)}")
    print(f"迭代轮次:          {N_ROUNDS}")
    print(f"平均最终保真度:    {avg_final_fidelity:.1f}/10")
    print(f"平均最终关键词保留: {avg_final_kw:.0%}")
    if corrupt_rounds:
        print(f"首次腐化平均轮次:  R{statistics.mean(corrupt_rounds):.1f}")
    else:
        print(f"首次腐化:          未发生（保真度始终 ≥ 7）")

    # 逐轮衰减曲线
    print("\n  逐轮平均保真度曲线:")
    for rnd in range(1, N_ROUNDS + 1):
        scores = [r["rounds"][rnd-1]["fidelity"] for r in all_results if len(r["rounds"]) >= rnd]
        avg_s = statistics.mean(scores) if scores else 0
        bar = "█" * int(avg_s) + "░" * (10 - int(avg_s))
        print(f"  R{rnd}: {bar} {avg_s:.1f}")

    output = {
        "system":    "Mem0",
        "test":      "semantic_drift",
        "timestamp": datetime.now().isoformat(),
        "n_facts":   len(ORIGINAL_FACTS),
        "n_rounds":  N_ROUNDS,
        "summary": {
            "avg_final_fidelity":    avg_final_fidelity,
            "avg_final_kw_retention": avg_final_kw,
            "avg_first_corrupt_round": statistics.mean(corrupt_rounds) if corrupt_rounds else None,
        },
        "per_fact": all_results,
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n💾 结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
