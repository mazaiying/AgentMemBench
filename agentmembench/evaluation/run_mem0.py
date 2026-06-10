"""
Step 2: Mem0 最小可验证实验 (Mini Pilot) — 千问版
运行方式: python step2_pilot_mem0.py
目的: 
  1. 往 Mem0 写入 10 条记忆
  2. 执行 5 次检索
  3. 打印延迟数字，确认系统正常工作
  
使用千问（通义千问）API，国内直连，无需梯子。
获取 API Key：https://dashscope.console.aliyun.com/
"""

import os
import time
import json
from datetime import datetime

# ── 配置区（修改这里） ─────────────────────────────────────
# 千问 API Key（在阿里云灵积 https://dashscope.console.aliyun.com/ 获取）
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# DashScope OpenAI 兼容接口地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Mem0 标准配置（论文 §4 中会固定这套参数）
# 使用千问 qwen-plus（性价比最高）+ text-embedding-v3
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",                      # 用 openai provider，指向千问接口
        "config": {
            "model": "qwen-plus",                   # 千问Plus，性价比最高
            "temperature": 0,                       # 固定为 0，保证可复现
            "api_key": QWEN_API_KEY,
            "openai_base_url": DASHSCOPE_BASE_URL,  # 关键：指向阿里云接口
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v3",           # 千问 Embedding 模型
            "api_key": QWEN_API_KEY,
            "openai_base_url": DASHSCOPE_BASE_URL,
            "embedding_dims": 1024,                 # text-embedding-v3 默认维度
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "memsysbench_pilot",
            "embedding_model_dims": 1024,           # 和 embedder 维度一致
            "on_disk": False,                       # in-memory 模式，不需要 Docker
        }
    },
    "version": "v1.1"
}

# ── 测试数据（模拟真实用户对话记忆）──────────────────────────
SAMPLE_MEMORIES = [
    {"user_id": "user_001", "text": "我叫马在营，是中国人民大学的博士生，研究方向是大语言模型。"},
    {"user_id": "user_001", "text": "我喜欢用 Python 写代码，不喜欢 Java。"},
    {"user_id": "user_001", "text": "我的导师是张老师，每周四下午开组会。"},
    {"user_id": "user_001", "text": "我正在写一篇投 ICDE 2027 的论文，方向是 Agent Memory Benchmark。"},
    {"user_id": "user_001", "text": "我住在北京，喜欢吃火锅，不能吃辣。"},
    {"user_id": "user_002", "text": "用户A是产品经理，在上海工作，负责 AI 产品线。"},
    {"user_id": "user_002", "text": "用户A上次说他们公司计划明年 Q2 上线新功能。"},
    {"user_id": "user_002", "text": "用户A更新：他已经从产品经理转岗到了技术总监。"},  # UPDATE 场景
    {"user_id": "user_003", "text": "这个对话讨论的是 KV Cache 优化，涉及 vLLM 和 SGLang 框架。"},
    {"user_id": "user_003", "text": "KV Cache 中最大的问题是语义相似的 prompt 无法命中精确缓存。"},
]

SAMPLE_QUERIES = [
    # expected_keywords: 中英文都接受（因为 Mem0 会把中文翻译成英文存储）
    {"user_id": "user_001", "query": "这个用户的研究方向是什么？",
     "expected_cn": "大语言模型", "expected_en": "large language model"},
    {"user_id": "user_001", "query": "用户喜欢哪种编程语言？",
     "expected_cn": "Python",     "expected_en": "python"},
    {"user_id": "user_001", "query": "用户在投哪个会议的论文？",
     "expected_cn": "ICDE",       "expected_en": "icde"},
    # ⚠️ 时序更新测试：期望返回新职位"技术总监"，而不是旧的"产品经理"
    {"user_id": "user_002", "query": "用户A现在的职位是什么？",
     "expected_cn": "技术总监",    "expected_en": "technical director",
     "old_cn": "产品经理",         "old_en": "product manager"},  # 旧事实（不应返回）
    {"user_id": "user_003", "query": "KV Cache 的核心问题是什么？",
     "expected_cn": "语义相似",    "expected_en": "semantic"},
]


def run_pilot():
    """运行 Mem0 最小验证实验"""
    
    if not QWEN_API_KEY:
        print("❌ 错误：请先设置 DASHSCOPE_API_KEY")
        print("   运行: export DASHSCOPE_API_KEY='sk-你的千问key'")
        print("   获取地址: https://dashscope.console.aliyun.com/")
        return

    print("=" * 60)
    print("MemSysBench Mini Pilot — Mem0 系统测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 初始化 Mem0
    print("\n[1/3] 初始化 Mem0 系统...")
    try:
        from mem0 import Memory
        m = Memory.from_config(MEM0_CONFIG)
        print("✅ Mem0 初始化成功")
    except Exception as e:
        print(f"❌ Mem0 初始化失败: {e}")
        return

    # 2. 写入记忆（测 Write Latency）
    print(f"\n[2/3] 写入 {len(SAMPLE_MEMORIES)} 条记忆...")
    write_latencies = []
    
    for i, mem in enumerate(SAMPLE_MEMORIES):
        t_start = time.perf_counter()
        try:
            result = m.add(mem["text"], user_id=mem["user_id"])
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            write_latencies.append(latency_ms)
            print(f"  [{i+1:2d}] ✅ 写入成功  延迟: {latency_ms:6.0f}ms  | {mem['text'][:40]}...")
        except Exception as e:
            t_end = time.perf_counter()
            print(f"  [{i+1:2d}] ❌ 写入失败: {e}")

    # 3. 检索测试（测 Read Latency）
    print(f"\n[3/3] 执行 {len(SAMPLE_QUERIES)} 次检索...")
    read_latencies = []
    results_log = []
    
    for i, q in enumerate(SAMPLE_QUERIES):
        t_start = time.perf_counter()
        try:
            memories = m.search(q["query"], filters={"user_id": q["user_id"]}, limit=3)
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            read_latencies.append(latency_ms)
            
            # Mem0 v2 返回格式：{"results": [{"memory": ..., "score": ...}, ...]}
            result_list = memories.get("results", []) if isinstance(memories, dict) else memories
            top_result = result_list[0]["memory"] if result_list else "（无结果）"
            top_lower = top_result.lower()

            # 双语命中判断（Mem0 会把中文翻译成英文存储）
            hit = (q["expected_cn"].lower() in top_lower or
                   q["expected_en"].lower() in top_lower)

            # 时序过时检测（仅针对有 old_cn/old_en 字段的查询）
            stale = False
            if "old_en" in q:
                stale = (q["old_cn"].lower() in top_lower or
                         q["old_en"].lower() in top_lower)

            status = "✅" if hit else ("🕰️ 过时" if stale else "⚠️ ")
            print(f"  [{i+1}] {status} 查询: {q['query']}")
            print(f"       返回: {top_result[:90]}")
            if stale:
                print(f"       ⚠️  返回了旧事实！(staleness 问题)")
            print(f"       延迟: {latency_ms:.0f}ms  |  命中: {'是' if hit else '否'}  |  过时: {'是' if stale else '否'}")

            results_log.append({
                "query": q["query"],
                "expected_cn": q["expected_cn"],
                "expected_en": q["expected_en"],
                "retrieved": top_result,
                "hit": hit,
                "stale": stale,
                "latency_ms": latency_ms
            })
        except Exception as e:
            t_end = time.perf_counter()
            print(f"  [{i+1}] ❌ 检索失败: {e}")

    # 4. 汇总统计
    print("\n" + "=" * 60)
    print("📊 实验结果汇总")
    print("=" * 60)
    
    if write_latencies:
        import statistics
        print(f"写入延迟 (n={len(write_latencies)}):")
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
        print(f"检索命中率: {hit_rate:.0%} ({sum(r['hit'] for r in results_log)}/{len(results_log)})")
    
    # 5. 保存结果到 JSON
    output = {
        "system": "Mem0",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm": "qwen-plus",
            "embedder": "text-embedding-v3",
            "vector_store": "qdrant (in-memory)"
        },
        "write_latencies_ms": write_latencies,
        "read_latencies_ms": read_latencies,
        "query_results": results_log,
    }
    
    out_path = "../results/pilot_mem0_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {out_path}")
    print("\n✅ Pilot 完成！下一步: 运行 step3_pilot_naive_rag.py 测对比基线")


if __name__ == "__main__":
    run_pilot()
