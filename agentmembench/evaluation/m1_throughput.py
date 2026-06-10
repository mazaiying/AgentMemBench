"""
Step 13: QPS 吞吐量测试 — RQ3 深化
测试 Mem0 和 Naive RAG 在并发压力下的吞吐量上限。

指标：
  Write QPS = 成功写入数 / 总时间
  Read  QPS = 成功读取数 / 总时间
  Mixed QPS = (读+写)总数 / 总时间（R:W = 4:1）
"""
import os, time, json, statistics, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        "collection_name": "qps_test", "embedding_model_dims": 1024, "on_disk": False,
    }},
    "version": "v1.1"
}

WRITE_TEXTS = [
    f"QPS test memory #{i}: User preferences and activity record number {i}."
    for i in range(50)
]
READ_QUERIES = [
    "What are the user's preferences?",
    "What activity was recorded?",
    "Tell me about the user.",
    "What memory was stored?",
    "Find relevant user information.",
]


def test_mem0_qps():
    from mem0 import Memory
    m = Memory.from_config(MEM0_CONFIG)

    print("\n── Mem0 QPS 测试 ──────────────────────────────")

    # 先写入 20 条供读取测试用
    print("[预热] 写入 20 条基础记忆...")
    for i in range(20):
        try: m.add(WRITE_TEXTS[i], user_id="qps_user")
        except: pass

    results = {}

    # 1. Write QPS (并发度 1/3/5/10)
    print("\n[1] Write QPS 测试（不同并发度）")
    for concurrency in [1, 3, 5]:
        texts = WRITE_TEXTS[20:20+concurrency*3]  # 每个并发度测 3 轮
        latencies = []
        t_start = time.perf_counter()
        success = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(lambda t=txt: m.add(t, user_id="qps_user"), ) for txt in texts]
            for fut in as_completed(futs):
                try:
                    fut.result(); success += 1
                    latencies.append(1)  # placeholder
                except: pass
        elapsed = time.perf_counter() - t_start
        qps = success / elapsed
        print(f"  并发度={concurrency:2d}  成功={success}/{len(texts)}  耗时={elapsed:.1f}s  QPS={qps:.2f}")
        results[f"write_qps_c{concurrency}"] = {"qps": qps, "success": success, "elapsed_s": elapsed}

    # 2. Read QPS (并发度 1/5/10)
    print("\n[2] Read QPS 测试（不同并发度）")
    for concurrency in [1, 5, 10]:
        queries = READ_QUERIES * 4  # 20 次查询
        latencies = []
        t_start = time.perf_counter()
        success = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(lambda q=query: m.search(q, filters={"user_id": "qps_user"}, limit=1)) for query in queries]
            for fut in as_completed(futs):
                try:
                    fut.result(); success += 1
                except: pass
        elapsed = time.perf_counter() - t_start
        qps = success / elapsed
        print(f"  并发度={concurrency:2d}  成功={success}/{len(queries)}  耗时={elapsed:.1f}s  QPS={qps:.2f}")
        results[f"read_qps_c{concurrency}"] = {"qps": qps, "success": success, "elapsed_s": elapsed}

    return results


def test_naive_rag_qps():
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

    oai = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)
    qdrant = QdrantClient(":memory:")
    COL = "qps_naive"; DIM = 1024
    if qdrant.collection_exists(COL): qdrant.delete_collection(COL)
    qdrant.create_collection(COL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    pid = [0]; pid_lock = threading.Lock()

    def write_one(text):
        vec = oai.embeddings.create(model="text-embedding-v3", input=text).data[0].embedding
        with pid_lock:
            p = pid[0]; pid[0] += 1
        qdrant.upsert(COL, points=[PointStruct(id=p, vector=vec, payload={"text": text, "user_id": "qps_user"})])

    def read_one(query):
        q_vec = oai.embeddings.create(model="text-embedding-v3", input=query).data[0].embedding
        return qdrant.query_points(collection_name=COL, query=q_vec, limit=1,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value="qps_user"))])).points

    print("\n── Naive RAG QPS 测试 ──────────────────────────")

    # 预热写入
    print("[预热] 写入 20 条基础记忆...")
    for i in range(20): write_one(WRITE_TEXTS[i])

    results = {}

    print("\n[1] Write QPS 测试")
    for concurrency in [1, 5, 10]:
        texts = WRITE_TEXTS[20:20+concurrency*3]
        t_start = time.perf_counter(); success = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(write_one, txt) for txt in texts]
            for fut in as_completed(futs):
                try: fut.result(); success += 1
                except: pass
        elapsed = time.perf_counter() - t_start
        qps = success / elapsed
        print(f"  并发度={concurrency:2d}  成功={success}/{len(texts)}  耗时={elapsed:.1f}s  QPS={qps:.2f}")
        results[f"write_qps_c{concurrency}"] = {"qps": qps, "success": success, "elapsed_s": elapsed}

    print("\n[2] Read QPS 测试")
    for concurrency in [1, 5, 10]:
        queries = READ_QUERIES * 4
        t_start = time.perf_counter(); success = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(read_one, q) for q in queries]
            for fut in as_completed(futs):
                try: fut.result(); success += 1
                except: pass
        elapsed = time.perf_counter() - t_start
        qps = success / elapsed
        print(f"  并发度={concurrency:2d}  成功={success}/{len(queries)}  耗时={elapsed:.1f}s  QPS={qps:.2f}")
        results[f"read_qps_c{concurrency}"] = {"qps": qps, "success": success, "elapsed_s": elapsed}

    return results


def run():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    print("=" * 60)
    print("MemSysBench QPS 吞吐量测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    mem0_results = test_mem0_qps()
    rag_results  = test_naive_rag_qps()

    print("\n" + "=" * 60)
    print("📊 QPS 对比汇总")
    print(f"{'指标':<25} {'Mem0':>10} {'Naive RAG':>12}")
    print("-" * 50)
    for key in ["write_qps_c1", "write_qps_c5", "read_qps_c1", "read_qps_c5", "read_qps_c10"]:
        m_qps = mem0_results.get(key, {}).get("qps", 0)
        r_qps = rag_results.get(key, {}).get("qps", 0)
        print(f"  {key:<23} {m_qps:>9.2f} {r_qps:>11.2f}")

    out = {"timestamp": datetime.now().isoformat(),
           "mem0": mem0_results, "naive_rag": rag_results}
    out_path = Path("../results/qps_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")

if __name__ == "__main__":
    run()
