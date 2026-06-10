"""
Step 14: MemScale 规模测试 — RQ1 延伸
测试随着记忆数量增加（100→500→1000），延迟如何变化。

目的：
  1. 延迟是否随规模线性增长？（向量检索理论是 O(n) 或 O(log n)）
  2. Mem0 的 LLM 提取延迟是否受规模影响？
  3. Naive RAG 的向量检索延迟随规模变化趋势

只测检索延迟（写入已在并发测试中测完）。
"""
import os, time, json, statistics
from pathlib import Path
from datetime import datetime

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SCALE_POINTS = [100, 300, 500, 1000]  # 规模测试点

QUERY_TEMPLATES = [
    "What is the user's research interest?",
    "What programming language does the user prefer?",
    "Where does the user live?",
    "What is the user's job title?",
    "What projects is the user working on?",
]

MEM0_CONFIG_TMPL = lambda col: {
    "llm": {"provider": "openai", "config": {
        "model": "qwen-plus", "temperature": 0,
        "api_key": QWEN_API_KEY, "openai_base_url": DASHSCOPE_URL,
    }},
    "embedder": {"provider": "openai", "config": {
        "model": "text-embedding-v3", "api_key": QWEN_API_KEY,
        "openai_base_url": DASHSCOPE_URL, "embedding_dims": 1024,
    }},
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": col, "embedding_model_dims": 1024, "on_disk": False,
    }},
    "version": "v1.1"
}


def generate_synthetic_memories(n):
    """生成 n 条合成记忆，覆盖多种主题"""
    templates = [
        "User_{i} is a software engineer at Company_{i} working on Project_{i}.",
        "User_{i} lives in City_{i} and enjoys Activity_{i}.",
        "User_{i} prefers Language_{i} and uses Tool_{i} for development.",
        "User_{i} is attending Conference_{i} to present Paper_{i}.",
        "User_{i} is currently reading Book_{i} about Topic_{i}.",
    ]
    memories = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        memories.append({
            "user_id": f"scale_user_{i % 10:02d}",  # 10个用户
            "text": tmpl.replace("{i}", str(i))
        })
    return memories


def test_mem0_scale(scale):
    from mem0 import Memory
    col = f"memscale_{scale}"
    m = Memory.from_config(MEM0_CONFIG_TMPL(col))

    mems = generate_synthetic_memories(scale)

    # 写入
    print(f"  [Mem0 n={scale}] 写入 {scale} 条...")
    write_lats = []
    for i, mem in enumerate(mems):
        t0 = time.perf_counter()
        try:
            m.add(mem["text"], user_id=mem["user_id"])
            write_lats.append((time.perf_counter()-t0)*1000)
        except:
            pass
        if (i+1) % (scale//5) == 0:
            print(f"    {i+1}/{scale} 写入中...")

    # 检索
    read_lats = []
    for q in QUERY_TEMPLATES:
        for uid in ["scale_user_00", "scale_user_05"]:
            t0 = time.perf_counter()
            try:
                m.search(q, filters={"user_id": uid}, limit=3)
                read_lats.append((time.perf_counter()-t0)*1000)
            except:
                pass

    return write_lats, read_lats


def test_naive_rag_scale(scale):
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

    client = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)
    qdrant = QdrantClient(":memory:")
    COL = f"rag_scale_{scale}"; DIM = 1024
    if qdrant.collection_exists(COL): qdrant.delete_collection(COL)
    qdrant.create_collection(COL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    mems = generate_synthetic_memories(scale)

    # 批量写入（embedding API 支持批量）
    print(f"  [Naive RAG n={scale}] 写入 {scale} 条...")
    write_lats = []
    BATCH = 10
    for b in range(0, scale, BATCH):
        batch = mems[b:b+BATCH]
        texts = [m["text"] for m in batch]
        t0 = time.perf_counter()
        resp = client.embeddings.create(model="text-embedding-v3", input=texts)
        lat_per = (time.perf_counter()-t0)*1000 / len(batch)
        write_lats.extend([lat_per] * len(batch))
        points = [PointStruct(id=b+j, vector=resp.data[j].embedding,
                              payload={"text": batch[j]["text"], "user_id": batch[j]["user_id"]})
                  for j in range(len(batch))]
        qdrant.upsert(COL, points=points)
        if (b+BATCH) % (scale//5) == 0:
            print(f"    {min(b+BATCH, scale)}/{scale} 写入中...")

    # 检索
    read_lats = []
    for q in QUERY_TEMPLATES:
        q_vec = client.embeddings.create(model="text-embedding-v3", input=q).data[0].embedding
        for uid in ["scale_user_00", "scale_user_05"]:
            t0 = time.perf_counter()
            qdrant.query_points(collection_name=COL, query=q_vec, limit=3,
                query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=uid))])).points
            read_lats.append((time.perf_counter()-t0)*1000)

    return write_lats, read_lats


def run():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return

    print("=" * 60)
    print("MemSysBench MemScale 规模测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"规模测试点: {SCALE_POINTS}")
    print("=" * 60)

    results = {"mem0": {}, "naive_rag": {}}

    for scale in SCALE_POINTS:
        print(f"\n{'─'*50}")
        print(f"规模 n={scale}")
        print(f"{'─'*50}")

        # Naive RAG（快，先跑）
        rag_w, rag_r = test_naive_rag_scale(scale)
        results["naive_rag"][str(scale)] = {
            "write_avg_ms": statistics.mean(rag_w) if rag_w else None,
            "read_avg_ms":  statistics.mean(rag_r) if rag_r else None,
            "read_p95_ms":  sorted(rag_r)[int(len(rag_r)*0.95)] if rag_r else None,
        }
        print(f"  Naive RAG: 写={statistics.mean(rag_w):.0f}ms  读={statistics.mean(rag_r):.0f}ms")

        # Mem0（慢，只测到500条，1000条太贵了）
        if scale <= 500:
            m0_w, m0_r = test_mem0_scale(scale)
            results["mem0"][str(scale)] = {
                "write_avg_ms": statistics.mean(m0_w) if m0_w else None,
                "read_avg_ms":  statistics.mean(m0_r) if m0_r else None,
            }
            print(f"  Mem0:      写={statistics.mean(m0_w):.0f}ms  读={statistics.mean(m0_r):.0f}ms")
        else:
            print(f"  Mem0:      跳过（>500条写入耗时过长，估计>{scale*6/1000:.0f}分钟）")
            results["mem0"][str(scale)] = {"note": "skipped (too expensive)"}

    # 汇总
    print("\n" + "=" * 60)
    print("📊 MemScale 延迟趋势")
    print(f"{'规模':>6}  {'RAG写':>8}  {'RAG读':>8}  {'Mem0写':>9}  {'Mem0读':>9}")
    print("-" * 50)
    for s in SCALE_POINTS:
        r = results["naive_rag"].get(str(s), {})
        m = results["mem0"].get(str(s), {})
        rw = f"{r.get('write_avg_ms', 0):.0f}ms" if r.get("write_avg_ms") else "—"
        rr = f"{r.get('read_avg_ms', 0):.0f}ms"  if r.get("read_avg_ms")  else "—"
        mw = f"{m.get('write_avg_ms', 0):.0f}ms" if m.get("write_avg_ms") else "skip"
        mr = f"{m.get('read_avg_ms', 0):.0f}ms"  if m.get("read_avg_ms")  else "skip"
        print(f"{s:>6}  {rw:>8}  {rr:>8}  {mw:>9}  {mr:>9}")

    out_path = Path("../results/memscale_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), **results}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_path}")


if __name__ == "__main__":
    run()
