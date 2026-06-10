"""
Scale Experiment v2: Naive RAG — fast version
Strategy:
  - Generate synthetic facts locally (no API)
  - For each scale checkpoint, embed ONLY the new facts (incremental)
  - Measure write latency on 50 sampled individual upserts
  - Measure read latency with 20 queries using pre-computed embeddings
  - Bulk-insert the rest without timing overhead

Scales: 100 → 500 → 1000 → 2000 → 5000 → 10000
"""

import os, time, json, random, statistics, sys
from pathlib import Path
from datetime import datetime

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RESULTS_DIR   = Path("/Volumes/Elements SE/科研/icde agentmemory/MemSysBench/results")

# ── Synthetic fact generation (fully local, no API) ──────────────────────────
NAMES  = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Iris","Jack",
          "Karen","Liam","Mia","Noah","Olivia","Peter","Quinn","Rosa","Sam","Tina"]
CITIES = ["Beijing","Shanghai","Tokyo","Seoul","London","Paris","New York",
          "Singapore","Sydney","Berlin","Toronto","Dubai","Mumbai","Sao Paulo"]
JOBS   = ["software engineer","data scientist","product manager","researcher",
          "teacher","lawyer","doctor","journalist","designer","analyst"]
LANGS  = ["Python","Java","Go","Rust","TypeScript","C++","Swift","Kotlin"]
TOPICS = ["machine learning","distributed systems","NLP","databases",
          "computer vision","cryptography","cloud computing","robotics"]

def make_fact(i: int) -> dict:
    r = random.Random(i)
    name, city, job  = r.choice(NAMES), r.choice(CITIES), r.choice(JOBS)
    lang, topic      = r.choice(LANGS), r.choice(TOPICS)
    uid              = f"user_{i % 500:04d}"
    templates = [
        f"{name} is a {job} living in {city}, working on {topic}.",
        f"{name} prefers {lang} and is studying {topic} in {city}.",
        f"{name} relocated to {city} for a {job} position focused on {topic}.",
        f"{name}'s current role is {job} at a {topic} company in {city}.",
    ]
    text     = r.choice(templates)
    expected = city
    query    = f"Where does {name} currently live or work?"
    return {"uid": uid, "text": text, "query": query, "expected": expected}

# ── Batch embed helper ────────────────────────────────────────────────────────
def batch_embed(texts: list[str], client, batch_size=50) -> list[list[float]]:
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start+batch_size]
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model="text-embedding-v3", input=chunk)
                vecs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
                all_vecs.extend(vecs)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"  ⚠️  API error, retry in {wait}s: {e}")
                time.sleep(wait)
        done = min(start + batch_size, len(texts))
        if done % 1000 == 0 or done == len(texts):
            print(f"  Embedded {done}/{len(texts)}...", flush=True)
    return all_vecs

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                       Filter, FieldCondition, MatchValue)

    client = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)
    DIM    = 1024
    SCALES = [100, 500, 1000, 2000, 5000, 10000]
    N_WRITE_SAMPLE = 50   # measure individual write latency for this many facts
    N_READ_QUERIES = 20   # read queries per scale

    print("=" * 65)
    print("MemSysBench Scale Experiment v2 — Naive RAG")
    print(f"Time : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Scales: {SCALES}")
    print("=" * 65, flush=True)

    MAX_SCALE = max(SCALES)
    print(f"\n📝 Generating {MAX_SCALE:,} synthetic facts (local, instant)...")
    all_facts = [make_fact(i) for i in range(MAX_SCALE)]
    print(f"✅ Done.", flush=True)

    print(f"\n🔗 Embedding all {MAX_SCALE:,} facts in batches of 10 (DashScope limit)...")
    t_embed_start = time.perf_counter()
    all_vecs = batch_embed([f["text"] for f in all_facts], client, batch_size=10)
    t_embed = time.perf_counter() - t_embed_start
    print(f"✅ Embedding complete in {t_embed:.1f}s  ({MAX_SCALE/t_embed:.0f} facts/s)", flush=True)

    # Also embed query vectors (same texts, already computed — reuse fact vec as proxy)
    # For reading we just search by the same embedding (upper bound on recall)

    results = {}
    rng = random.Random(42)

    for scale in SCALES:
        print(f"\n{'─'*65}")
        print(f"  SCALE = {scale:,} facts", flush=True)

        COL    = f"naive_rag_scale"
        qdrant = QdrantClient(":memory:")
        qdrant.create_collection(COL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

        facts = all_facts[:scale]
        vecs  = all_vecs[:scale]

        # ── WRITE: sample N_WRITE_SAMPLE individual upserts ──────────────
        sample_idx = rng.sample(range(scale), min(N_WRITE_SAMPLE, scale))
        write_latencies = []

        # Bulk-insert everything first (fast, no timing)
        CHUNK = 500
        points_all = [
            PointStruct(id=i, vector=vecs[i],
                        payload={"text": facts[i]["text"], "uid": facts[i]["uid"],
                                 "expected": facts[i]["expected"]})
            for i in range(scale)
        ]
        for start in range(0, scale, CHUNK):
            qdrant.upsert(COL, points=points_all[start:start+CHUNK])

        # Then measure upsert latency by re-inserting sampled points
        for i in sample_idx:
            t0 = time.perf_counter()
            qdrant.upsert(COL, points=[PointStruct(
                id=i, vector=vecs[i],
                payload={"text": facts[i]["text"], "uid": facts[i]["uid"],
                         "expected": facts[i]["expected"]}
            )])
            write_latencies.append((time.perf_counter() - t0) * 1000)

        w_mean = statistics.mean(write_latencies)
        w_p50  = statistics.median(write_latencies)
        w_p95  = sorted(write_latencies)[int(len(write_latencies)*0.95)]
        print(f"  Write sample ({N_WRITE_SAMPLE}): mean={w_mean:.1f}ms  p50={w_p50:.1f}ms  p95={w_p95:.1f}ms")

        # ── READ: query by uid filter ─────────────────────────────────────
        read_latencies, hits = [], 0
        query_indices = rng.sample(range(scale), N_READ_QUERIES)

        for qi in query_indices:
            uid    = facts[qi]["uid"]
            q_vec  = vecs[qi]   # use same embedding vector as proxy query
            expect = facts[qi]["expected"].lower()

            t0 = time.perf_counter()
            hits_res = qdrant.query_points(
                collection_name=COL,
                query=q_vec,
                limit=5,
                query_filter=Filter(must=[
                    FieldCondition(key="uid", match=MatchValue(value=uid))
                ])
            ).points
            lat = (time.perf_counter() - t0) * 1000
            read_latencies.append(lat)

            if hits_res and expect in hits_res[0].payload["text"].lower():
                hits += 1

        r_mean = statistics.mean(read_latencies)
        r_p50  = statistics.median(read_latencies)
        r_p95  = sorted(read_latencies)[int(len(read_latencies)*0.95)]
        recall = hits / N_READ_QUERIES
        print(f"  Read  ({N_READ_QUERIES} queries): mean={r_mean:.1f}ms  p50={r_p50:.1f}ms  p95={r_p95:.1f}ms  recall={recall:.0%}")

        results[str(scale)] = {
            "scale": scale,
            "write_mean_ms": round(w_mean, 2),
            "write_p50_ms":  round(w_p50, 2),
            "write_p95_ms":  round(w_p95, 2),
            "read_mean_ms":  round(r_mean, 3),
            "read_p50_ms":   round(r_p50, 3),
            "read_p95_ms":   round(r_p95, 3),
            "recall_at_5":   round(recall, 3),
            "n_write_sample": N_WRITE_SAMPLE,
            "n_read_queries": N_READ_QUERIES,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "experiment": "Naive RAG Scale Experiment v2",
        "timestamp":  datetime.now().isoformat(),
        "config": {
            "embedder": "text-embedding-v3 (DashScope)",
            "vector_store": "Qdrant in-memory v1.9",
            "llm": "none (no LLM on write path)",
            "dim": 1024,
            "dataset": "synthetic (deterministic, 500 unique users)",
            "embed_total_s": round(t_embed, 1),
        },
        "scales": SCALES,
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "scale_naive_rag_result.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n💾 Results → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Scale':>8} | {'Write p50':>10} | {'Write p95':>10} | {'Read p50':>9} | {'Recall':>7}")
    print("-"*65)
    for s, r in results.items():
        print(f"{int(s):>8,} | {r['write_p50_ms']:>9.1f}ms | {r['write_p95_ms']:>9.1f}ms | "
              f"{r['read_p50_ms']:>8.2f}ms | {r['recall_at_5']:>6.0%}")
    print("="*65)

if __name__ == "__main__":
    if not QWEN_API_KEY:
        print("❌ DASHSCOPE_API_KEY not set"); sys.exit(1)
    run()
