"""
Step 4d: 扩展 MemDialogue 至 10,000 条（从 3,000 续跑）
- 完全增量：已有 3,000 条不会重复
- 每 50 条保存一次
- 支持 Ctrl+C 优雅退出
运行: python step4d_extend_to_10k.py
"""

import os, json, time, signal, sys
from pathlib import Path
from datetime import datetime

QWEN_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DATA_DIR      = Path(__file__).parent.parent / "data"
OUTPUT_PATH   = DATA_DIR / "memdialogue.jsonl"

TARGET_TOTAL    = 10000        # 目标总条数
MIN_TURNS       = 6
TARGET_LANGUAGE = "English"
SAVE_EVERY      = 50           # 每 50 条保存一次
HF_STREAM_LIMIT = 500000       # 最多扫描 50 万条原始数据

EXTRACT_PROMPT = """You are analyzing a conversation to build a memory benchmark dataset.

Given the following conversation excerpt, extract ONE factual memory event that:
1. Is a clear, verifiable fact about the user (NOT about AI capabilities)
2. Can be turned into a retrieval question with a definite short answer

Conversation:
{conversation}

Output JSON only, no markdown:
{{"fact": "<one sentence fact about the user>", "query": "<question to retrieve this fact>", "answer": "<short answer, 1-5 words>", "event_type": "INSERT"}}"""


def extract_event(turns, client):
    excerpt = "\n".join(
        f"[{t['role'].upper()}]: {t['content'][:200]}"
        for t in turns[:6]
    )
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(conversation=excerpt)}],
            temperature=0, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        result = json.loads(raw)
        # Basic validation
        if not result.get("fact") or not result.get("query") or not result.get("answer"):
            return None
        return result
    except Exception as e:
        return None


def load_existing():
    """读取已有进度，返回 (results列表, seen_ids集合)"""
    if not OUTPUT_PATH.exists():
        return [], set()
    results = []
    seen_ids = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                results.append(r)
                seen_ids.add(r["session_id"])
    return results, seen_ids


def save_results(results):
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def update_meta(results, new_this_run, scanned, skipped, api_errors):
    meta = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "new_this_run": new_this_run,
        "scanned": scanned,
        "skipped": skipped,
        "api_errors": api_errors,
    }
    (DATA_DIR / "memdialogue_meta.json").write_text(json.dumps(meta, indent=2))


def build():
    if not QWEN_API_KEY:
        print("❌ 请设置 DASHSCOPE_API_KEY"); return
    if not HF_TOKEN:
        print("❌ 请设置 HF_TOKEN"); return

    from openai import OpenAI
    from datasets import load_dataset

    client = OpenAI(api_key=QWEN_API_KEY, base_url=DASHSCOPE_URL)

    # 加载已有进度
    results, seen_ids = load_existing()
    already = len(results)
    print("=" * 60)
    print(f"MemDialogue 扩展至 {TARGET_TOTAL} 条")
    print(f"已有进度: {already} 条，目标: {TARGET_TOTAL} 条，还需: {TARGET_TOTAL - already} 条")
    print("=" * 60)

    if already >= TARGET_TOTAL:
        print(f"✅ 已达目标 {TARGET_TOTAL} 条，无需继续。")
        return

    # Ctrl+C 优雅退出
    interrupted = False
    def handle_interrupt(sig, frame):
        nonlocal interrupted
        print(f"\n⚠️  收到中断信号，正在保存当前进度（{len(results)} 条）...")
        save_results(results)
        update_meta(results, new_this_run, scanned, skipped, api_errors)
        print(f"✅ 已保存 {len(results)} 条到 {OUTPUT_PATH}")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_interrupt)

    print("\n加载 HF 数据集流（streaming）...")
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    scanned = 0
    skipped = 0
    api_errors = 0
    new_this_run = 0
    start_time = time.time()

    print(f"开始处理，已有 {already} 条，还需 {TARGET_TOTAL - already} 条...\n")

    for row in ds:
        if len(results) >= TARGET_TOTAL:
            break
        if scanned >= HF_STREAM_LIMIT:
            print(f"⚠️  已扫描 {scanned} 条原始数据，达到上限，停止。")
            break

        scanned += 1
        sid = f"lmsys_{row['conversation_id']}"

        # 跳过已处理
        if sid in seen_ids:
            continue

        turns = row.get("conversation", [])
        if len(turns) < MIN_TURNS or row.get("language", "") != TARGET_LANGUAGE:
            skipped += 1
            continue

        event = extract_event(turns, client)
        if event is None:
            api_errors += 1
            continue

        r = {
            "session_id": sid,
            "model":      row.get("model", "unknown"),
            "turns":      len(turns),
            "language":   row.get("language", ""),
            "memory_events": [{
                "turn_idx":     0,
                "event_type":   event.get("event_type", "INSERT"),
                "raw_text":     event.get("fact", ""),
                "query":        event.get("query", ""),
                "ground_truth": event.get("answer", ""),
            }]
        }
        results.append(r)
        seen_ids.add(sid)
        new_this_run += 1

        # 增量保存
        if new_this_run % SAVE_EVERY == 0:
            save_results(results)
            update_meta(results, new_this_run, scanned, skipped, api_errors)
            elapsed = time.time() - start_time
            rate = new_this_run / elapsed * 3600 if elapsed > 0 else 0
            remaining = TARGET_TOTAL - len(results)
            eta_h = remaining / rate if rate > 0 else 0
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] 共 {len(results)}/{TARGET_TOTAL} 条 | "
                  f"本次+{new_this_run} | 扫描={scanned} | 跳过={skipped} | "
                  f"API错={api_errors} | 速率={rate:.0f}条/h | ETA≈{eta_h:.1f}h")

    # 最终保存
    save_results(results)
    update_meta(results, new_this_run, scanned, skipped, api_errors)

    elapsed = time.time() - start_time
    print(f"\n✅ 完成！")
    print(f"   总计: {len(results)} 条（本次新增 {new_this_run} 条）")
    print(f"   扫描原始数据: {scanned} 条")
    print(f"   跳过（轮数/语言）: {skipped} 条")
    print(f"   API错误: {api_errors} 条")
    print(f"   耗时: {elapsed/3600:.2f} 小时")
    print(f"   输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
