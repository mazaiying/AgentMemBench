"""
Step 3: 下载并预处理 LMSYS-Chat-1M 数据集
运行方式: python step3_download_lmsys.py

功能：
  1. 下载 LMSYS-Chat-1M（1M条对话，1.49GB）
  2. 过滤出多轮对话（≥5轮）
  3. 保存为 MemDialogue 格式，用于后续 benchmark

运行前需要：
  export HF_TOKEN='hf_你的token'
"""

import os
import json
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

def download_and_preview():
    """先小规模预览，确认数据格式正确"""
    
    if not HF_TOKEN:
        print("❌ 请先设置 HF_TOKEN")
        print("   export HF_TOKEN='hf_你的token'")
        return

    print("=" * 60)
    print("正在加载 LMSYS-Chat-1M 数据集（小规模预览）...")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 请先安装: pip install datasets")
        return

    # 先只加载前 1000 条预览（streaming 模式，不下载全部）
    print("\n[1/3] 使用 streaming 模式预览前 1000 条...")
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        token=HF_TOKEN
    )

    samples = []
    for i, row in enumerate(ds):
        if i >= 1000:
            break
        samples.append(row)

    print(f"✅ 预览加载完成，共 {len(samples)} 条")

    # 分析数据结构
    print("\n[2/3] 分析数据结构...")
    first = samples[0]
    print(f"字段: {list(first.keys())}")
    print(f"\n示例对话（conversation_id: {first['conversation_id']}）:")
    print(f"  模型: {first['model']}")
    print(f"  轮数: {len(first['conversation'])}")
    for turn in first['conversation'][:3]:
        role = turn['role']
        content = turn['content'][:60]
        print(f"  [{role}]: {content}...")

    # 统计轮数分布
    turn_counts = [len(s['conversation']) for s in samples]
    multi_turn = [t for t in turn_counts if t >= 5]
    print(f"\n[3/3] 统计（前1000条）:")
    print(f"  平均轮数: {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"  ≥5轮的对话: {len(multi_turn)} / {len(samples)} ({len(multi_turn)/len(samples):.0%})")
    print(f"  ≥10轮的对话: {len([t for t in turn_counts if t>=10])} 条")

    # 过滤多轮对话并保存预览
    filtered = [s for s in samples if len(s['conversation']) >= 5]
    
    preview_path = DATA_DIR / "lmsys_preview_1k.jsonl"
    with open(preview_path, "w", encoding="utf-8") as f:
        for s in filtered:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n💾 预览数据已保存: {preview_path}")
    print(f"   共 {len(filtered)} 条多轮对话（≥5轮）")
    print("\n✅ 预览成功！运行 step4_build_memdialogue.py 构建完整数据集")


def download_full():
    """下载完整数据集（1.49GB）到 data/ 目录"""
    print("正在下载完整 LMSYS-Chat-1M 数据集（1.49GB）...")
    print("存储位置: ../data/hf_cache/")
    from datasets import load_dataset
    
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        token=HF_TOKEN,
        cache_dir=str(DATA_DIR / "hf_cache")   # → MemSysBench/data/hf_cache/
    )
    print(f"✅ 下载完成！共 {len(ds['train'])} 条对话")
    return ds


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        download_full()
    else:
        download_and_preview()
