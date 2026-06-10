"""
Step 1: 环境检查脚本
运行方式: python step1_check_env.py
目的: 确认所有依赖都安装正确
"""

import sys
import importlib

print(f"Python 版本: {sys.version}")
print("=" * 50)

checks = [
    ("openai",      "OpenAI API 客户端"),
    ("mem0",        "Mem0 Memory 系统"),
    ("qdrant_client", "Qdrant 向量数据库"),
]

all_ok = True
for module, desc in checks:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"✅ {desc:<25} ({module} v{version})")
    except ImportError:
        print(f"❌ {desc:<25} ({module}) — 未安装")
        all_ok = False

print("=" * 50)

# 检查 OpenAI API Key
import os
api_key = os.environ.get("OPENAI_API_KEY", "")
if api_key and api_key.startswith("sk-"):
    print(f"✅ OPENAI_API_KEY 已设置 (sk-...{api_key[-4:]})")
else:
    print("⚠️  OPENAI_API_KEY 未设置 — 请在终端运行: export OPENAI_API_KEY='sk-你的key'")
    all_ok = False

print("=" * 50)
if all_ok:
    print("🎉 所有检查通过！可以运行 step2_pilot_mem0.py")
else:
    print("⚠️  请先解决上面的问题再继续")
