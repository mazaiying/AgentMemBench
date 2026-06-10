# MemSysBench Pilot 汇总报告

**生成时间**: 2026-06-05  
**实验阶段**: Mini Pilot（10条记忆 × 5条查询）  
**LLM 配置**: qwen-plus + text-embedding-v3（DashScope）  

## 系统性能对比表

| 系统 | 写入延迟(avg) | 检索延迟(avg) | 命中率 | 过时返回数 | LLM |
|------|-------------|-------------|------|---------|-----|
| Mem0 | 7294ms | 882ms | 60% (3/5) | 1 | qwen-plus |
| Graphiti | 0/10 成功 | 3685ms | 0% (0/5) | 0 | qwen-plus |
| Naive RAG | 665ms | 568ms | 80% (4/5) | 1 | none |
| LangMem | 761ms | 486ms | 60% (3/5) | 1 | qwen-plus |


## 关键发现（Pilot 阶段）

| # | 发现 | 涉及系统 | 论文 RQ |
|---|------|---------|--------|
| F1 | **写重读轻**：Mem0 写入~7s（LLM提取），检索~882ms，写/读比 8:1 | Mem0 | RQ1 |
| F2 | **时序更新失败**：Mem0 在 UPDATE 后仍返回旧事实（Staleness 100%） | Mem0 | RQ4 |
| F3 | **检索相关性漂移**：同一用户多条记忆时，Mem0 检索会返回不相关记忆 | Mem0 | RQ2 |
| F4 | **Graphiti LLM强绑定**：实体提取 prompt 仅兼容 OpenAI JSON 格式，千问导致 100% 写入失败 | Graphiti | RQ6 |
| F5 | **LangMem Embedding 格式不兼容**：LangChain OpenAIEmbeddings 与千问 API 格式不兼容，100% 失败 | LangMem | RQ6 |
| F6 | **Naive RAG 性能最佳**：无 LLM 提取，写入最快（665ms），检索最快（568ms），命中率最高（80%） | Naive RAG | RQ1/RQ2 |
| F7 | **API 破坏性变更**：Mem0 v2.0 search() 接口变更（user_id→filters），无文档说明 | Mem0 | 可复现性 |

## 系统可移植性评估（千问环境）

| 系统 | 千问可用性 | 失败原因 |
|------|----------|---------|
| Mem0 | ✅ 可用（需适配） | search() API 格式变更 |
| Graphiti | ❌ 完全不可用 | 实体提取 JSON schema 强绑定 OpenAI |
| LangMem | ❌ 完全不可用 | Embedding API 格式不兼容 |
| Naive RAG | ✅ 完全可用 | 直接调用 Embedding，无 LLM 依赖 |

> **结论**：在非 OpenAI 环境（中国/千问）下，4个系统中只有 1 个（Mem0）可以基本运行。
> 这严重限制了 Agent Memory 系统在国内的可部署性，是 MemSysBench 论文的核心发现之一。


## 下一步

- [ ] 扩大规模：1000条记忆压测（当前仅10条）
- [ ] 修复 Graphiti：尝试 prompt engineering 使其兼容千问
- [ ] 修复 LangMem：patch embedding 调用格式
- [ ] 添加 Letta 系统
- [ ] 并发压测（RQ3）
- [ ] 构建完整 MemDialogue 数据集（基于 LMSYS-Chat-1M）
